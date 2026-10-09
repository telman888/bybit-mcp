"""
Bybit public market data as a remote MCP server (READ-ONLY).

- Uses only Bybit's PUBLIC market endpoints: no API keys, no account access, no orders.
- Intended to be added to Claude as a custom connector (Settings -> Connectors).
- Set env var MCP_SECRET to a long random string: the endpoint then becomes
  https://<host>/<MCP_SECRET>/mcp  (obscure path = light protection for a public URL).
"""
import asyncio
import functools
import os
import re
from datetime import datetime, timezone

import httpx
from mcp.server.fastmcp import FastMCP

PORT = int(os.environ.get("PORT", "10000"))
SECRET = os.environ.get("MCP_SECRET", "").strip().strip("/")
PATH = f"/{SECRET}/mcp" if SECRET else "/mcp"
BASES = ["https://api.bybit.com", "https://api.bytick.com"]  # both are official mainnet hosts
UA = "Mozilla/5.0 (compatible; bybit-readonly-mcp/1.0)"

mcp = FastMCP(
    "bybit-public-readonly",
    instructions=(
        "Read-only public market data from Bybit USDT perpetuals (category=linear). "
        "No account access, no orders. Funding rates are per funding interval (see funding_interval_h)."
    ),
    host="0.0.0.0",
    port=PORT,
    streamable_http_path=PATH,
    stateless_http=True,
    json_response=True,
)


# ------------------------------------------------------------------ helpers
def _f(x, nd=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if nd is not None else v


def _iso(ms):
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return None


def _sym(s):
    s = (s or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{2,20}", s):
        raise ValueError("Bad symbol. Example: BTCUSDT")
    return s


async def bybit_get(path, params=None):
    last = "no attempt"
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": UA}) as client:
        for base in BASES:
            try:
                r = await client.get(base + path, params=params or {})
            except httpx.HTTPError as e:
                last = f"{base}: {type(e).__name__}"
                continue
            if r.status_code != 200:
                last = f"{base}: HTTP {r.status_code}"
                continue
            j = r.json()
            if j.get("retCode") != 0:
                raise RuntimeError(f"Bybit error {j.get('retCode')}: {j.get('retMsg')}")
            return j["result"]
    raise RuntimeError(
        f"Bybit unreachable from this server ({last}). HTTP 403 usually means this server's IP/region "
        "is blocked by Bybit (US/China IPs and some cloud ranges are)."
    )


def safe(fn):
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:  # return a readable error instead of crashing the tool call
            return {"error": f"{type(e).__name__}: {e}"}
    return wrapper


def ticker_row(t):
    last, hi, lo = _f(t.get("lastPrice")), _f(t.get("highPrice24h")), _f(t.get("lowPrice24h"))
    bid, ask = _f(t.get("bid1Price")), _f(t.get("ask1Price"))
    pos = round((last - lo) / (hi - lo), 2) if None not in (last, hi, lo) and hi > lo else None
    spread = round((ask - bid) / ((ask + bid) / 2) * 1e4, 2) if bid and ask else None
    return {
        "symbol": t.get("symbol"),
        "last": last,
        "chg24h_pct": round((_f(t.get("price24hPcnt")) or 0) * 100, 2),
        "range24h_pos": pos,  # 0 = at 24h low, 1 = at 24h high
        "turnover24h_usd": round(_f(t.get("turnover24h")) or 0),
        "oi_usd": round(_f(t.get("openInterestValue")) or 0),
        "funding_pct": round((_f(t.get("fundingRate")) or 0) * 100, 4),
        "funding_interval_h": _f(t.get("fundingIntervalHour")),
        "spread_bps": spread,
    }


def summarize_book(bids, asks, depth_pcts=(0.1, 0.25, 0.5, 1.0), top=5):
    bids = [(float(p), float(q)) for p, q in bids]
    asks = [(float(p), float(q)) for p, q in asks]
    if not bids or not asks:
        return {"error": "empty order book"}
    bb, ba = bids[0][0], asks[0][0]
    mid = (bb + ba) / 2
    cov_dn = round((mid - bids[-1][0]) / mid * 100, 3)
    cov_up = round((asks[-1][0] - mid) / mid * 100, 3)
    depth = {}
    for pct in depth_pcts:
        lo, hi = mid * (1 - pct / 100), mid * (1 + pct / 100)
        b = sum(p * q for p, q in bids if p >= lo)
        a = sum(p * q for p, q in asks if p <= hi)
        depth[f"{pct}%"] = {
            "bids_usd": round(b),
            "asks_usd": round(a),
            "imbalance": round((b - a) / (b + a), 3) if (b + a) else None,
            "complete": cov_dn >= pct and cov_up >= pct,
        }

    def big(levels):
        lo, hi = mid * 0.98, mid * 1.02
        lv = sorted(((p, q, p * q) for p, q in levels if lo <= p <= hi), key=lambda x: -x[2])
        return [{"price": p, "qty": q, "usd": round(n), "dist_pct": round((p - mid) / mid * 100, 3)}
                for p, q, n in lv[:top]]

    return {
        "best_bid": bb, "best_ask": ba, "mid": round(mid, 6),
        "spread_bps": round((ba - bb) / mid * 1e4, 2),
        "book_covers_down_pct": cov_dn, "book_covers_up_pct": cov_up,
        "depth_usd_within_pct_of_mid": depth,
        "largest_bids_within_2pct": big(bids),
        "largest_asks_within_2pct": big(asks),
        "note": "Snapshot only; resting orders can be pulled or spoofed. 'complete': false means the fetched book does not reach that distance.",
    }


def summarize_trades(trades, big_usd=100_000):
    rows = []
    for t in trades:
        p, q, ts = _f(t.get("price")), _f(t.get("size")), t.get("time")
        side = t.get("side")
        if p is None or q is None or side not in ("Buy", "Sell"):
            continue
        rows.append((int(ts or 0), side, p, q, p * q))
    if not rows:
        return {"error": "no trades"}
    buy = sum(r[4] for r in rows if r[1] == "Buy")
    sell = sum(r[4] for r in rows if r[1] == "Sell")
    t0, t1 = min(r[0] for r in rows), max(r[0] for r in rows)
    bigs = sorted((r for r in rows if r[4] >= big_usd), key=lambda r: -r[4])[:5]
    return {
        "trades_count": len(rows),
        "window_start_utc": _iso(t0), "window_end_utc": _iso(t1),
        "window_seconds": round((t1 - t0) / 1000),
        "taker_buy_usd": round(buy), "taker_sell_usd": round(sell),
        "delta_usd": round(buy - sell),
        "buy_share": round(buy / (buy + sell), 3) if (buy + sell) else None,
        "big_trades": [{"time_utc": _iso(r[0]), "side": r[1], "price": r[2], "usd": round(r[4])} for r in bigs],
        "note": "Latest trades only (max ~1000), NOT a long CVD history. side = taker side per Bybit docs.",
    }


# ------------------------------------------------------------------ tools
@mcp.tool()
@safe
async def bybit_check() -> dict:
    """Connectivity test: can this server reach Bybit's public API? Run this first. Reports HTTP status per host and this server's outbound IP."""
    out = {}
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": UA}) as client:
        for base in BASES:
            try:
                r = await client.get(base + "/v5/market/time")
                out[base] = {"http": r.status_code, "ok": r.status_code == 200}
            except httpx.HTTPError as e:
                out[base] = {"error": type(e).__name__}
        try:
            out["server_outbound_ip"] = (await client.get("https://api.ipify.org")).text.strip()
        except httpx.HTTPError:
            out["server_outbound_ip"] = "unknown"
    out["hint"] = "HTTP 403 on both hosts = Bybit blocks this server's IP/region; the server must run elsewhere."
    return out


@mcp.tool()
@safe
async def bybit_scan(top_n: int = 40, sort_by: str = "turnover") -> dict:
    """Screen Bybit USDT perpetuals. sort_by: turnover (24h USD volume), change (largest absolute 24h move), oi (open interest USD), funding (largest absolute funding rate). top_n max 100."""
    top_n = max(1, min(int(top_n), 100))
    res = await bybit_get("/v5/market/tickers", {"category": "linear"})
    rows = [ticker_row(t) for t in res.get("list", [])
            if str(t.get("symbol", "")).endswith("USDT") and "-" not in str(t.get("symbol", ""))]
    keys = {
        "turnover": lambda r: -r["turnover24h_usd"],
        "change": lambda r: -abs(r["chg24h_pct"]),
        "oi": lambda r: -r["oi_usd"],
        "funding": lambda r: -abs(r["funding_pct"]),
    }
    if sort_by not in keys:
        raise ValueError("sort_by must be one of: " + ", ".join(keys))
    rows.sort(key=keys[sort_by])
    return {"count_total": len(rows), "sorted_by": sort_by, "rows": rows[:top_n]}


@mcp.tool()
@safe
async def bybit_ticker(symbol: str) -> dict:
    """Current ticker for one USDT perpetual (e.g. BTCUSDT): last/mark/index price, 24h stats, funding, next funding time, open interest."""
    s = _sym(symbol)
    res = await bybit_get("/v5/market/tickers", {"category": "linear", "symbol": s})
    lst = res.get("list", [])
    if not lst:
        raise ValueError(f"No ticker for {s}")
    t = lst[0]
    row = ticker_row(t)
    idx, mark = _f(t.get("indexPrice")), _f(t.get("markPrice"))
    row.update({
        "mark": mark, "index": idx,
        "basis_pct": round((row["last"] - idx) / idx * 100, 4) if idx and row["last"] else None,
        "high24h": _f(t.get("highPrice24h")), "low24h": _f(t.get("lowPrice24h")),
        "open_interest_contracts": _f(t.get("openInterest")),
        "next_funding_utc": _iso(t.get("nextFundingTime")),
    })
    return row


@mcp.tool()
@safe
async def bybit_klines(symbol: str, interval: str = "60", limit: int = 100) -> dict:
    """Candles from Bybit, oldest to newest. interval: 1,3,5,15,30,60,120,240,360,720 (minutes) or D,W,M. limit max 300. Rows: [time_utc, open, high, low, close, volume, turnover_usd]. The last candle may still be forming."""
    s = _sym(symbol)
    if interval not in {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}:
        raise ValueError("Bad interval")
    limit = max(1, min(int(limit), 300))
    res = await bybit_get("/v5/market/kline", {"category": "linear", "symbol": s, "interval": interval, "limit": limit})
    rows = [[_iso(k[0]), _f(k[1]), _f(k[2]), _f(k[3]), _f(k[4]), _f(k[5], 2), round(_f(k[6]) or 0)]
            for k in reversed(res.get("list", []))]
    return {"symbol": s, "interval": interval, "columns": ["time_utc", "open", "high", "low", "close", "volume", "turnover_usd"], "rows": rows}


@mcp.tool()
@safe
async def bybit_funding_history(symbol: str, limit: int = 30) -> dict:
    """Past funding rates (oldest to newest) for a USDT perpetual. limit max 200. Rates are per funding interval, in percent."""
    s = _sym(symbol)
    limit = max(1, min(int(limit), 200))
    res = await bybit_get("/v5/market/funding/history", {"category": "linear", "symbol": s, "limit": limit})
    rows = [{"time_utc": _iso(x.get("fundingRateTimestamp")), "funding_pct": round((_f(x.get("fundingRate")) or 0) * 100, 4)}
            for x in reversed(res.get("list", []))]
    vals = [r["funding_pct"] for r in rows]
    return {"symbol": s, "avg_funding_pct": round(sum(vals) / len(vals), 4) if vals else None, "rows": rows}


@mcp.tool()
@safe
async def bybit_open_interest(symbol: str, interval: str = "1h", limit: int = 48) -> dict:
    """Open interest history (oldest to newest). interval: 5min,15min,30min,1h,4h,1d. limit max 200. OI is in contracts (base coin); oi_usd uses the current last price as an approximation."""
    s = _sym(symbol)
    if interval not in {"5min", "15min", "30min", "1h", "4h", "1d"}:
        raise ValueError("Bad interval")
    limit = max(2, min(int(limit), 200))
    res = await bybit_get("/v5/market/open-interest", {"category": "linear", "symbol": s, "intervalTime": interval, "limit": limit})
    tk = await bybit_get("/v5/market/tickers", {"category": "linear", "symbol": s})
    last = _f((tk.get("list") or [{}])[0].get("lastPrice"))
    rows = [{"time_utc": _iso(x.get("timestamp")), "oi_contracts": _f(x.get("openInterest"))} for x in reversed(res.get("list", []))]
    first, lastv = (rows[0]["oi_contracts"], rows[-1]["oi_contracts"]) if rows else (None, None)
    return {
        "symbol": s, "interval": interval,
        "change_over_window_pct": round((lastv - first) / first * 100, 2) if first and lastv is not None else None,
        "latest_oi_usd_approx": round(lastv * last) if lastv is not None and last else None,
        "rows": rows,
    }


@mcp.tool()
@safe
async def bybit_orderbook(symbol: str, levels: int = 200) -> dict:
    """Order book summary: spread, bid/ask depth in USD within 0.1/0.25/0.5/1% of mid, imbalance, largest resting levels within 2%. levels = book depth to fetch (1-500)."""
    s = _sym(symbol)
    levels = max(1, min(int(levels), 500))
    res = await bybit_get("/v5/market/orderbook", {"category": "linear", "symbol": s, "limit": levels})
    out = summarize_book(res.get("b", []), res.get("a", []))
    out["symbol"], out["book_time_utc"] = s, _iso(res.get("ts"))
    return out


@mcp.tool()
@safe
async def bybit_recent_trades(symbol: str, limit: int = 1000, big_trade_usd: float = 100000) -> dict:
    """Taker flow over the latest trades (up to 1000): taker buy vs sell USD, delta, share, largest prints. Short window, not a long CVD history."""
    s = _sym(symbol)
    limit = max(1, min(int(limit), 1000))
    res = await bybit_get("/v5/market/recent-trade", {"category": "linear", "symbol": s, "limit": limit})
    out = summarize_trades(res.get("list", []), float(big_trade_usd))
    out["symbol"] = s
    return out


@mcp.tool()
@safe
async def bybit_long_short_ratio(symbol: str, period: str = "1h", limit: int = 24) -> dict:
    """Bybit account long/short ratio history (oldest to newest). period: 5min,15min,30min,1h,4h,1d. limit max 500. Account-count based, so it mostly reflects small accounts, not size."""
    s = _sym(symbol)
    if period not in {"5min", "15min", "30min", "1h", "4h", "1d"}:
        raise ValueError("Bad period")
    limit = max(1, min(int(limit), 500))
    res = await bybit_get("/v5/market/account-ratio", {"category": "linear", "symbol": s, "period": period, "limit": limit})
    rows = [{"time_utc": _iso(x.get("timestamp")), "long_ratio": _f(x.get("buyRatio"), 4), "short_ratio": _f(x.get("sellRatio"), 4)}
            for x in reversed(res.get("list", []))]
    return {"symbol": s, "period": period, "rows": rows}


@mcp.tool()
@safe
async def bybit_snapshot(symbol: str) -> dict:
    """One-call overview of a USDT perpetual: ticker, last 8 funding rates, 1h open interest for 24h, order book summary, taker flow. Use this before building a trade plan."""
    s = _sym(symbol)
    parts = await asyncio.gather(
        bybit_ticker(s), bybit_funding_history(s, 8), bybit_open_interest(s, "1h", 24),
        bybit_orderbook(s, 200), bybit_recent_trades(s, 1000),
    )
    names = ["ticker", "funding_last_8", "open_interest_1h_24h", "orderbook", "taker_flow"]
    return dict(zip(names, parts))


if __name__ == "__main__":
    print(f"Serving MCP on 0.0.0.0:{PORT} path={PATH}", flush=True)
    mcp.run(transport="streamable-http")
