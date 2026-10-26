"""
Bybit MCP server: public market data + GUARDED trading.

PART 1 - market data tools (bybit_*): PUBLIC endpoints only, no keys needed.
PART 2 - trading tools (trade_*): only active when BYBIT_API_KEY / BYBIT_API_SECRET are set.
  * default environment is DEMO (api-demo.bybit.com). Real money needs BOTH
    BYBIT_ENV=real AND ALLOW_REAL=yes.
  * trading tools refuse to run unless MCP_SECRET (>=16 chars) is set, because the URL
    is the only password protecting this server.
  * the only way to open a position is a plain PostOnly LIMIT order with a mandatory stop-loss
    and take-profit attached. Size is computed here from the stop distance: the loss if the stop
    is hit is at most 7% of capital, up to 5 positions at once, exchange leverage 5-10x.
    Capital down 10% from its peak = LOSS HALT: no new entries, resting orders are pulled, the
    owner must be notified and decides what happens next.
    Every rule (risk, leverage, max open, loss halt, ATR stop, trend filter, news blackout)
    is checked in code, not in the prompt.
  * kill switch: delete the API key on Bybit, or set TRADING_ENABLED=0 on the host.

Set env var MCP_SECRET to a long random string: the endpoint then becomes
https://<host>/<MCP_SECRET>/mcp  (obscure path = light protection for a public URL).
"""
import asyncio
import functools
import hashlib
import hmac
import json
import math
import os
import re
import statistics
import time
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import Optional
from urllib.parse import urlencode

import httpx
from mcp.server.fastmcp import FastMCP

PORT = int(os.environ.get("PORT", "10000"))
SECRET = os.environ.get("MCP_SECRET", "").strip().strip("/")
PATH = f"/{SECRET}/mcp" if SECRET else "/mcp"
BASES = ["https://api.bybit.com", "https://api.bytick.com"]  # both are official mainnet hosts
UA = "Mozilla/5.0 (compatible; bybit-mcp/2.0)"

mcp = FastMCP(
    "bybit-guarded-trader",
    instructions=(
        "Bybit USDT perpetuals (category=linear). bybit_* tools: public market data. "
        "trade_* tools: guarded trading (plain PostOnly limit entries with mandatory SL/TP, risk-sized by the server; "
        "demo by default). Call trade_status first; call trade_housekeeping at the start of every scheduled run. "
        "Funding rates are per funding interval (see funding_interval_h)."
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


# ====================================================================== PART 2: GUARDED TRADING
class Refused(Exception):
    """A guard said no. Nothing was sent to the exchange."""


class BybitError(RuntimeError):
    """The exchange answered and rejected the request (so the request definitely did not take effect)."""


def _env_f(name, default):
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return float(default)


API_KEY = os.environ.get("BYBIT_API_KEY", "").strip()
API_SECRET = os.environ.get("BYBIT_API_SECRET", "").strip()
TRADE_ENV = os.environ.get("BYBIT_ENV", "demo").strip().lower()            # demo | real
ALLOW_REAL = os.environ.get("ALLOW_REAL", "").strip().lower() == "yes"     # 2nd lock for real money
TRADING_ENABLED = os.environ.get("TRADING_ENABLED", "1").strip() == "1"    # 0 = no NEW entries (cancel/close still work)
TRADE_BASES = {"demo": ["https://api-demo.bybit.com"], "real": BASES}
# Capital is tracked from this date: capital = budget + realized PnL since then + open PnL. Move it forward to RESET the stop after a halt.
TRACKING_START = os.environ.get("TRACKING_START", "2026-10-26").strip()

# --- limits (override with env vars on the host; hard ceilings protect against typos)
ACCOUNT_BUDGET_USD = _env_f("ACCOUNT_BUDGET_USD", 100)       # starting capital: sizing never assumes more than budget + results so far
MAX_RISK_PCT = min(_env_f("MAX_RISK_PCT", 7), 7)              # max loss per trade (stop hit) as % of capital; 7 is also the hard ceiling
RISK_USD = _env_f("RISK_USD", 0)                              # optional extra $ cap per trade, 0 = off
MAX_OPEN = max(1, min(int(_env_f("MAX_OPEN", 5)), 5))         # positions + resting entry orders; 5 is the hard ceiling
MAX_LEVERAGE = max(5.0, min(_env_f("MAX_LEVERAGE", 10), 10))  # exchange leverage is set between MIN and MAX; total notional <= capital * MAX
MIN_LEVERAGE = max(1.0, min(_env_f("MIN_LEVERAGE", 5), MAX_LEVERAGE))
HALT_DRAWDOWN_PCT = _env_f("HALT_DRAWDOWN_PCT", 10)           # capital down this % from its peak = STOP, tell the owner, wait for orders
MAX_LOSSES_PER_DAY = int(_env_f("MAX_LOSSES_PER_DAY", 0))     # optional: pause after N losing closes per UTC day, 0 = off
ORDER_MAX_AGE_H = _env_f("ORDER_MAX_AGE_H", 12)               # unfilled entry orders older than this get cancelled
MIN_TURNOVER_USD = _env_f("MIN_TURNOVER_USD", 5_000_000)      # 24h volume floor
MAX_SPREAD_PCT = _env_f("MAX_SPREAD_PCT", 0.3)
MIN_RR = _env_f("MIN_RR", 1.5)
MAX_STOP_PCT = _env_f("MAX_STOP_PCT", 15)

# --- v8 rule constants
MIN_STOP_ATR = 2.5                       # stop >= 2.5 x ATR(1h), both tactics
ENTRY_MAX_ATR = {1: 1.5, 2: 4.0}         # entry must be within this many ATR(1h) of last price
TREND_MIN_DIST_PCT = 1.0                 # tactic 1: last must be >=1% beyond EMA200(4h) in trade direction
FUNDING_EXTREME_4H_PCT = 0.1             # skip the side that pays >0.1% funding per 4h
FRESH_PEAK_HOURS = 3                     # tactic 2: peak made < 3h ago counts as "still rising"
PEAK_NO_GO_PCT = 10                      # ...and entry must then be >=10% below that peak
ENTRY_STATUSES = ("New", "PartiallyFilled", "Filled", "PartiallyFilledCanceled")

# --- no-trade windows around big macro events (UTC). Add more with env NO_TRADE_WINDOWS="start/end,start/end"
DEFAULT_BLACKOUTS = [
    ("2026-10-28T12:00", "2026-10-28T21:00", "FOMC decision (18:00 UTC)"),
    ("2026-10-29T06:30", "2026-10-29T14:30", "Core PCE (12:30 UTC)"),
    ("2026-11-03T12:00", "2026-11-04T12:00", "US midterm elections"),
]


def _parse_dt(s):
    return datetime.fromisoformat(s.strip().replace("Z", "")).replace(tzinfo=timezone.utc)


def _blackout_label(now):
    wins = [(_parse_dt(a), _parse_dt(b), lab) for a, b, lab in DEFAULT_BLACKOUTS]
    for part in os.environ.get("NO_TRADE_WINDOWS", "").split(","):
        if "/" in part:
            a, b = part.split("/", 1)
            try:
                wins.append((_parse_dt(a), _parse_dt(b), "custom window from NO_TRADE_WINDOWS"))
            except ValueError:
                continue
    for a, b, lab in wins:
        if a <= now <= b:
            return f"{lab} until {b.strftime('%Y-%m-%d %H:%M')} UTC"
    return None


# ---- signed requests
def _sign_request(method, path, params=None, now_ms=None):
    """Build (target, headers, body) for a signed Bybit v5 request. Pure function."""
    params = {k: v for k, v in (params or {}).items() if v is not None}
    recv = "10000"
    ts = str(now_ms or int(time.time() * 1000))
    if method == "GET":
        payload = urlencode(params)
        target, content = path + ("?" + payload if payload else ""), None
    else:
        payload = json.dumps(params, separators=(",", ":"))
        target, content = path, payload
    sig = hmac.new(API_SECRET.encode(), (ts + API_KEY + recv + payload).encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": API_KEY, "X-BAPI-TIMESTAMP": ts, "X-BAPI-RECV-WINDOW": recv,
        "X-BAPI-SIGN": sig, "X-BAPI-SIGN-TYPE": "2", "Content-Type": "application/json",
    }
    return target, headers, content


async def bybit_signed(method, path, params=None, ok_codes=()):
    """Authenticated call. POSTs go to ONE host only (never retried on another host: a retry could double an order)."""
    bases = TRADE_BASES[TRADE_ENV]
    if method != "GET":
        bases = bases[:1]
    last = "no attempt"
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": UA}) as client:
        for base in bases:
            target, headers, content = _sign_request(method, path, params)
            try:
                r = await client.request(method, base + target, headers=headers, content=content)
            except httpx.HTTPError as e:
                last = f"{base}: {type(e).__name__}"
                continue
            if r.status_code != 200:
                last = f"{base}: HTTP {r.status_code}"
                continue
            j = r.json()
            code = j.get("retCode")
            if code in ok_codes:
                return j.get("result") or {}
            if code != 0:
                raise BybitError(f"Bybit {path}: error {code}: {j.get('retMsg')}")
            return j.get("result") or {}
    raise RuntimeError(f"Bybit signed request failed ({last}). HTTP 401/403 = wrong key, wrong environment (demo key on real or vice versa) or IP not whitelisted.")


def _guard_account():
    if TRADE_ENV not in TRADE_BASES:
        raise Refused("BYBIT_ENV must be 'demo' or 'real'.")
    if TRADE_ENV == "real" and not ALLOW_REAL:
        raise Refused("BYBIT_ENV=real but ALLOW_REAL is not 'yes': real-money trading is locked.")
    if not (API_KEY and API_SECRET):
        raise Refused("API keys are not set on the server (BYBIT_API_KEY / BYBIT_API_SECRET).")
    if len(SECRET) < 16:
        raise Refused("MCP_SECRET (16+ characters) must be set on the server before trading tools work; otherwise anyone who finds the URL could trade.")


# ---- numbers
def _q(x, step, mode=ROUND_HALF_UP):
    d, s = Decimal(str(x)), Decimal(str(step))
    return (d / s).to_integral_value(rounding=mode) * s


def _s(d):
    return format(Decimal(d).normalize(), "f")


def _age_h(ms):
    try:
        return round((time.time() * 1000 - int(ms)) / 3_600_000, 1)
    except (TypeError, ValueError):
        return None


def _candles(res, interval_min, now_ms):
    """Bybit returns newest first. Returns (completed, all) oldest->newest as (t, o, h, l, c)."""
    rows = [(int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])) for k in reversed(res.get("list", []))]
    return [r for r in rows if r[0] + interval_min * 60_000 <= now_ms], rows


def _atr(rows, n=14):
    trs = [max(rows[i][2] - rows[i][3], abs(rows[i][2] - rows[i - 1][4]), abs(rows[i][3] - rows[i - 1][4]))
           for i in range(1, len(rows))]
    t = trs[-n:]
    return sum(t) / len(t) if len(t) >= n else None


def _ema(values, n):
    if len(values) < n:
        return None
    k = 2 / (n + 1)
    e = sum(values[:n]) / n
    for v in values[n:]:
        e = v * k + e * (1 - k)
    return e


def _norm_side(side):
    m = {"buy": "Buy", "long": "Buy", "sell": "Sell", "short": "Sell"}
    s = m.get(str(side).strip().lower())
    if not s:
        raise ValueError("side must be Buy/Sell (or long/short)")
    return s


# ---- account state
def _day_start_ms():
    n = datetime.now(timezone.utc)
    return int(n.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)


def _is_entry_order(o):
    return (str(o.get("reduceOnly")).lower() != "true" and str(o.get("closeOnTrigger")).lower() != "true"
            and (o.get("stopOrderType") or "") in ("", "UNKNOWN"))


def _pos_row(p):
    sl, tp = _f(p.get("stopLoss")) or 0, _f(p.get("takeProfit")) or 0
    return {"symbol": p.get("symbol"), "side": p.get("side"), "size": _f(p.get("size")),
            "entry": _f(p.get("avgPrice")), "mark": _f(p.get("markPrice")),
            "upnl_usd": _f(p.get("unrealisedPnl"), 4), "stop_loss": sl or None, "take_profit": tp or None,
            "value_usd": _f(p.get("positionValue"), 2), "leverage": _f(p.get("leverage")),
            "protected": sl > 0, "age_h": _age_h(p.get("createdTime"))}


def _order_row(o):
    return {"order_id": o.get("orderId"), "symbol": o.get("symbol"), "side": o.get("side"),
            "price": _f(o.get("price")), "qty": _f(o.get("qty")), "status": o.get("orderStatus"),
            "age_h": _age_h(o.get("createdTime")), "link_id": o.get("orderLinkId")}


async def _wallet():
    try:
        w = await bybit_signed("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"})
        acc = (w.get("list") or [{}])[0]
        eq, av = _f(acc.get("totalEquity")), _f(acc.get("totalAvailableBalance"))
        if eq is not None:
            return eq, av
    except RuntimeError:
        pass
    w = await bybit_signed("GET", "/v5/account/wallet-balance", {"accountType": "CONTRACT", "coin": "USDT"})
    coins = ((w.get("list") or [{}])[0]).get("coin") or []
    c0 = next((x for x in coins if x.get("coin") == "USDT"), {})
    eq, av = _f(c0.get("equity")), _f(c0.get("availableToWithdraw"))
    if eq is None:
        raise RuntimeError("Could not read account equity")
    return eq, av


def _tracking_start_ms():
    try:
        return int(_parse_dt(TRACKING_START).timestamp() * 1000)
    except ValueError:
        return int((time.time() - 7 * 86400) * 1000)


async def _closed_pnl_since(start_ms):
    """All closed-PnL records since start_ms (Bybit allows max 7 days per request, so walk forward in weekly windows)."""
    out, t, end = [], start_ms, int(time.time() * 1000)
    week = 7 * 86_400_000 - 1000
    for _ in range(60):
        if t >= end:
            break
        e = min(t + week, end)
        cursor = None
        for _ in range(30):
            r = await bybit_signed("GET", "/v5/position/closed-pnl",
                                   {"category": "linear", "startTime": t, "endTime": e, "limit": 100, "cursor": cursor})
            out += r.get("list", [])
            cursor = r.get("nextPageCursor")
            if not cursor:
                break
        t = e + 1
    return out


def _rec_time(x):
    return int(x.get("updatedTime") or x.get("createdTime") or 0)


async def _account_state(need_history=False):
    day0 = _day_start_ms()
    calls = [
        _wallet(),
        bybit_signed("GET", "/v5/position/list", {"category": "linear", "settleCoin": "USDT", "limit": 200}),
        bybit_signed("GET", "/v5/order/realtime", {"category": "linear", "settleCoin": "USDT", "limit": 50}),
        _closed_pnl_since(_tracking_start_ms()),
    ]
    if need_history:
        calls.append(bybit_signed("GET", "/v5/order/history", {"category": "linear", "startTime": day0, "limit": 50}))
    res = await asyncio.gather(*calls)
    (equity, avail), pos, orders, closed_all = res[0], res[1], res[2], res[3]
    positions = [p for p in pos.get("list", []) if (_f(p.get("size")) or 0) > 0]
    entry_orders = [o for o in orders.get("list", []) if _is_entry_order(o)]
    closed = [x for x in closed_all if _rec_time(x) >= day0]
    unreal = sum(_f(p.get("unrealisedPnl")) or 0 for p in positions)
    # capital curve: budget + realized PnL since TRACKING_START; the peak is taken on realized results only
    cap = peak = ACCOUNT_BUDGET_USD
    for x in sorted(closed_all, key=_rec_time):
        cap += _f(x.get("closedPnl")) or 0
        peak = max(peak, cap)
    capital_now = cap + unreal
    drawdown_pct = max(0.0, (peak - capital_now) / peak * 100) if peak > 0 else 0.0
    t2_today = None
    if need_history:
        seen = set()
        for o in res[4].get("list", []) + entry_orders:
            if (str(o.get("orderLinkId", "")).startswith("v8t2-") and o.get("orderStatus") in ENTRY_STATUSES
                    and int(o.get("createdTime") or 0) >= day0):
                seen.add(o.get("orderId"))
        t2_today = len(seen)
    return {
        "equity": equity, "available": avail, "positions": positions, "orders": entry_orders,
        "realized_today": sum(_f(x.get("closedPnl")) or 0 for x in closed),
        "losing_closes": sum(1 for x in closed if (_f(x.get("closedPnl")) or 0) < 0),
        "unreal_open": unreal,
        "capital": capital_now, "peak_capital": peak, "drawdown_pct": drawdown_pct,
        "realized_total": cap - ACCOUNT_BUDGET_USD,
        "t2_entries_today": t2_today,
    }


async def _market_ctx(symbol, tactic):
    now_ms = int(time.time() * 1000)
    k4_limit = 300 if tactic == 1 else 30
    inst_r, tk_r, k1_r, k4_r = await asyncio.gather(
        bybit_get("/v5/market/instruments-info", {"category": "linear", "symbol": symbol}),
        bybit_get("/v5/market/tickers", {"category": "linear", "symbol": symbol}),
        bybit_get("/v5/market/kline", {"category": "linear", "symbol": symbol, "interval": "60", "limit": 100}),
        bybit_get("/v5/market/kline", {"category": "linear", "symbol": symbol, "interval": "240", "limit": k4_limit}),
    )
    if not inst_r.get("list") or not tk_r.get("list"):
        raise ValueError(f"Unknown symbol {symbol}")
    inst, tk = inst_r["list"][0], tk_r["list"][0]
    c1, c1_all = _candles(k1_r, 60, now_ms)
    c4, _ = _candles(k4_r, 240, now_ms)
    window = c1_all[-24:]
    peak = None
    if window:
        i = max(range(len(window)), key=lambda j: window[j][2])
        peak = {"price": window[i][2], "hours_ago": len(window) - 1 - i}
    interval_h = _f(tk.get("fundingIntervalHour")) or 8
    return {
        "inst": inst, "last": _f(tk.get("lastPrice")), "bid": _f(tk.get("bid1Price")), "ask": _f(tk.get("ask1Price")),
        "turnover24h": _f(tk.get("turnover24h")) or 0,
        "funding_4h_pct": (_f(tk.get("fundingRate")) or 0) * 100 * 4 / interval_h,
        "atr1h": _atr(c1), "peak": peak,
        "range4h": statistics.median([r[2] - r[3] for r in c4[-12:]]) if len(c4) >= 6 else None,
        "ema200_4h": _ema([r[4] for r in c4], 200),
    }


# ---- the rulebook (pure function: easy to test, no network)
def check_plan(c, p):
    """c = account + market context, p = plan (entry/sl/tp already rounded to tick). Returns problems, warnings, sizing."""
    problems, warnings = [], []
    side, tactic, short = p["side"], p["tactic"], p["side"] == "Sell"
    entry_d, sl_d, tp_d = Decimal(str(p["entry"])), Decimal(str(p["sl"])), Decimal(str(p["tp"]))
    entry, sl, tp = float(entry_d), float(sl_d), float(tp_d)
    last, bid, ask, atr, inst = c["last"], c["bid"], c["ask"], c["atr1h"], c["inst"]

    if tactic not in (1, 2):
        problems.append("tactic must be 1 or 2")
    if tactic == 2 and not short:
        problems.append("tactic 2 is short-only")
    lab = _blackout_label(c["now"])
    if lab:
        problems.append(f"news blackout: {lab}")

    # instrument and liquidity
    if inst.get("status") != "Trading":
        problems.append(f"instrument status is {inst.get('status')}, not Trading")
    if c["turnover24h"] < MIN_TURNOVER_USD:
        problems.append(f"24h turnover ${c['turnover24h']:,.0f} below minimum ${MIN_TURNOVER_USD:,.0f}")
    if bid and ask and (ask - bid) / ((ask + bid) / 2) * 100 > MAX_SPREAD_PCT:
        problems.append(f"spread {(ask - bid) / ((ask + bid) / 2) * 100:.2f}% above {MAX_SPREAD_PCT}%")

    # account-level limits
    if c["drawdown_pct"] >= HALT_DRAWDOWN_PCT:
        problems.append(f"LOSS HALT: capital is down {c['drawdown_pct']:.1f}% from its peak (limit {HALT_DRAWDOWN_PCT:g}%). No new entries until the owner gives new instructions; notify him.")
    if MAX_LOSSES_PER_DAY and c["losing_closes"] >= MAX_LOSSES_PER_DAY:
        problems.append(f"{c['losing_closes']} losing closes today (max {MAX_LOSSES_PER_DAY}); paused until 00:00 UTC")
    if tactic == 2 and (c.get("t2_entries_today") or 0) >= 1:
        problems.append("tactic 2: only 1 new entry per UTC day, already used")
    syms = {x["symbol"] for x in c["positions"]} | {o["symbol"] for o in c["orders"]}
    if p["symbol"] in syms:
        problems.append("there is already a position or entry order on this symbol")
    n_open = len(c["positions"]) + len(c["orders"])
    if n_open >= MAX_OPEN:
        problems.append(f"{n_open} positions/entry orders already open (max {MAX_OPEN})")
    same_side = (sum(1 for x in c["positions"] if x.get("side") == side)
                 + sum(1 for o in c["orders"] if o.get("side") == side))

    # price logic (plain limit, PostOnly, protective SL and TP on the right sides)
    if short:
        if not entry > last:
            problems.append(f"limit sell {entry} must be ABOVE last price {last}")
        if bid and not entry > bid:
            problems.append(f"limit sell {entry} must be above best bid {bid}")
        if sl <= 0:
            problems.append("stop-loss is mandatory")
        if not sl > entry:
            problems.append("stop-loss must be above entry for a short")
        if not tp < entry:
            problems.append("take-profit must be below entry for a short")
    else:
        if not entry < last:
            problems.append(f"limit buy {entry} must be BELOW last price {last}")
        if ask and not entry < ask:
            problems.append(f"limit buy {entry} must be below best ask {ask}")
        if sl <= 0:
            problems.append("stop-loss is mandatory")
        if not sl < entry:
            problems.append("stop-loss must be below entry for a long")
        if not tp > entry:
            problems.append("take-profit must be above entry for a long")
    sd_d = abs(entry_d - sl_d)
    sd = float(sd_d)

    # ATR-based rules
    if atr is None:
        problems.append("not enough 1h candles to compute ATR")
    else:
        dist_atr = abs(entry - last) / atr
        if tactic in ENTRY_MAX_ATR and dist_atr > ENTRY_MAX_ATR[tactic]:
            problems.append(f"entry is {dist_atr:.1f} ATR(1h) from price (max {ENTRY_MAX_ATR[tactic]} for tactic {tactic})")
        if sd < MIN_STOP_ATR * atr:
            problems.append(f"stop distance {sd:.6g} is below {MIN_STOP_ATR} x ATR(1h) = {MIN_STOP_ATR * atr:.6g}")
    if tactic == 1:
        if c["range4h"] is None:
            problems.append("not enough 4h candles for the typical 4h range")
        elif sd < c["range4h"]:
            problems.append(f"tactic 1: stop distance {sd:.6g} is below the typical 4h candle range {c['range4h']:.6g}")
    if entry > 0 and sd / entry * 100 > MAX_STOP_PCT:
        problems.append(f"stop is {sd / entry * 100:.1f}% from entry (max {MAX_STOP_PCT:g}%)")
    rr = abs(tp - entry) / sd if sd > 0 else 0
    if rr < MIN_RR:
        problems.append(f"reward:risk {rr:.2f} is below {MIN_RR}")

    # tactic 1: trade only with the 4h EMA200 trend
    ema = c["ema200_4h"]
    if tactic == 1:
        if ema is None:
            problems.append("tactic 1: not enough 4h history for EMA200")
        else:
            d = (last - ema) / ema * 100
            if short and d > -TREND_MIN_DIST_PCT:
                problems.append(f"tactic 1 short: price is {d:+.1f}% vs EMA200(4h), needs to be at least {TREND_MIN_DIST_PCT}% below")
            if not short and d < TREND_MIN_DIST_PCT:
                problems.append(f"tactic 1 long: price is {d:+.1f}% vs EMA200(4h), needs to be at least {TREND_MIN_DIST_PCT}% above")

    # funding: do not take the side that pays extreme funding
    f4 = c["funding_4h_pct"]
    if short and f4 <= -FUNDING_EXTREME_4H_PCT:
        problems.append(f"funding {f4:.3f}% per 4h: shorts pay too much")
    if not short and f4 >= FUNDING_EXTREME_4H_PCT:
        problems.append(f"funding {f4:+.3f}% per 4h: longs pay too much")

    # tactic 2: never short within ~10% of a peak that is still rising
    pk = c.get("peak")
    if tactic == 2 and short and pk and pk["hours_ago"] < FRESH_PEAK_HOURS and entry > pk["price"] * (1 - PEAK_NO_GO_PCT / 100):
        problems.append(f"tactic 2: peak {pk['price']} was made {pk['hours_ago']}h ago (still rising); entry is within {PEAK_NO_GO_PCT}% of it")

    # sizing
    sizing = None
    if sd > 0 and entry > 0 and c["equity"] is not None:
        eff_eq = min(c["equity"], c["capital"])
        if eff_eq <= 0:
            problems.append("no capital left")
        cap = eff_eq * MAX_RISK_PCT / 100
        if RISK_USD > 0:
            cap = min(cap, RISK_USD)
        risk = min(p.get("risk_req") or cap, cap)
        stop_pct = sd / entry * 100
        # leverage that still leaves the liquidation price (about 100/lev - 1 % away, isolated margin) beyond the stop with a 20% margin
        lev_cap = min(int(MAX_LEVERAGE), int(100 // (stop_pct * 1.2 + 1.0)))
        if lev_cap < int(MIN_LEVERAGE):
            problems.append(f"stop is {stop_pct:.1f}% away: at {int(MIN_LEVERAGE)}x leverage the position would be liquidated before the stop")
        if same_side >= 1:
            risk = min(risk, cap / 2)
            warnings.append("second trade in the same direction: risk halved")
        used = sum(float(x.get("positionValue") or 0) for x in c["positions"]) + \
            sum((_f(o.get("qty")) or 0) * (_f(o.get("price")) or 0) for o in c["orders"])
        room = eff_eq * MAX_LEVERAGE - used
        # keep at least 20% of the free margin as a buffer (fees, funding, price moves against us)
        margin_room = c["available"] * 0.8 * max(lev_cap, 1) if c["available"] is not None else room
        lf = inst.get("lotSizeFilter", {})
        step, min_qty = Decimal(lf.get("qtyStep", "0.001")), Decimal(lf.get("minOrderQty", "0.001"))
        max_qty = Decimal(lf.get("maxOrderQty") or "1e12")
        min_notional = Decimal(lf.get("minNotionalValue") or "5")
        if risk <= 0:
            problems.append("risk per trade is zero")
        if room <= 0:
            problems.append(f"leverage room used up: open exposure ${used:,.0f} vs max ${eff_eq * MAX_LEVERAGE:,.0f}")
        if margin_room <= 0:
            problems.append("no free margin left")
        cap_notional = max(min(room, margin_room), 0)
        qty = min(Decimal(str(max(risk, 0))) / sd_d, Decimal(str(cap_notional)) / entry_d, max_qty)
        qty = (qty / step).to_integral_value(rounding=ROUND_DOWN) * step
        notional = float(qty * entry_d)
        if qty < min_qty or Decimal(str(notional)) < min_notional:
            problems.append(f"size too small: risk ${risk:.2f} / stop distance {sd:.6g} gives qty {qty} (min {min_qty}, min notional ${min_notional}); stop is too wide for this risk")
        margin_base = min(eff_eq, c["available"]) if c["available"] is not None else eff_eq
        need = math.ceil(notional / (0.8 * margin_base)) if margin_base > 0 else 1
        lev = int(max(int(MIN_LEVERAGE), min(max(lev_cap, 1), need)))
        max_lev = _f((inst.get("leverageFilter") or {}).get("maxLeverage"))
        if max_lev is not None and max_lev < MIN_LEVERAGE:
            problems.append(f"the coin allows only {max_lev:g}x leverage, below the minimum {int(MIN_LEVERAGE)}x")
        elif max_lev is not None and lev > max_lev:
            problems.append(f"needs {lev}x but the coin allows {max_lev:g}x")
        if c["available"] is not None and notional / lev > c["available"] * 0.95:
            problems.append(f"not enough available margin: need ${notional / lev:,.2f}, have ${c['available']:,.2f}")
        if qty >= min_qty and float(qty * sd_d) < 0.7 * risk:
            warnings.append(f"size was cut by the leverage/margin limits: real risk ${float(qty * sd_d):.2f} instead of ${risk:.2f}")
        sizing = {
            "qty": _s(qty), "notional_usd": round(notional, 2), "leverage": lev,
            "risk_usd_if_stopped": round(float(qty * sd_d), 2), "risk_budget_usd": round(risk, 2),
            "reward_usd_if_tp": round(float(qty * abs(tp_d - entry_d)), 2), "rr": round(rr, 2),
            "capital_used_for_sizing_usd": round(eff_eq, 2), "stop_pct": round(stop_pct, 2),
        }
    return {"problems": problems, "warnings": warnings, "sizing": sizing}


# ---- order helpers
async def _find_order(symbol, order_id=None, link_id=None):
    p = {"category": "linear", "symbol": symbol, "orderId": order_id, "orderLinkId": link_id}
    r = await bybit_signed("GET", "/v5/order/realtime", p)
    if r.get("list"):
        return r["list"][0]
    h = await bybit_signed("GET", "/v5/order/history", p)
    return (h.get("list") or [None])[0]


async def _market_close(symbol, side, size):
    opp = "Sell" if side == "Buy" else "Buy"
    return await bybit_signed("POST", "/v5/order/create", {
        "category": "linear", "symbol": symbol, "side": opp, "orderType": "Market", "qty": str(size),
        "reduceOnly": True, "positionIdx": 0,
    })


def _blocked_reasons(a, now):
    out = []
    if not TRADING_ENABLED:
        out.append("TRADING_ENABLED=0 (new entries switched off on the server)")
    lab = _blackout_label(now)
    if lab:
        out.append(f"news blackout: {lab}")
    if a["drawdown_pct"] >= HALT_DRAWDOWN_PCT:
        out.append(f"LOSS HALT: capital is down {a['drawdown_pct']:.1f}% from its peak (limit {HALT_DRAWDOWN_PCT:g}%). "
                   "Send the owner a notification and wait for his instructions; do not place new orders.")
    if MAX_LOSSES_PER_DAY and a["losing_closes"] >= MAX_LOSSES_PER_DAY:
        out.append(f"{a['losing_closes']} losing closes today")
    if len(a["positions"]) + len(a["orders"]) >= MAX_OPEN:
        out.append(f"no free slot ({len(a['positions'])} positions + {len(a['orders'])} orders, max {MAX_OPEN})")
    return out


# ---- tools
@mcp.tool()
@safe
async def trade_status() -> dict:
    """START HERE. Trading environment (demo/real), capital and drawdown, open positions (with stop-loss), resting entry orders, today's PnL, limits, and whether a new entry is allowed right now (blocked_reasons). If loss_halt/notify_owner is true: capital is down 10% from its peak, send the owner a push notification and wait for his instructions. Read-only."""
    try:
        _guard_account()
    except Refused as e:
        return {"env": TRADE_ENV, "ready_to_trade": False, "blocked_reasons": [str(e)],
                "keys_configured": bool(API_KEY and API_SECRET), "mcp_secret_ok": len(SECRET) >= 16}
    a = await _account_state()
    now = datetime.now(timezone.utc)
    blocked = _blocked_reasons(a, now)
    halted = a["drawdown_pct"] >= HALT_DRAWDOWN_PCT
    return {
        "env": TRADE_ENV, "ready_to_trade": not blocked, "blocked_reasons": blocked,
        "loss_halt": halted, "notify_owner": halted,
        "capital": {"capital_usd": round(a["capital"], 2), "peak_usd": round(a["peak_capital"], 2),
                    "drawdown_pct": round(a["drawdown_pct"], 2), "halt_at_drawdown_pct": HALT_DRAWDOWN_PCT,
                    "realized_since_start_usd": round(a["realized_total"], 2), "tracking_since": TRACKING_START,
                    "used_for_sizing_usd": round(min(a["equity"], a["capital"]), 2)},
        "equity_usd": round(a["equity"], 2),
        "available_usd": round(a["available"], 2) if a["available"] is not None else None,
        "today_utc": {"realized_pnl_usd": round(a["realized_today"], 2), "unrealized_open_usd": round(a["unreal_open"], 2),
                      "losing_closes": a["losing_closes"]},
        "positions": [_pos_row(p) for p in a["positions"]],
        "entry_orders": [_order_row(o) for o in a["orders"]],
        "limits": {"max_risk_per_trade_pct": MAX_RISK_PCT, "max_open": MAX_OPEN,
                   "leverage_min": MIN_LEVERAGE, "leverage_max": MAX_LEVERAGE,
                   "entry_order_max_age_h": ORDER_MAX_AGE_H, "starting_capital_usd": ACCOUNT_BUDGET_USD},
        "time_utc": now.strftime("%Y-%m-%d %H:%M"),
    }


@mcp.tool()
@safe
async def trade_place_limit(symbol: str, side: str, entry: float, stop_loss: float, take_profit: float,
                            tactic: int, risk_usd: Optional[float] = None, dry_run: bool = False) -> dict:
    """Place ONE plain PostOnly limit entry (Buy below / Sell above the market) with stop-loss and take-profit attached. The server rounds prices, computes qty from the stop distance (loss if stopped is at most 7% of capital), sets leverage 5-10x, checks every rule and verifies the order is active. tactic: 1 (trend) or 2 (fade shorts). risk_usd can only LOWER the default risk. dry_run=true validates and sizes without sending anything. If any rule fails, nothing is sent and `problems` lists why."""
    _guard_account()
    s, side_n, tactic = _sym(symbol), _norm_side(side), int(tactic)
    if not dry_run and not TRADING_ENABLED:
        raise Refused("TRADING_ENABLED=0: new entries are switched off on the server.")
    acct, mkt = await asyncio.gather(_account_state(need_history=(tactic == 2)), _market_ctx(s, tactic))
    tick = Decimal(mkt["inst"]["priceFilter"]["tickSize"])
    e_d, sl_d, tp_d = _q(entry, tick), _q(stop_loss, tick), _q(take_profit, tick)
    plan = {"symbol": s, "side": side_n, "entry": e_d, "sl": sl_d, "tp": tp_d, "tactic": tactic, "risk_req": risk_usd}
    ctx = {**acct, **mkt, "now": datetime.now(timezone.utc)}
    chk = check_plan(ctx, plan)
    base = {"env": TRADE_ENV, "symbol": s, "side": side_n, "tactic": tactic, "entry": float(e_d),
            "stop_loss": float(sl_d), "take_profit": float(tp_d),
            "context": {"last": mkt["last"], "atr_1h": mkt["atr1h"], "typical_4h_range": mkt["range4h"],
                        "ema200_4h": mkt["ema200_4h"], "funding_per_4h_pct": round(mkt["funding_4h_pct"], 4)}}
    if chk["problems"]:
        return {**base, "placed": False, "problems": chk["problems"], "warnings": chk["warnings"], "sizing": chk["sizing"]}
    sz = chk["sizing"]
    if dry_run:
        return {**base, "placed": False, "dry_run": True, "all_rules_pass": True, "sizing": sz, "warnings": chk["warnings"]}

    link = f"v8t{tactic}-{s}-{int(time.time())}"[:36]
    await bybit_signed("POST", "/v5/position/set-leverage", {
        "category": "linear", "symbol": s, "buyLeverage": str(sz["leverage"]), "sellLeverage": str(sz["leverage"])},
        ok_codes=(110043,))
    body = {
        "category": "linear", "symbol": s, "side": side_n, "orderType": "Limit", "qty": sz["qty"], "price": _s(e_d),
        "timeInForce": "PostOnly", "positionIdx": 0, "reduceOnly": False, "orderLinkId": link,
        "stopLoss": _s(sl_d), "takeProfit": _s(tp_d), "tpslMode": "Full",
        "slTriggerBy": "LastPrice", "tpTriggerBy": "LastPrice",
    }
    try:
        r = await bybit_signed("POST", "/v5/order/create", body)
        order_id = r.get("orderId")
    except BybitError:
        raise  # exchange answered with a definite rejection: nothing was placed
    except RuntimeError:
        found = await _find_order(s, link_id=link)  # network trouble: did the order land anyway?
        if not found:
            raise
        order_id = found.get("orderId")
    await asyncio.sleep(0.7)
    o = await _find_order(s, order_id=order_id)
    status = (o or {}).get("orderStatus")
    active = status in ("New", "PartiallyFilled", "Untriggered")
    after = await _account_state()
    out = {**base, "placed": bool(active or status == "Filled"), "order_id": order_id, "order_link_id": link,
           "qty": sz["qty"], "sizing": sz, "warnings": chk["warnings"],
           "verified": {"order_status": status, "in_active_orders": active, "reject_reason": (o or {}).get("rejectReason"),
                        "open_positions_now": [x["symbol"] for x in after["positions"]],
                        "entry_orders_now": [x["symbol"] for x in after["orders"]]}}
    if not out["placed"]:
        out["note"] = "The exchange did not keep the order (PostOnly orders are cancelled if they would execute immediately). Re-check price and retry next run."
    return out


@mcp.tool()
@safe
async def trade_housekeeping() -> dict:
    """Run at the START of every scheduled run. Cancels entry orders older than ORDER_MAX_AGE_H (default 12h), or ALL resting entry orders during a loss halt, and market-closes any open position that has no stop-loss. Open positions with a stop are left alone. Returns what it did plus the fresh trade_status."""
    _guard_account()
    a = await _account_state()
    cancelled, closed = [], []
    halted = a["drawdown_pct"] >= HALT_DRAWDOWN_PCT
    for o in a["orders"]:
        age = _age_h(o.get("createdTime"))
        if halted or (age is not None and age >= ORDER_MAX_AGE_H):
            try:
                await bybit_signed("POST", "/v5/order/cancel", {"category": "linear", "symbol": o["symbol"], "orderId": o["orderId"]})
                cancelled.append({"symbol": o["symbol"], "order_id": o["orderId"], "age_h": age})
            except Exception as e:
                cancelled.append({"symbol": o["symbol"], "order_id": o["orderId"], "error": str(e)})
    for p in a["positions"]:
        if not (_f(p.get("stopLoss")) or 0) > 0:
            try:
                await _market_close(p["symbol"], p["side"], p["size"])
                closed.append({"symbol": p["symbol"], "reason": "no stop-loss on the position"})
            except Exception as e:
                closed.append({"symbol": p["symbol"], "reason": "no stop-loss", "error": str(e)})
    return {"cancelled_stale_orders": cancelled, "closed_unprotected_positions": closed, "status": await trade_status()}


@mcp.tool()
@safe
async def trade_cancel(symbol: str = "ALL") -> dict:
    """Cancel resting ENTRY orders for one symbol, or ALL. Does not touch open positions or their stops."""
    _guard_account()
    a = await _account_state()
    want = None if symbol.strip().upper() == "ALL" else _sym(symbol)
    out = []
    for o in a["orders"]:
        if want and o["symbol"] != want:
            continue
        try:
            await bybit_signed("POST", "/v5/order/cancel", {"category": "linear", "symbol": o["symbol"], "orderId": o["orderId"]})
            out.append({"symbol": o["symbol"], "order_id": o["orderId"], "cancelled": True})
        except Exception as e:
            out.append({"symbol": o["symbol"], "order_id": o["orderId"], "error": str(e)})
    return {"cancelled": out}


@mcp.tool()
@safe
async def trade_close(symbol: str = "ALL") -> dict:
    """Close a position at market (reduce-only) and cancel that symbol's entry orders. symbol=ALL closes everything and cancels all entry orders (panic button)."""
    _guard_account()
    a = await _account_state()
    want = None if symbol.strip().upper() == "ALL" else _sym(symbol)
    closed, cancelled = [], []
    for p in a["positions"]:
        if want and p["symbol"] != want:
            continue
        try:
            await _market_close(p["symbol"], p["side"], p["size"])
            closed.append({"symbol": p["symbol"], "side": p["side"], "size": p["size"], "closed": True})
        except Exception as e:
            closed.append({"symbol": p["symbol"], "error": str(e)})
    for o in a["orders"]:
        if want and o["symbol"] != want:
            continue
        try:
            await bybit_signed("POST", "/v5/order/cancel", {"category": "linear", "symbol": o["symbol"], "orderId": o["orderId"]})
            cancelled.append({"symbol": o["symbol"], "order_id": o["orderId"]})
        except Exception as e:
            cancelled.append({"symbol": o["symbol"], "error": str(e)})
    return {"closed_positions": closed, "cancelled_orders": cancelled}


@mcp.tool()
@safe
async def trade_set_stop(symbol: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> dict:
    """Change the stop-loss / take-profit of an OPEN position. The stop can only be moved CLOSER to price (tighter), never loosened or removed. Use it to lock in profit or move to breakeven."""
    _guard_account()
    s = _sym(symbol)
    if stop_loss is None and take_profit is None:
        raise ValueError("give stop_loss and/or take_profit")
    r = await bybit_signed("GET", "/v5/position/list", {"category": "linear", "symbol": s})
    pos = next((x for x in r.get("list", []) if (_f(x.get("size")) or 0) > 0), None)
    if not pos:
        raise ValueError(f"no open position on {s}")
    tk = (await bybit_get("/v5/market/instruments-info", {"category": "linear", "symbol": s}))["list"][0]
    tick = Decimal(tk["priceFilter"]["tickSize"])
    short = pos["side"] == "Sell"
    mark, cur_sl = _f(pos.get("markPrice")), _f(pos.get("stopLoss")) or 0
    body = {"category": "linear", "symbol": s, "tpslMode": "Full", "positionIdx": 0}
    if stop_loss is not None:
        new = _q(stop_loss, tick)
        if (short and not new > Decimal(str(mark))) or (not short and not new < Decimal(str(mark))):
            raise Refused(f"stop {new} must be on the loss side of mark price {mark}")
        if cur_sl and ((short and new > Decimal(str(cur_sl))) or (not short and new < Decimal(str(cur_sl)))):
            raise Refused(f"stop can only be tightened: current {cur_sl}, requested {new}")
        body.update(stopLoss=_s(new), slTriggerBy="LastPrice")
    if take_profit is not None:
        new = _q(take_profit, tick)
        if (short and not new < Decimal(str(mark))) or (not short and not new > Decimal(str(mark))):
            raise Refused(f"take-profit {new} must be on the profit side of mark price {mark}")
        body.update(takeProfit=_s(new), tpTriggerBy="LastPrice")
    await bybit_signed("POST", "/v5/position/trading-stop", body, ok_codes=(34040,))
    r2 = await bybit_signed("GET", "/v5/position/list", {"category": "linear", "symbol": s})
    return {"updated": True, "position": _pos_row(next((x for x in r2.get("list", []) if (_f(x.get("size")) or 0) > 0), pos))}


if __name__ == "__main__":
    print(f"Serving MCP on 0.0.0.0:{PORT} path={PATH}", flush=True)
    mcp.run(transport="streamable-http")
