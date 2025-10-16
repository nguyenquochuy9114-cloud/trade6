# smart_tradehub.py
import os
import time
import threading
import logging
import json
from typing import Optional

import requests
import pandas as pd
import numpy as np
import ccxt
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from telegram import Bot

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TESTNET = os.getenv("TESTNET", "false").lower() in ("1","true","yes")
INTERVAL = os.getenv("INTERVAL", "1h")
CHECK_EVERY_SEC = int(os.getenv("CHECK_EVERY_SEC", "900"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.8"))
TOP_N = int(os.getenv("TOP_N", "300"))
PORT = int(os.getenv("PORT", os.getenv("RENDER_PORT", 8000)))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable is required.")
bot = Bot(token=BOT_TOKEN)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_tradehub")

exchange_id = "binance"
exchange_kwargs = {"enableRateLimit": True}
if BINANCE_API_KEY and BINANCE_API_SECRET:
    exchange_kwargs.update({"apiKey": BINANCE_API_KEY, "secret": BINANCE_API_SECRET})
if TESTNET:
    exchange_kwargs['urls'] = {'api': {'public': 'https://testnet.binance.vision/api', 'private': 'https://testnet.binance.vision/api'}}
exchange = getattr(ccxt, exchange_id)(exchange_kwargs)

app = FastAPI(title="SmartTradeHub")

def send_telegram(text: str, chat_id: Optional[str] = None):
    try:
        target = chat_id if chat_id else TELEGRAM_CHAT_ID
        if not target:
            logger.warning("No chat_id provided, cannot send Telegram message.")
            return False
        bot.send_message(chat_id=target, text=text, parse_mode="HTML")
        return True
    except Exception as e:
        logger.exception("Telegram send error: %s", e)
        return False

def fetch_top_n_coins(n=300):
    coins = []
    per_page = 250
    page = 1
    while len(coins) < n:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": per_page, "page": page}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        coins.extend(data)
        page += 1
        if len(data) < per_page:
            break
    coins = coins[:n]
    symbols = [c["symbol"].upper() for c in coins]
    logger.info("Fetched top %d coins from CoinGecko", len(symbols))
    return symbols

def get_binance_spot_symbols():
    try:
        markets = exchange.fetch_markets()
        symbols = set([m['symbol'] if isinstance(m, dict) and 'symbol' in m else (m['base'] + m['quote']) for m in markets])
        return symbols
    except Exception as e:
        logger.exception("fetch_markets error: %s", e)
        return set()

def fetch_klines_ccxt(symbol_ccxt, timeframe=INTERVAL, limit=200):
    try:
        ohlc = exchange.fetch_ohlcv(symbol_ccxt, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlc, columns=['ts','o','h','l','c','v'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        logger.debug("fetch_klines error for %s: %s", symbol_ccxt, e)
        return None

def compute_indicators(df):
    if df is None or len(df) < 50:
        return None
    close = df['c'].astype(float)
    vol = df['v'].astype(float)
    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
    ema20 = EMAIndicator(close, window=20).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    vol_mean = vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else vol.mean()
    last_vol = vol.iloc[-1]
    vol_std = vol.std() if vol.std() != 0 else 1.0
    vol_z = (last_vol - vol_mean) / vol_std
    return {
        "rsi": float(rsi),
        "ema20": float(ema20),
        "ema50": float(ema50),
        "vol_z": float(vol_z),
        "vol_mean": float(vol_mean),
        "last_vol": float(last_vol),
        "close": float(close.iloc[-1]),
    }

def detect_msb_simple(df):
    if df is None or len(df) < 12:
        return 'none'
    highs = df['h'].rolling(5).max()
    lows = df['l'].rolling(5).min()
    try:
        last_high = highs.iloc[-1]
        prev_high = highs.iloc[-6]
        last_low = lows.iloc[-1]
        prev_low = lows.iloc[-6]
    except Exception:
        return 'none'
    if last_high > prev_high and last_low > prev_low:
        return 'bull'
    if last_high < prev_high and last_low < prev_low:
        return 'bear'
    return 'none'

def detect_order_block_simple(df):
    if df is None or len(df) < 40:
        return 'none'
    vol = df['v'].astype(float)
    thr = vol.mean() + vol.std()*1.0
    window = df.tail(40)
    for i in range(len(window)-1, -1, -1):
        row = window.iloc[i]
        if row['v'] > thr:
            if row['c'] < row['o']:
                return 'supply'
            else:
                return 'demand'
    return 'none'

def get_long_short_ratio_binance(symbol_pair):
    try:
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        params = {"symbol": symbol_pair, "period": "5m", "limit": 1}
        r = requests.get(url, params=params, timeout=8)
        j = r.json()
        if not j:
            return None
        ratio = float(j[0].get('longShortRatio', 1.0))
        long_acc = float(j[0].get('longAccount', 0))
        short_acc = float(j[0].get('shortAccount', 0))
        total = long_acc + short_acc if (long_acc + short_acc) > 0 else 1.0
        return ratio, long_acc/total
    except Exception as e:
        logger.debug("LS ratio fetch error: %s", e)
        return None

def score_signal(ind, msb, ob):
    if ind is None:
        return 0.0
    weights = {'rsi':0.15, 'msb':0.25, 'ob':0.20, 'vol':0.15, 'ema':0.15, 'lux':0.0}
    r = ind['rsi']
    if 55 < r < 75:
        r_score = 1.0
    elif 50 < r <= 55:
        r_score = 0.7
    elif r <= 50:
        r_score = 0.3
    else:
        r_score = 0.8
    msb_s = 1.0 if msb == 'bull' else 0.0 if msb == 'bear' else 0.5
    ob_s = 1.0 if ob == 'demand' else 0.0 if ob == 'supply' else 0.5
    vz = ind.get('vol_z', 0)
    vol_s = 1.0 if vz > 1.0 else 0.7 if vz > 0.3 else 0.4
    ema_s = 1.0 if ind['ema20'] > ind['ema50'] else 0.0
    score = r_score*weights['rsi'] + msb_s*weights['msb'] + ob_s*weights['ob'] + vol_s*weights['vol'] + ema_s*weights['ema']
    return max(0.0, min(1.0, score))

def place_spot_market_order_ccxt(symbol_ccxt, side, amount):
    try:
        order = exchange.create_market_order(symbol_ccxt, side, amount)
        return True, order
    except Exception as e:
        logger.exception("Order placement error: %s", e)
        return False, str(e)

def monitor_loop():
    logger.info("Monitor loop started. TOP_N=%d, interval=%s, threshold=%.2f", TOP_N, INTERVAL, SCORE_THRESHOLD)
    all_symbols = fetch_top_n_coins(TOP_N)
    bin_symbols = get_binance_spot_symbols()
    watch_list = []
    for s in all_symbols:
        candidate = s + "USDT"
        if candidate in bin_symbols:
            watch_list.append(candidate)
    logger.info("Monitoring %d pairs on Binance USDT market", len(watch_list))

    while True:
        start = time.time()
        alerts = []
        for pair in watch_list:
            try:
                pair_ccxt = pair[:-4] + "/USDT"
                df = fetch_klines_ccxt(pair_ccxt, timeframe=INTERVAL, limit=200)
                ind = compute_indicators(df)
                if ind is None:
                    continue
                msb = detect_msb_simple(df)
                ob = detect_order_block_simple(df)
                score = score_signal(ind, msb, ob)
                if score >= SCORE_THRESHOLD:
                    lsr = get_long_short_ratio_binance(pair)
                    lsr_txt = f" | L/S ratio: {lsr[0]:.2f} long%:{lsr[1]*100:.1f}%" if lsr else ""
                    msg = (
                        f"📢 <b>{pair}</b> ({INTERVAL})\n"
                        f"Price: {ind['close']:.6f}\n"
                        f"RSI: {ind['rsi']:.2f}\nEMA20/50: {ind['ema20']:.4f}/{ind['ema50']:.4f}\n"
                        f"MSB: {msb} | OB: {ob} | Vol z: {ind['vol_z']:.2f}\n"
                        f"Score: {score*100:.1f}%{lsr_txt}\n"
                        f"Commands: /buy {pair[:-4]} qty  or /status {pair[:-4]}"
                    )
                    alerts.append(msg)
            except Exception as e:
                logger.debug("Error processing %s: %s", pair, e)
                continue
        for a in alerts:
            send_telegram(a)
            time.sleep(0.5)
        elapsed = time.time() - start
        sleep_for = max(5, CHECK_EVERY_SEC - elapsed)
        logger.info("Monitor cycle complete. Alerts=%d. Sleeping %ds", len(alerts), sleep_for)
        time.sleep(sleep_for)

@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=monitor_loop, daemon=True)
    t.start()
    logger.info("Background monitor thread started.")

@app.post("/webhook")
async def tradingview_webhook(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    symbol = data.get("symbol") or data.get("ticker") or "UNKNOWN"
    if ":" in symbol:
        symbol = symbol.split(":")[-1]
    symbol = symbol.upper()

    signal = data.get("signal", "").upper()
    price = float(data.get("price", 0))
    rsi = float(data.get("rsi", 0)) if data.get("rsi") is not None else None
    ema_fast = float(data.get("ema_fast", 0)) if data.get("ema_fast") is not None else None
    ema_slow = float(data.get("ema_slow", 0)) if data.get("ema_slow") is not None else None
    msb = data.get("msb", "none")
    ob = data.get("ob", "none")
    volume = float(data.get("volume", 0)) if data.get("volume") is not None else 0.0

    lsr = get_long_short_ratio_binance(symbol)
    lsr_txt = f"\nL/S ratio: {lsr[0]:.2f} | Long%: {lsr[1]*100:.1f}%" if lsr else ""

    trend = "📈 Bullish" if ema_fast and ema_slow and ema_fast > ema_slow and (rsi is None or rsi > 50) else "📉 Bearish"

    msg = (
        f"⚡ <b>TradingView Alert</b>\nSymbol: {symbol}\nSignal: {signal}\nTrend: {trend}\n"
        f"Price: {price}\nRSI: {rsi if rsi is not None else 'N/A'}\nEMA20/50: {ema_fast if ema_fast else 'N/A'}/{ema_slow if ema_slow else 'N/A'}\n"
        f"MSB: {msb} | OB: {ob}\nVolume: {volume}{lsr_txt}\n"
        f"Commands: /buy {symbol.replace('USDT','')} qty   or /status {symbol.replace('USDT','')}"
    )

    send_telegram(msg)
    return JSONResponse({"status":"ok"})

@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    message = data.get("message") or data.get("edited_message")
    if not message:
        return JSONResponse({"status":"no_message"})

    chat = message.get("chat", {})
    chat_id = str(chat.get("id"))
    text = message.get("text", "").strip()
    from_user = message.get("from", {}).get("username", "")
    logger.info("Telegram message from %s (%s): %s", from_user, chat_id, text)

    if text.startswith("/start"):
        send_telegram("SmartTradeHub: Ready. Use /buy SYMBOL qty | /sell SYMBOL qty | /status SYMBOL", chat_id)
        return JSONResponse({"status":"ok"})

    if text.startswith("/status"):
        try:
            parts = text.split()
            if len(parts) < 2:
                send_telegram("Usage: /status SYMBOL (e.g., /status BNB)", chat_id)
                return JSONResponse({"status":"ok"})
            sym = parts[1].upper()
            pair_ccxt = f"{sym}/USDT"
            df = fetch_klines_ccxt(pair_ccxt, timeframe=INTERVAL, limit=200)
            ind = compute_indicators(df)
            if ind is None:
                send_telegram(f"Không đủ dữ liệu cho {sym}/USDT", chat_id)
                return JSONResponse({"status":"ok"})
            msb = detect_msb_simple(df)
            ob = detect_order_block_simple(df)
            lsr = get_long_short_ratio_binance(sym + "USDT")
            score = score_signal(ind, msb, ob)
            txt = (
                f"<b>{sym}/USDT ({INTERVAL})</b>\nPrice: {ind['close']:.6f}\n"
                f"RSI: {ind['rsi']:.2f}\nEMA20/50: {ind['ema20']:.4f}/{ind['ema50']:.4f}\n"
                f"MSB: {msb} | OB: {ob} | Vol z: {ind['vol_z']:.2f}\nScore: {score*100:.1f}%\n"
            )
            if lsr:
                txt += f"Long/Short Ratio: {lsr[0]:.2f} | Long%: {lsr[1]*100:.1f}%\n"
            send_telegram(txt, chat_id)
            return JSONResponse({"status":"ok"})
        except Exception as e:
            logger.exception("status command error: %s", e)
            send_telegram(f"Error on /status: {e}", chat_id)
            return JSONResponse({"status":"error"})

    if text.startswith("/buy") or text.startswith("/sell"):
        try:
            parts = text.split()
            if len(parts) < 3:
                send_telegram("Usage: /buy SYMBOL qty  (e.g., /buy BNB 0.5)", chat_id)
                return JSONResponse({"status":"ok"})
            sym = parts[1].upper()
            qty = float(parts[2])
            pair_ccxt = f"{sym}/USDT"
            side = 'buy' if text.startswith("/buy") else 'sell'
            if not BINANCE_API_KEY or not BINANCE_API_SECRET:
                send_telegram("API keys not configured on server. Cannot place order.", chat_id)
                return JSONResponse({"status":"no_api"})
            ok, resp = place_spot_market_order_ccxt(pair_ccxt, side, qty)
            if ok:
                send_telegram(f"✅ {side.upper()} order placed for {sym} qty {qty}\n{resp}", chat_id)
            else:
                send_telegram(f"❌ Order failed: {resp}", chat_id)
            return JSONResponse({"status":"ok"})
        except Exception as e:
            logger.exception("Order command error: %s", e)
            send_telegram(f"Error processing order: {e}", chat_id)
            return JSONResponse({"status":"error"})

    send_telegram("Unknown command. Use /status /buy /sell", chat_id)
    return JSONResponse({"status":"ok"})

@app.get("/health")
async def health():
    return {"status":"ok"}

if __name__ == "__main__":
    uvicorn.run("smart_tradehub:app", host="0.0.0.0", port=PORT, reload=False)
