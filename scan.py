"""Сканер сигналов OverZone: считает сам, шлёт в Telegram, ведёт журнал.

TradingView здесь не участвует — расчёт идёт по тому же коду, что сверен с
ним бар в бар. Поэтому нет ни лимита на число алертов, ни платного вебхука:
монет может быть сколько угодно.

Журнал важнее оповещений. Каждый сигнал записывается в signals.csv в момент
появления, вместе с ценой входа, стопом и целью. Через месяц это будет
настоящая форвардная выборка — данные, собранные вперёд, а не нарезанные из
прошлого. У неё нет ни подгонки, ни ошибки выжившего, ни сомнений в том, что
мы подсматривали будущее.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

import oz

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
LOG = os.path.join(HERE, "signals.csv")
CFG = os.path.join(HERE, "config.json")

KLINES = "https://fapi.binance.com/fapi/v1/klines"
TG = "https://api.telegram.org/bot{}/sendMessage"

# Свечей на запрос. Supersmoother с периодом 200 успокаивается примерно за
# 320 баров, так что запас более чем достаточный, а укладываемся в один запрос.
BARS = 1500

# Сколько закрытых баров просматривать назад. Расписание GitHub Actions
# нередко задерживается, и одного бара мало: пропустили запуск — потеряли
# сигнал. С запасом сигнал всё равно уйдёт, просто позже.
LOOKBACK = 4


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def fetch(symbol, interval):
    r = requests.get(KLINES, params={"symbol": symbol, "interval": interval,
                                     "limit": BARS}, timeout=30)
    r.raise_for_status()
    k = r.json()
    # Последняя свеча ещё формируется — сигналы по ней считать нельзя,
    # она изменится до закрытия.
    k = k[:-1]
    return dict(
        time=np.array([int(x[0]) for x in k]),
        open=np.array([float(x[1]) for x in k]),
        high=np.array([float(x[2]) for x in k]),
        low=np.array([float(x[3]) for x in k]),
        close=np.array([float(x[4]) for x in k]),
        volume=np.array([float(x[5]) for x in k]))


class Frame:
    """Минимальная замена DataFrame — oz.compute обращается только к df[key]."""

    def __init__(self, d):
        self.d = d

    def __getitem__(self, k):
        return _Col(self.d[k])

    def __len__(self):
        return len(self.d["close"])


class _Col:
    def __init__(self, a):
        self.a = a

    def to_numpy(self, dtype=float):
        return self.a.astype(dtype)


def send(text):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TG_CHAT_ID")
    if not token or not chat:
        print("  telegram не настроен — пропускаю отправку")
        return False
    r = requests.post(TG.format(token), timeout=20,
                      json={"chat_id": chat, "text": text,
                            "parse_mode": "HTML", "disable_web_page_preview": True})
    if not r.ok:
        print(f"  telegram ошибка {r.status_code}: {r.text[:150]}")
    return r.ok


def main():
    cfg = load_json(CFG, {})
    symbols = cfg.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    interval = cfg.get("interval", "15m")
    tfm = cfg.get("tfm", 15)
    swing = cfg.get("stop_swing_bars", 10)
    rr = cfg.get("target_r", 1.0)
    hold = cfg.get("max_hold_bars", 100)
    x_dev = cfg.get("x_dev", 0.3)

    state = load_json(STATE, {})
    new_rows, msgs = [], []

    for sym in symbols:
        try:
            d = fetch(sym, interval)
        except Exception as e:
            print(f"{sym:12} загрузка не удалась: {str(e)[:60]}")
            continue
        if len(d["close"]) < 600:
            print(f"{sym:12} мало истории ({len(d['close'])})")
            continue

        sig = oz.compute(Frame(d), tfm=tfm, x_dev=x_dev)
        n = len(d["close"])
        last_seen = int(state.get(sym, 0))
        found = 0

        for i in range(max(0, n - LOOKBACK), n):
            bar_ms = int(d["time"][i])
            if bar_ms <= last_seen:
                continue
            for key, side, is_long in (("x_l", "LONG", True), ("x_s", "SHORT", False)):
                if not sig[key][i]:
                    continue
                entry = float(d["close"][i])
                lo = float(d["low"][max(0, i - swing + 1):i + 1].min())
                hi = float(d["high"][max(0, i - swing + 1):i + 1].max())
                stop = lo if is_long else hi
                risk = abs(entry - stop)
                if risk <= 0:
                    continue
                target = entry + risk * rr if is_long else entry - risk * rr
                combo = bool(sig["cb_l"][i] if is_long else sig["cb_s"][i])
                ts = datetime.fromtimestamp(bar_ms / 1000, timezone.utc)

                new_rows.append(dict(
                    bar_time=ts.strftime("%Y-%m-%d %H:%M"), bar_ms=bar_ms,
                    symbol=sym, side=side, interval=interval, combo=int(combo),
                    entry=f"{entry:.8g}", stop=f"{stop:.8g}", target=f"{target:.8g}",
                    risk_pct=f"{risk / entry * 100:.3f}", max_hold=hold,
                    logged_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")))

                emo = "🟢" if is_long else "🔴"
                msgs.append(
                    f"{emo} <b>{side} {sym.replace('USDT', '')}</b>"
                    f"{'  ⭐COMBO' if combo else ''}\n"
                    f"<code>{interval}</code> · бар {ts.strftime('%H:%M UTC')}\n"
                    f"вход <b>{entry:.8g}</b>\n"
                    f"стоп {stop:.8g}  ({risk / entry * 100:.2f}%)\n"
                    f"цель {target:.8g}  ({rr:g}R)\n"
                    f"предохранитель: {hold} баров")
                found += 1

        state[sym] = int(d["time"][n - 1])
        print(f"{sym:12} баров {n:>5}  новых сигналов {found}")

    if new_rows:
        exists = os.path.exists(LOG)
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(new_rows[0]))
            if not exists:
                w.writeheader()
            w.writerows(new_rows)
        # Telegram ограничивает длину сообщения — шлём пачками
        for i in range(0, len(msgs), 5):
            send("\n\n".join(msgs[i:i + 5]))
            time.sleep(0.5)

    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)

    print(f"\nитого новых сигналов: {len(new_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
