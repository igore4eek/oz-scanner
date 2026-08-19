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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import requests

import oz

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "state.json")
LOG = os.path.join(HERE, "signals.csv")
CFG = os.path.join(HERE, "config.json")

BINANCE = "https://fapi.binance.com/fapi/v1/klines"
BYBIT = "https://api.bybit.com/v5/market/kline"
HYPERLIQUID = "https://api.hyperliquid.xyz/info"
TG = "https://api.telegram.org/bot{}/sendMessage"

# Длительность бара в миллисекундах — Hyperliquid просит диапазон времени,
# а не количество свечей
TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
         "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

# Свечей на запрос. Supersmoother с периодом 200 успокаивается примерно за
# 320 баров — с запасом хватает, и укладывается в один запрос к обеим биржам
# (у Bybit максимум 1000).
BARS = 1000

# Bybit ждёт интервал числом минут, Binance — строкой вида "15m"
BYBIT_TF = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "1d": "D"}

# Сколько закрытых баров просматривать назад. GitHub не только задерживает
# запуски, но и выбрасывает часть: наблюдались провалы до 45 минут подряд.
# Восемь баров M15 — это два часа, вчетверо больше худшего провала, так что
# пропущенный запуск сигнал не теряет.
LOOKBACK = 8


def load_json(path, default):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default


def load_env():
    """Локальный запуск берёт токены из .env рядом со скриптом.

    В облаке их подставляет GitHub из секретов, поэтому файла там нет и
    отсутствие его — не ошибка.
    """
    p = os.path.join(HERE, ".env")
    if not os.path.exists(p):
        return
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def local_is_alive(state, minutes):
    """Свежесть state.json — это признак жизни компьютера-основного сканера.

    Он толкает состояние в репозиторий после каждого прогона, поэтому если
    последний обработанный бар свежий, значит компьютер работает и уже всё
    отправил. Облаку в этот момент делать нечего: иначе придут дубли.
    """
    if not state:
        return False
    newest = max(state.values()) / 1000
    return (time.time() - newest) < minutes * 60


def _pack(rows):
    """rows: [[время_мс, o, h, l, c, v], ...] по возрастанию времени.

    Последняя свеча ещё формируется и до закрытия изменится — считать по ней
    сигналы нельзя, поэтому она отбрасывается.
    """
    rows = rows[:-1]
    return dict(
        time=np.array([int(x[0]) for x in rows]),
        open=np.array([float(x[1]) for x in rows]),
        high=np.array([float(x[2]) for x in rows]),
        low=np.array([float(x[3]) for x in rows]),
        close=np.array([float(x[4]) for x in rows]),
        volume=np.array([float(x[5]) for x in rows]))


def _binance(symbol, interval):
    r = requests.get(BINANCE, timeout=30,
                     params={"symbol": symbol, "interval": interval, "limit": BARS})
    r.raise_for_status()
    return _pack(r.json())


def _bybit(symbol, interval):
    r = requests.get(BYBIT, timeout=30,
                     params={"category": "linear", "symbol": symbol,
                             "interval": BYBIT_TF.get(interval, "15"), "limit": BARS})
    r.raise_for_status()
    d = r.json()
    if d.get("retCode") != 0:
        raise RuntimeError(f"bybit retCode={d.get('retCode')} {d.get('retMsg')}")
    # Bybit отдаёт от новых к старым — разворачиваем
    return _pack(list(reversed(d["result"]["list"])))


def _hyperliquid(symbol, interval):
    # Hyperliquid называет монеты коротко: BTCUSDT -> BTC
    coin = symbol[:-4] if symbol.endswith("USDT") else symbol
    ms = TF_MS.get(interval, 15 * 60_000)
    now = int(time.time() * 1000)
    r = requests.post(HYPERLIQUID, timeout=30,
                      json={"type": "candleSnapshot",
                            "req": {"coin": coin, "interval": interval,
                                    "startTime": now - BARS * ms, "endTime": now}})
    r.raise_for_status()
    d = r.json()
    if not isinstance(d, list) or not d:
        raise RuntimeError(f"hyperliquid: нет данных по {coin}")
    return _pack([[x["t"], x["o"], x["h"], x["l"], x["c"], x["v"]] for x in d])


# Централизованные биржи отдают 451 с американских адресов, а раннеры GitHub
# Actions именно там: и Binance, и Bybit оттуда недоступны. Hyperliquid —
# децентрализованная биржа без геоблокировок, поэтому она замыкает список и
# работает как гарантированный запасной вариант.
SOURCES = [("binance", _binance), ("bybit", _bybit), ("hyperliquid", _hyperliquid)]


def fetch(symbol, interval, tries=2):
    """Повтор нужен из-за придерживания на стороне биржи при параллельных
    запросах: разовый сбой иначе выбросил бы монету из журнала за этот бар."""
    errs = []
    for attempt in range(tries):
        for name, fn in SOURCES:
            try:
                d = fn(symbol, interval)
                if len(d["close"]) >= 600:
                    return d, name
                errs.append(f"{name}: только {len(d['close'])} баров")
            except Exception as e:
                errs.append(f"{name}: {str(e)[:60]}")
        if attempt + 1 < tries:
            time.sleep(2)
    raise RuntimeError("; ".join(errs[-3:]))


def fetch_all(symbols, interval, workers=8):
    """Загрузка идёт параллельно: она упирается в сеть, а не в процессор.

    Один запрос к бирже занимает секунды, и последовательно полсотни монет
    тянулись бы минуты. Расчёт сигналов на их фоне почти ничего не стоит —
    около 0.1 с на монету.
    """
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch, s, interval): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                out[sym] = (fut.result(), None)
            except Exception as e:
                out[sym] = (None, e)
    return out


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
    load_env()
    cfg = load_json(CFG, {})
    symbols = cfg.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    interval = cfg.get("interval", "15m")
    tfm = cfg.get("tfm", 15)
    swing = cfg.get("stop_swing_bars", 10)
    rr = cfg.get("target_r", 1.0)
    hold = cfg.get("max_hold_bars", 100)
    x_dev = cfg.get("x_dev", 0.3)

    state = load_json(STATE, {})

    # Резервный режим: облако молчит, пока основной сканер на компьютере жив
    fresh = int(os.getenv("SKIP_IF_FRESH_MINUTES", "0"))
    if fresh and local_is_alive(state, fresh):
        newest = datetime.fromtimestamp(max(state.values()) / 1000, timezone.utc)
        print(f"основной сканер жив — последний бар {newest:%H:%M UTC}, "
              f"свежее {fresh} мин. Резерв не нужен, выхожу.")
        return 0

    new_rows, msgs = [], []
    ok_syms = failed = 0

    loaded = fetch_all(symbols, interval)

    for sym in symbols:
        res, err = loaded.get(sym, (None, RuntimeError("нет результата")))
        if err is not None:
            print(f"{sym:12} загрузка не удалась — {err}")
            failed += 1
            continue
        d, src = res
        ok_syms += 1

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
                    source=src,
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
        print(f"{sym:12} {src:8} баров {n:>5}  новых сигналов {found}")

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

    print(f"\nмонет обработано: {ok_syms}, не удалось: {failed}, "
          f"новых сигналов: {len(new_rows)}")

    # Молчаливый «успех», при котором не загрузилась ни одна монета, хуже
    # явной ошибки: сканер выглядел бы работающим, ничего при этом не делая.
    # Именно так прошёл первый запуск — зелёная галка и пустой state.json.
    if ok_syms == 0:
        print("ОШИБКА: ни одна монета не загрузилась")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
