"""Порт OverZone Signals PRO на Python — построчный аналог overzone-signals.pine.

Цель одна: получить те же сигналы, что TradingView, но считать их локально,
чтобы прогонять сотни комбинаций параметров и десятки инструментов.

Всё, что в Pine делается встроенными ta.*, здесь реализовано вручную и
ИМЕННО так, как в Pine — иначе цифры разойдутся и переносить выводы будет
некуда. Отличия от Pine, которые не удалось устранить, помечены ОТЛИЧИЕ.
"""
import numpy as np

# ============================================================
# Pine-совместимые TA-функции
# ============================================================


def _first_valid(x):
    idx = np.flatnonzero(~np.isnan(x))
    return int(idx[0]) if len(idx) else len(x)


def sma(x, n):
    """Pine ta.sma: ведущие na пропускаются, отсчёт окна начинается с данных."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    f = _first_valid(x)
    for i in range(f + n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        if not np.isnan(w).any():
            out[i] = w.mean()
    return out


def _seeded_recursive(x, n, alpha):
    """Pine ta.ema/ta.rma: первое значение — sma(n), дальше рекурсия.

    Ведущие na игнорируются, как в Pine: иначе один na в начале ряда
    (например от ta.rsi) обнулил бы весь последующий расчёт.
    """
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    f = _first_valid(x)
    if f + n > len(x):
        return out
    seed = x[f:f + n]
    if np.isnan(seed).any():
        return out
    out[f + n - 1] = seed.mean()
    for i in range(f + n, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1] if not np.isnan(x[i]) else out[i - 1]
    return out


def ema(x, n):
    return _seeded_recursive(np.asarray(x, float), n, 2.0 / (n + 1))


def rma(x, n):
    return _seeded_recursive(np.asarray(x, float), n, 1.0 / n)


def true_range(h, l, c):
    """Pine ta.tr: на первом баре high-low, дальше классический TR."""
    tr = np.empty(len(h))
    tr[0] = h[0] - l[0]
    pc = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    return tr


def atr(h, l, c, n):
    return rma(true_range(h, l, c), n)


def rsi(x, n):
    x = np.asarray(x, float)
    d = np.diff(x, prepend=x[0])
    u = rma(np.maximum(d, 0.0), n)
    dn = rma(np.maximum(-d, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = u / dn
    out = 100.0 - 100.0 / (1.0 + rs)
    out[np.isnan(u)] = np.nan
    out[(dn == 0) & ~np.isnan(u)] = 100.0
    out[(u == 0) & (dn == 0)] = 50.0
    return out


def stdev(x, n):
    """Pine ta.stdev с biased=true — делитель N, не N-1."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        if np.isnan(w).any():
            continue
        out[i] = np.sqrt(((w - w.mean()) ** 2).sum() / n)
    return out


def percentile_nearest_rank(x, n, pct):
    """Pine ta.percentile_nearest_rank: n-й по порядку элемент, ceil(pct/100*N)."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    k = int(np.ceil(pct / 100.0 * n))
    k = min(max(k, 1), n)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        if np.isnan(w).any():
            continue
        out[i] = np.sort(w)[k - 1]
    return out


def crossover(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def crossunder(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.zeros(len(a), bool)
    out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


NEVER = 99999


def barssince(cond):
    """Баров с последнего True, включая текущий (0 если True сейчас)."""
    out = np.full(len(cond), NEVER, int)
    last = -1
    for i, c in enumerate(cond):
        if c:
            last = i
        out[i] = i - last if last >= 0 else NEVER
    return out


def pivot(src, left, right, high=True):
    """Pine ta.pivothigh/pivotlow: значение возвращается через `right` баров.

    ОТЛИЧИЕ: TradingView не документирует поведение при равных значениях.
    Здесь строгое сравнение с обеих сторон. Затрагивает только DIVER.
    """
    src = np.asarray(src, float)
    out = np.full(len(src), np.nan)
    for i in range(left + right, len(src)):
        p = i - right
        v = src[p]
        if np.isnan(v):
            continue
        w = np.concatenate([src[p - left:p], src[p + 1:p + right + 1]])
        if np.isnan(w).any():
            continue
        if (v > w).all() if high else (v < w).all():
            out[i] = v
    return out


def valuewhen(cond, src, occ):
    """Pine ta.valuewhen: значение src на occ-е по счёту срабатывание назад."""
    src = np.asarray(src, float)
    out = np.full(len(src), np.nan)
    hits = []
    for i in range(len(src)):
        if cond[i]:
            hits.append(i)
        if len(hits) > occ:
            out[i] = src[hits[-1 - occ]]
    return out


def supersmoother(src, n):
    """Ehlers SuperSmoother — ядро OverZone.

    Прогрев как в Pine: nz() на первых барах даёт нули, поэтому первые
    несколько сотен баров переходные. Тестовое окно берём с запасом.
    """
    src = np.asarray(src, float)
    pi = np.pi
    a1 = np.exp(-np.sqrt(2) * pi / n)
    b1 = 2 * a1 * np.cos(np.sqrt(2) * pi / n)
    c3 = -a1 ** 2
    c2 = b1
    c1 = 1 - c2 - c3
    ss = np.zeros(len(src))
    for i in range(len(src)):
        p1 = ss[i - 1] if i >= 1 else 0.0
        p2 = ss[i - 2] if i >= 2 else 0.0
        ss[i] = c1 * src[i] + c2 * p1 + c3 * p2
    return ss


# ============================================================
# Параметры (значения по умолчанию из .pine)
# ============================================================

ZN_OFF, ZN_ULTRA, ZN_SMALL, ZN_CENTER, ZN_DEEP = (
    "Отключено", "Сверхмалая", "Малая", "Центральная", "Глубокая")

DEFAULTS = dict(
    zone_global=ZN_SMALL,
    oz_length=200,
    # X
    x_zone=ZN_OFF, x_dev=0.0, x_max_age=20, x_confirm=False,
    x_pf_on=True, x_pf_pct=0.5, x_pf_bars=10, x_sens=0.0,
    # RSI
    rsi_zone=ZN_OFF, rsi_dev=0.0,
    rsi_pf_on=True, rsi_pf_pct=0.3, rsi_pf_bars=5,
    rsi_lvl_mode="Авто по ТФ", rsi_adp_len=300, rsi_adp_pct=10,
    # MACD
    macd_zone=ZN_OFF, macd_dev=0.0,
    macd_pf_on=True, macd_pf_pct=1.0, macd_pf_bars=20, macd_mode="OverZone",
    # DIVER
    div_zone=ZN_OFF, div_dev=0.0,
    div_pf_on=True, div_pf_pct=0.5, div_pf_bars=10,
    div_rng_lo=5, div_rng_hi=60,
    # COMBO
    combo_logic="Требуется два (И)",
    combo_rsi=True, combo_div=True, combo_macd=True,
)


def auto_params(tfm):
    """Блок «АВТОПОДБОР ПАРАМЕТРОВ ПО ТФ» из .pine. tfm — минут в баре."""
    a = {}
    a["rsi_len"] = 7 if tfm <= 5 else 9 if tfm <= 15 else 14 if tfm <= 240 else 21
    a["rsi_ob"] = 85.0 if tfm <= 5 else 80.0 if tfm <= 60 else 70.0
    a["rsi_os"] = 15.0 if tfm <= 5 else 20.0 if tfm <= 60 else 30.0
    a["mac_fast"] = 8 if tfm <= 5 else 10 if tfm <= 15 else 12
    a["mac_slow"] = 17 if tfm <= 5 else 21 if tfm <= 15 else 26
    a["mac_sig"] = 9
    a["div_lbL"] = 5 if tfm <= 60 else 6 if tfm <= 240 else 7
    a["div_lbR"] = 2 if tfm <= 60 else 3
    a["combo_win"] = 5 if tfm <= 5 else 6 if tfm <= 15 else 10 if tfm <= 60 else 12 if tfm <= 240 else 14
    a["zone_mem"] = 3 if tfm <= 5 else 4 if tfm <= 15 else 6 if tfm <= 60 else 8 if tfm <= 240 else 10
    a["x_rsi"] = 7 if tfm <= 5 else 9 if tfm <= 15 else 14 if tfm <= 240 else 21
    a["x_vel"] = 3 if tfm <= 5 else 4 if tfm <= 15 else 5 if tfm <= 60 else 6 if tfm <= 240 else 8
    a["x_smooth"] = 2 if tfm <= 5 else 3 if tfm <= 60 else 4 if tfm <= 240 else 5
    a["x_ob"] = 80.0 if tfm <= 5 else 78.0 if tfm <= 15 else 75.0 if tfm <= 60 else 73.0 if tfm <= 240 else 70.0
    a["x_os"] = 20.0 if tfm <= 5 else 22.0 if tfm <= 15 else 25.0 if tfm <= 60 else 27.0 if tfm <= 240 else 30.0
    return a


# ============================================================
# Расчёт сигналов
# ============================================================

def compute(df, tfm, **over):
    p = dict(DEFAULTS)
    p.update(over)
    a = auto_params(tfm)

    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    hlc3 = (h + l + c) / 3.0
    mintick = 0.1

    # --- OverZone ---
    meanline = supersmoother(hlc3, p["oz_length"])
    meanrange = supersmoother(true_range(h, l, c), p["oz_length"])
    MULT2, GRAD = np.pi * 2.415, 0.5
    up2 = meanline + meanrange * MULT2
    lo2 = meanline - meanrange * MULT2

    thr_ob = {ZN_ULTRA: up2 + meanrange * GRAD * -6, ZN_SMALL: up2 + meanrange * GRAD * -4,
              ZN_CENTER: up2 + meanrange * GRAD * 0, ZN_DEEP: up2 + meanrange * GRAD * 3}
    thr_os = {ZN_ULTRA: lo2 - meanrange * GRAD * -6, ZN_SMALL: lo2 - meanrange * GRAD * -4,
              ZN_CENTER: lo2 - meanrange * GRAD * 0, ZN_DEEP: lo2 - meanrange * GRAD * 3}

    def zone_of(own):
        return p["zone_global"] if own == ZN_OFF else own

    def in_ob(z, dev):
        return np.zeros(n, bool) if z == ZN_OFF else h >= thr_ob[z] * (1 + dev / 100.0)

    def in_os(z, dev):
        return np.zeros(n, bool) if z == ZN_OFF else l <= thr_os[z] * (1 - dev / 100.0)

    def ok(z, dev, mem, long):
        if z == ZN_OFF:
            return np.ones(n, bool)
        return barssince(in_os(z, dev) if long else in_ob(z, dev)) <= mem

    mem = a["zone_mem"]

    # --- Метрика X ---
    pos_n = (c - meanline) / np.maximum(meanrange * MULT2, mintick)
    m_pos = 50 + 50 * np.clip(pos_n, -1.5, 1.5) / 1.5
    m_rsi = rsi(c, a["x_rsi"])
    vel = np.diff(c, prepend=c[0]) / np.maximum(meanrange, mintick)
    vel_s = ema(vel, a["x_vel"])
    vel_sd = stdev(vel_s, 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        vel_z = np.where(vel_sd > 0, vel_s / vel_sd, 0.0)
    m_vel = 50 + 50 * np.clip(np.nan_to_num(vel_z), -2, 2) / 2
    x_osc = ema((m_pos + m_rsi + m_vel) / 3.0, a["x_smooth"])

    ob_lvl = a["x_ob"] + p["x_sens"]
    os_lvl = a["x_os"] - p["x_sens"]
    x_ext_os = x_osc <= os_lvl
    x_ext_ob = x_osc >= ob_lvl

    zx = zone_of(p["x_zone"])
    zx = ZN_SMALL if zx == ZN_OFF else zx
    x_zone_l = ok(zx, p["x_dev"], mem, True)
    x_zone_s = ok(zx, p["x_dev"], mem, False)

    # --- RSI ---
    r = rsi(c, a["rsi_len"])
    if p["rsi_lvl_mode"] == "Адаптивные":
        r_ob = percentile_nearest_rank(r, p["rsi_adp_len"], 100 - p["rsi_adp_pct"])
        r_os = percentile_nearest_rank(r, p["rsi_adp_len"], p["rsi_adp_pct"])
    else:
        r_ob = np.full(n, a["rsi_ob"])
        r_os = np.full(n, a["rsi_os"])
    zr = zone_of(p["rsi_zone"])
    rsi_l_pre = crossover(r, r_os) & ok(zr, p["rsi_dev"], mem, True)
    rsi_s_pre = crossunder(r, r_ob) & ok(zr, p["rsi_dev"], mem, False)

    # --- MACD ---
    if p["macd_mode"] == "OverZone":
        msrc = np.diff(hlc3, prepend=hlc3[0]) / np.maximum(meanrange, mintick)
    else:
        msrc = c
    mline = ema(msrc, a["mac_fast"]) - ema(msrc, a["mac_slow"])
    msig = ema(mline, a["mac_sig"])
    zm = zone_of(p["macd_zone"])
    macd_l_pre = crossover(mline, msig) & ok(zm, p["macd_dev"], mem, True)
    macd_s_pre = crossunder(mline, msig) & ok(zm, p["macd_dev"], mem, False)

    # --- DIVER ---
    lbL, lbR = a["div_lbL"], a["div_lbR"]
    pl = pivot(r, lbL, lbR, high=False)
    ph = pivot(r, lbL, lbR, high=True)
    plF, phF = ~np.isnan(pl), ~np.isnan(ph)
    r_sh = np.concatenate([np.full(lbR, np.nan), r[:-lbR]])
    l_sh = np.concatenate([np.full(lbR, np.nan), l[:-lbR]])
    h_sh = np.concatenate([np.full(lbR, np.nan), h[:-lbR]])

    def in_range(cond):
        prev = np.concatenate([[False], cond[:-1]])
        b = barssince(prev)
        return (b >= p["div_rng_lo"]) & (b <= p["div_rng_hi"])

    rsiHL = (r_sh > valuewhen(plF, r_sh, 1)) & in_range(plF)
    priceLL = l_sh < valuewhen(plF, l_sh, 1)
    zd = zone_of(p["div_zone"])
    div_l_pre = plF & priceLL & rsiHL & ok(zd, p["div_dev"], mem, True)

    rsiLH = (r_sh < valuewhen(phF, r_sh, 1)) & in_range(phF)
    priceHH = h_sh > valuewhen(phF, h_sh, 1)
    div_s_pre = phF & priceHH & rsiLH & ok(zd, p["div_dev"], mem, False)

    # --- Побарный проход: пружина X, фильтры по цене, COMBO ---
    sel_n = int(p["combo_rsi"]) + int(p["combo_div"]) + int(p["combo_macd"])
    need = sel_n if p["combo_logic"] == "Требуются все (И)" else \
        min(2, sel_n) if p["combo_logic"] == "Требуется два (И)" else 1
    cw = a["combo_win"]

    out = {k: np.zeros(n, bool) for k in
           ("x_l", "x_s", "rsi_l", "rsi_s", "macd_l", "macd_s", "div_l", "div_s", "cb_l", "cb_s")}

    os_armed = ob_armed = False
    os_bars = ob_bars = 0
    # состояние фильтров по цене: последняя цена и бар срабатывания
    pf = {k: [np.nan, -10 ** 9] for k in out}
    last = {k: -10 ** 9 for k in ("rsi_l", "rsi_s", "macd_l", "macd_s", "div_l", "div_s")}

    def price_filter(key, raw, on, pct, bars, i):
        px, bi = pf[key]
        ok_ = (not on) or np.isnan(px) or abs(c[i] - px) / px * 100.0 >= pct or (i - bi) >= bars
        res = raw and ok_
        if res:
            pf[key] = [c[i], i]
        return res

    for i in range(n):
        # X: экстремум взводит пружину, выход из него даёт сигнал
        if x_ext_os[i]:
            os_armed, os_bars = True, 0
        elif os_armed:
            os_bars += 1
        if x_ext_ob[i]:
            ob_armed, ob_bars = True, 0
        elif ob_armed:
            ob_bars += 1

        ex_l = os_armed and not x_ext_os[i] and os_bars <= p["x_max_age"] and \
            (c[i] > c[i - 1] if p["x_confirm"] and i else True)
        ex_s = ob_armed and not x_ext_ob[i] and ob_bars <= p["x_max_age"] and \
            (c[i] < c[i - 1] if p["x_confirm"] and i else True)

        xl_pre = ex_l and x_zone_l[i]
        xs_pre = ex_s and x_zone_s[i]

        if ex_l or os_bars > p["x_max_age"]:
            os_armed = False
        if ex_s or ob_bars > p["x_max_age"]:
            ob_armed = False

        out["x_l"][i] = price_filter("x_l", xl_pre, p["x_pf_on"], p["x_pf_pct"], p["x_pf_bars"], i)
        out["x_s"][i] = price_filter("x_s", xs_pre, p["x_pf_on"], p["x_pf_pct"], p["x_pf_bars"], i)

        for key, pre, on, pct, bars in (
                ("rsi_l", rsi_l_pre[i], p["rsi_pf_on"], p["rsi_pf_pct"], p["rsi_pf_bars"]),
                ("rsi_s", rsi_s_pre[i], p["rsi_pf_on"], p["rsi_pf_pct"], p["rsi_pf_bars"]),
                ("macd_l", macd_l_pre[i], p["macd_pf_on"], p["macd_pf_pct"], p["macd_pf_bars"]),
                ("macd_s", macd_s_pre[i], p["macd_pf_on"], p["macd_pf_pct"], p["macd_pf_bars"]),
                ("div_l", div_l_pre[i], p["div_pf_on"], p["div_pf_pct"], p["div_pf_bars"]),
                ("div_s", div_s_pre[i], p["div_pf_on"], p["div_pf_pct"], p["div_pf_bars"])):
            v = price_filter(key, bool(pre), on, pct, bars, i)
            out[key][i] = v
            if v:
                last[key] = i

        # COMBO: голоса засчитываются, если случились не раньше cw баров назад
        vl = (int(p["combo_rsi"] and i - last["rsi_l"] <= cw)
              + int(p["combo_div"] and i - last["div_l"] <= cw)
              + int(p["combo_macd"] and i - last["macd_l"] <= cw))
        vs = (int(p["combo_rsi"] and i - last["rsi_s"] <= cw)
              + int(p["combo_div"] and i - last["div_s"] <= cw)
              + int(p["combo_macd"] and i - last["macd_s"] <= cw))
        out["cb_l"][i] = out["x_l"][i] and sel_n > 0 and vl >= need
        out["cb_s"][i] = out["x_s"][i] and sel_n > 0 and vs >= need

    out["_osc"] = x_osc
    out["_rsi"] = r
    return out


# ============================================================
# Движок бэктеста — построчный аналог bt_step
# ============================================================

def backtest(df, sig, is_long, tp=1.2, sl=1.1, maxbars=25, comm=0.1,
             mode="TP/SL", horizon=20, overlap=False, start=0, collect=False):
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    open_tr = []          # [entry, bar, mae, mfe]
    rets = []             # результат каждой сделки — для оценки значимости
    res = dict(trades=0, wins=0, skipped=0, sum_w=0.0, sum_l=0.0,
               sum_r=0.0, sum_mae=0.0, sum_mfe=0.0, streak=0, maxstk=0)

    for i in range(len(c)):
        for t in list(open_tr):
            fav = (h[i] - t[0]) / t[0] * 100 if is_long else (t[0] - l[i]) / t[0] * 100
            adv = (l[i] - t[0]) / t[0] * 100 if is_long else (t[0] - h[i]) / t[0] * 100
            t[3] = max(t[3], fav)
            t[2] = min(t[2], adv)

            done, ret = False, 0.0
            if mode == "TP/SL":
                # Стоп проверяется РАНЬШЕ тейка — как в .pine, худший вариант
                hit_sl = l[i] <= t[0] * (1 - sl / 100) if is_long else h[i] >= t[0] * (1 + sl / 100)
                hit_tp = h[i] >= t[0] * (1 + tp / 100) if is_long else l[i] <= t[0] * (1 - tp / 100)
                if hit_sl:
                    ret, done = -sl, True
                elif hit_tp:
                    ret, done = tp, True
                elif i - t[1] >= maxbars:
                    ret = (c[i] - t[0]) / t[0] * 100 if is_long else (t[0] - c[i]) / t[0] * 100
                    done = True
            elif i - t[1] >= horizon:
                ret = (c[i] - t[0]) / t[0] * 100 if is_long else (t[0] - c[i]) / t[0] * 100
                done = True

            if done:
                ret -= comm
                rets.append(ret)
                res["trades"] += 1
                res["sum_r"] += ret
                res["sum_mae"] += t[2]
                res["sum_mfe"] += t[3]
                if ret > 0:
                    res["wins"] += 1
                    res["sum_w"] += ret
                    res["streak"] = 0
                else:
                    res["sum_l"] += abs(ret)
                    res["streak"] += 1
                    res["maxstk"] = max(res["maxstk"], res["streak"])
                open_tr.remove(t)

        if sig[i] and i >= start:
            if overlap or not open_tr:
                open_tr.append([c[i], i, 0.0, 0.0])
            else:
                res["skipped"] += 1
    out = stats(res)
    if collect:
        out["rets"] = rets
    return out


def stats(r):
    t, w = r["trades"], r["wins"]
    return dict(
        trades=t,
        wr=w / t * 100 if t else float("nan"),
        pf=r["sum_w"] / r["sum_l"] if r["sum_l"] > 0 else float("nan"),
        exp=r["sum_r"] / t if t else float("nan"),
        avg_w=r["sum_w"] / w if w else float("nan"),
        avg_l=r["sum_l"] / (t - w) if t - w else float("nan"),
        mae=r["sum_mae"] / t if t else float("nan"),
        mfe=r["sum_mfe"] / t if t else float("nan"),
        maxstk=r["maxstk"], skipped=r["skipped"])


# ============================================================
# Режим рынка: трендовый или боковой
# ============================================================
# X — возврат к среднему. Такая логика живёт в боковике и умирает в тренде.
# Три независимые меры одного и того же: если они дадут разный ответ,
# значит меряется шум, а не режим.

def adx(h, l, c, n=14, n_adx=14):
    """Классический ADX Уайлдера. Высокий = сильный тренд."""
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    p_dm = np.where((up > dn) & (up > 0), up, 0.0)
    m_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    trur = rma(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 100 * rma(p_dm, n) / trur
        m = 100 * rma(m_dm, n) / trur
        s = p + m
        dx = 100 * np.abs(p - m) / np.where(s == 0, 1, s)
    return rma(dx, n_adx)


def choppiness(h, l, c, n=14):
    """Индекс choppiness. Высокий (>61.8) = боковик, низкий (<38.2) = тренд."""
    tr = true_range(h, l, c)
    out = np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        rng = h[i - n + 1:i + 1].max() - l[i - n + 1:i + 1].min()
        if rng > 0:
            out[i] = 100 * np.log10(tr[i - n + 1:i + 1].sum() / rng) / np.log10(n)
    return out


def efficiency_ratio(c, n=14):
    """Коэффициент Кауфмана: путь по прямой / пройденный путь.
    Близко к 1 — направленное движение, близко к 0 — топтание."""
    c = np.asarray(c, float)
    move = np.abs(c - np.concatenate([np.full(n, np.nan), c[:-n]]))
    step = np.abs(np.diff(c, prepend=c[0]))
    vol = np.full(len(c), np.nan)
    for i in range(n, len(c)):
        vol[i] = step[i - n + 1:i + 1].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(vol > 0, move / vol, np.nan)


def rolling_rank(x, i, win=500):
    """Процентиль значения x[i] среди последних win баров ДО него.

    Ранг вместо абсолютного порога намеренно: у каждой монеты свой масштаб
    ADX и своя волатильность, а порог «ниже своей медианы» переносится
    между инструментами без подгонки под каждый.
    """
    lo = max(0, i - win)
    w = x[lo:i]
    w = w[~np.isnan(w)]
    if len(w) < 30 or np.isnan(x[i]):
        return np.nan
    return (w < x[i]).sum() / len(w) * 100


# ============================================================
# Выходы по состоянию, а не по времени
# ============================================================
# Фиксированный выход через N баров срезает крупные движения: сделка
# закрывается по расписанию, даже если цена продолжает идти в нужную
# сторону. Здесь три альтернативы, которые закрываются по факту разворота,
# а не по счётчику баров.
#
# У всех есть потолок max_bars — иначе одна незакрытая сделка тянулась бы
# до конца истории и портила статистику.

def backtest_exit(df, sig, is_long, mode="time", horizon=12, trail_mult=2.0,
                  atr_n=14, osc=None, osc_exit=50.0, opp=None, max_bars=200,
                  comm=0.1, overlap=True, start=0, hard_sl=0.0):
    """hard_sl — катастрофический стоп в процентах, 0 = выключен.

    Он нужен не как способ выйти, а как ограничитель хвоста: выходы по
    состоянию ловят движение целиком, но вместе с ним и обвал без дна.
    """
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    a = atr(h, l, c, atr_n)
    open_tr, rets = [], []

    for i in range(len(c)):
        for t in list(open_tr):
            # t = [entry, bar, mae, mfe, best, trail_dist]
            fav = (h[i] - t[0]) / t[0] * 100 if is_long else (t[0] - l[i]) / t[0] * 100
            adv = (l[i] - t[0]) / t[0] * 100 if is_long else (t[0] - h[i]) / t[0] * 100
            t[3] = max(t[3], fav)
            t[2] = min(t[2], adv)
            t[4] = max(t[4], h[i]) if is_long else min(t[4], l[i])

            done, ret = False, 0.0
            age = i - t[1]

            if hard_sl > 0:
                lvl = t[0] * (1 - hard_sl / 100) if is_long else t[0] * (1 + hard_sl / 100)
                if (l[i] <= lvl) if is_long else (h[i] >= lvl):
                    ret, done = -hard_sl, True

            if done:
                pass
            elif mode == "time":
                if age >= horizon:
                    ret, done = (c[i] - t[0]) / t[0] * 100 if is_long else (t[0] - c[i]) / t[0] * 100, True
            elif mode == "trail":
                stop = t[4] - t[5] if is_long else t[4] + t[5]
                hit = l[i] <= stop if is_long else h[i] >= stop
                if hit:
                    # Выход по цене стопа, а не по close: стоп срабатывает внутри бара
                    ret, done = (stop - t[0]) / t[0] * 100 if is_long else (t[0] - stop) / t[0] * 100, True
            elif mode == "osc":
                back = osc[i] >= osc_exit if is_long else osc[i] <= 100 - osc_exit
                if back and age >= 1:
                    ret, done = (c[i] - t[0]) / t[0] * 100 if is_long else (t[0] - c[i]) / t[0] * 100, True
            elif mode == "opp":
                if opp[i] and age >= 1:
                    ret, done = (c[i] - t[0]) / t[0] * 100 if is_long else (t[0] - c[i]) / t[0] * 100, True

            if not done and age >= max_bars:
                ret, done = (c[i] - t[0]) / t[0] * 100 if is_long else (t[0] - c[i]) / t[0] * 100, True

            if done:
                rets.append(ret - comm)
                open_tr.remove(t)

        if sig[i] and i >= start and (overlap or not open_tr):
            dist = nz_(a[i]) * trail_mult
            open_tr.append([c[i], i, 0.0, 0.0, c[i], dist])
    return rets


def nz_(v, alt=0.0):
    return alt if v is None or np.isnan(v) else v


def backtest_struct(df, sig, is_long, swing_n=1, rr=1.0, buf=0.0, max_bars=100,
                    comm=0.1, overlap=True):
    """Стоп за экстремум последних swing_n баров, цель — кратное риску.

    Риск здесь задаёт не волатильность, а сама структура: стоп стоит за
    точкой, ниже которой идея входа перестаёт быть верной. Расстояние до
    него получается разным от сделки к сделке, и цель в R подстраивается
    вместе с ним.

    rr = 0 — цели нет, выход только по стопу или по времени.
    """
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    lo_n = np.array([l[max(0, i - swing_n + 1):i + 1].min() for i in range(len(l))])
    hi_n = np.array([h[max(0, i - swing_n + 1):i + 1].max() for i in range(len(h))])
    open_tr, rets, risks = [], [], []

    for i in range(len(c)):
        for t in list(open_tr):
            entry, bar, stop, tgt = t
            done, ret = False, 0.0
            # Стоп проверяется раньше цели: порядок тиков внутри бара неизвестен
            if (l[i] <= stop) if is_long else (h[i] >= stop):
                ret = (stop - entry) / entry * 100 if is_long else (entry - stop) / entry * 100
                done = True
            elif tgt > 0 and ((h[i] >= tgt) if is_long else (l[i] <= tgt)):
                ret = (tgt - entry) / entry * 100 if is_long else (entry - tgt) / entry * 100
                done = True
            elif i - bar >= max_bars:
                ret = (c[i] - entry) / entry * 100 if is_long else (entry - c[i]) / entry * 100
                done = True
            if done:
                rets.append(ret - comm)
                open_tr.remove(t)

        if sig[i] and (overlap or not open_tr):
            ext = lo_n[i] if is_long else hi_n[i]
            stop = ext * (1 - buf / 100) if is_long else ext * (1 + buf / 100)
            risk = abs(c[i] - stop)
            if risk <= 0:
                continue
            tgt = (c[i] + risk * rr) if is_long else (c[i] - risk * rr)
            open_tr.append([c[i], i, stop, tgt if rr > 0 else 0.0])
            risks.append(risk / c[i] * 100)
    return rets, risks
