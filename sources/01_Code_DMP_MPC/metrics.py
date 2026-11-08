# -*- coding: utf-8 -*-
"""
metrics.py — شاخص‌های ارزیابی (مطابق جدول متریک پروپوزال + الحاقات)
متریک‌های اصلی پروپوزال: RMS شتاب بدنه، RMS کورس فنر، RMS تغییرشکل لاستیک،
RMS نیرو، زمان محاسبه.  الحاقات ما: پیک‌ها، jerk، انرژی اکچویتور، نرخ موفقیت حل‌گر.

نکته (v7): «vdv» در این ماژول روی شتابِ وزن‌نداده محاسبه می‌شود — نسخهٔ
رسمی ISO 2631-1 باید روی a_w (بعد از فیلتر Wk) باشد؛ پیاده‌سازی استاندارد
و تأییدشده در 02_TD3_Suspension_Project/python/iso2631.py موجود است
(weighted_metrics).  «ctrl_dt» آرگومان قدیمی بدون استفاده است و برای
سازگاری API نگه داشته شده.
"""
import numpy as np


def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def peak(x):
    x = np.asarray(x, dtype=float)
    return float(np.max(np.abs(x))) if x.size else 0.0


def summarize(log, ctrl_dt=0.02):
    """خلاصهٔ متریک یک ران؛ log خروجی ClosedLoopSim.run()
    (ctrl_dt بدون استفاده — سازگاری API)"""
    t = log['t']
    dt = float(t[1] - t[0]) if t.size > 1 else 1e-3
    a_s = log['a_s']
    jerk = np.gradient(a_s, dt)
    power = np.abs(log['ua'] * (log['zds'] - log['zdu']))
    # VDV (ISO 2631): دُز ارتعاش حساس به پیک — متریک فراتر از پروپوزال
    vdv = float(np.trapezoid(a_s ** 4, t)) ** 0.25
    out = {
        'a_rms': rms(a_s),
        'a_peak': peak(a_s),
        'vdv': vdv,
        'jerk_rms': rms(jerk),
        'travel_rms': rms(log['travel']),
        'travel_peak': peak(log['travel']),
        'tire_rms': rms(log['tire']),
        'tire_peak': peak(log['tire']),
        'u_rms': rms(log['ua']),
        'u_peak': peak(log['ua']),
        'u_cmd_peak': peak(log['ucmd']),
        'energy_J': float(np.trapezoid(power, t)),
        'mean_power_W': float(np.mean(power)),
    }
    if 'info_modes' in log and len(log['info_modes']) > 0:
        modes = log['info_modes']
        out['mode_mpc_frac'] = float(np.mean([m == 'mpc' for m in modes]))
        out['mode_hold_frac'] = float(np.mean([m == 'hold' for m in modes]))
        out['mode_lqr_frac'] = float(np.mean([m == 'lqr' for m in modes]))
    # v7: removed dead no-op `if hasattr(...): pass` leftover
    return out


def reduction(ctrl_summary, passive_summary, keys=('a_rms', 'travel_rms',
                                                   'tire_rms', 'a_peak',
                                                   'jerk_rms', 'vdv')):
    """درصد کاهش نسبت به حالت کنش‌پذیر"""
    red = {}
    for k in keys:
        base = passive_summary.get(k, 0.0)
        val = ctrl_summary.get(k, 0.0)
        red[k] = float(100.0 * (base - val) / base) if base > 1e-12 else 0.0
    return red


def controller_stats(ctrl):
    """آمار حل‌گر از روی کنترلر"""
    if ctrl is None:
        return {}
    ms = np.asarray(ctrl.solve_ms)
    return {
        'solve_ms_mean': float(np.mean(ms)) if ms.size else 0.0,
        'solve_ms_p95': float(np.percentile(ms, 95)) if ms.size else 0.0,
        'solve_ms_max': float(np.max(ms)) if ms.size else 0.0,
        'success_rate': float(ctrl.n_success / max(ctrl.n_solves, 1)),
        'n_solves': int(ctrl.n_solves),
        'lqr_fallback_steps': int(ctrl.n_fallback_steps),
        'dmp_triggers': int(ctrl.dmp.n_triggers) if ctrl.dmp is not None else 0,
    }
