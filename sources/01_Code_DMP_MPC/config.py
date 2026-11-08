# -*- coding: utf-8 -*-
"""
config.py — پیکربندی مرکزی پروژهٔ DMP + MPC برای تعلیق فعال Quarter-Car
پارامترهای گیاه دقیقاً مطابق پروپوزال: ms=290, mu=59, ks=16800, cs=1000, kt=190000, umax=3000
همهٔ ارقام فیزیکی در SI هستند.
"""
from dataclasses import dataclass, field, asdict
import json

__version__ = '8.2.0'   # هم‌گام با VERSION ریشه (ببین: check_version.py)


# ---------------------------------------------------------------------------
# 1) گیاه (مدل یک‌چهارم خودرو + دینامیک اکچویتور)
# ---------------------------------------------------------------------------
@dataclass
class PlantParams:
    ms: float = 290.0          # جرم بدنه [kg]
    mu: float = 59.0           # جرم چرخ [kg]
    ks: float = 16800.0        # سختی فنر [N/m]
    cs: float = 1000.0         # میرایی کنش‌پذیر داخلی [N.s/m]
    kt: float = 190000.0       # سختی لاستیک [N/m]
    umax: float = 3000.0       # حد نیروی اکچویتور [N]
    zlim: float = 0.08         # حد کورس فنر [m]
    tau_act: float = 0.015     # ثابت زمانی اکچویتور (تأخیر مرتبهٔ اول) [s]  <-- فراتر از پروپوزال

    @property
    def f_body(self) -> float:
        """فرکانس طبیعی مود بدنه [Hz]"""
        import numpy as np
        return np.sqrt(self.ks / self.ms) / (2 * np.pi)

    @property
    def f_wheel(self) -> float:
        """فرکانس مود پرش چرخ [Hz]"""
        import numpy as np
        return np.sqrt((self.ks + self.kt) / self.mu) / (2 * np.pi)


# ---------------------------------------------------------------------------
# 2) کنترلر NMPC  (CasADi / IPOPT)
# ---------------------------------------------------------------------------
@dataclass
class MpcConfig:
    dt_mpc: float = 0.01       # گام کنترل/پیش‌بینی [s]  (100Hz — ارتقا v2)
    N: int = 100               # طول افق → 1.0s  (پروپوزال: 20×0.01=0.2s)
    # --- وزن‌های هزینه (پس از ریزتنظیم مختصاتی tune4 — نسخهٔ v3) ---
    q_r: float = 0.3           # ردیابی مرجع DMP در حالت رخداد (فعال/در انتظار)
    q_r_idle: float = 0.52     # ردیابی مرجع در حالت بی‌کار (مرجع≈صفر)
    q_rv: float = 0.0          # ردیابی سرعت مرجع (اختیاری)
    q_a: float = 32.0          # شتاب بدنه (آسایش — وزن غالب ارتقایافته v8.3)
    q_s: float = 2.2           # کورس فنر
    q_t: float = 4.0           # تغییرشکل لاستیک
    r_u: float = 0.035         # نیرو (بهینه‌سازی جهت پاسخ سریع‌تر)
    r_du: float = 0.08         # همواری دستور نیروی اکچویتور
    rho_susp: float = 5e4      # جریمهٔ اسلاک قید نرم کورس فنر
    # --- مقیاس‌های فیزیکی برای نرمال‌سازی هزینه (رفع نقص شمارهٔ ۶ پروپوزال) ---
    s_r: float = 0.02          # [m]
    s_a: float = 2.0           # [m/s^2]
    s_s: float = 0.04          # [m]  (نصف حد کورس)
    s_t: float = 0.01          # [m]
    s_u: float = 1000.0        # [N]
    # --- اکچویتور داخل مدل MPC ---
    tau_act_model: float = 0.015
    # --- IPOPT ---
    max_iter: int = 100
    tol: float = 1e-5
    acceptable_tol: float = 1e-3
    # --- پایدارساز و ایمنی ---
    use_terminal: bool = True  # هزینهٔ ترمینال DARE
    fallback_after: int = 3    # تعداد شکست پیوسته → حالت LQR ایمن
    use_dmp: bool = True       # خاموش = خط پایهٔ MPC-only (سناریوی B)


# ---------------------------------------------------------------------------
# 3) DMP — مولد مرجع پیش‌نگر رویداد-محور
# ---------------------------------------------------------------------------
@dataclass
class DmpConfig:
    n_basis: int = 40          # تعداد توابع پایهٔ گاوسی (پوشش فازی متراکم)
    alpha_s: float = 4.0       # سرعت افت فاز
    alpha_z: float = 25.0
    beta_z: float = 6.25       # بحرانی‌میرا (beta = alpha/4)
    k_shape: float = 0.35      # v7: LEGACY/unused — dmp.py derives the
                               # reference height from k_relief·c_exc·zlim.
                               # Kept only for old diagnostics/tune scripts
                               # that still pass it to DmpConfig(...).
    stretch: float = 3.0       # کشش زمانی کل مسیر نمایشی نسبت به طول برجستگی
    zlim: float = 0.08         # حد کورس فنر [m] (همان گیاه)
    k_relief: float = 0.6      # ضریب «حداقل‌مداخلهٔ رفع‌قید» (تنظیم tune3)
    c_exc: float = 0.75        # ضریب آستانهٔ اضافه‌ارتفاع رخداد نسبت به zlim
    z_thresh: float = 0.012    # آستانهٔ آشکارسازی رخداد از پیش‌نما [m]
    isol_ratio: float = 0.35   # نسبت جدایی رخداد (بیشینهٔ بقیهٔ پنجره / پیک)
    te_min: float = 0.30       # کمترین زمان-تا-رخداد برای فعال‌سازی [s]
    T_return: float = 2.5      # v7: LEGACY/unused — return-to-zero is governed
                               # by y_ref_decay (dmp.py). Kept for old scripts.
    g_abs_max: float = 0.04    # سقف ایمنی هدف مرجع [m]
    y_ref_decay: float = 1.5   # ثابت زمانی بازگشت آرام مرجع در حالت بی‌کار [s]


# ---------------------------------------------------------------------------
# 4) جاده و شبیه‌سازی
# ---------------------------------------------------------------------------
ISO_GD0 = {          # Gd(n0) در n0=0.1 [m^3] — ISO 8608
    'A': 16e-6, 'B': 64e-6, 'C': 256e-6, 'D': 1024e-6,
}


@dataclass
class SimConfig:
    v: float = 20.0            # سرعت longitudinal [m/s]  (≈72 km/h)
    dt_sim: float = 0.005      # گام RK4 گیاه [s]  (200Hz)
    T: float = 12.0            # مدت شبیه‌سازی [s]
    seed: int = 0
    # نویز سنسور برای سناریوهای استحکام (None = ایده‌آل)
    noise_sig: tuple = None    # (σ_zs[m], σ_ẏ[m/s], σ_ua[N])
    v_kalman: float = None     # سرعتی که کنترلر برای پیش‌نما فرض می‌کند (خطای preview)


def save_json(obj: dict, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)


if __name__ == '__main__':
    p = PlantParams()
    print('f_body =', round(p.f_body, 3), 'Hz | f_wheel =', round(p.f_wheel, 3), 'Hz')
