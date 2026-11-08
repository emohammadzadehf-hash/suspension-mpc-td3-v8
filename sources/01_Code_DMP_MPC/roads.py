# -*- coding: utf-8 -*-
"""
roads.py — تولید سه نوع جاده:
  1) IsoRoad  : جادهٔ تصادفی استاندارد ISO 8608 (کلاس A/B/C/D) با IFFT طیفی
  2) BumpRoad : برجستگی هَوِرساین (نیم‌سینوس با شیب صفر در دو انتها)
  3) StepRoad : پلهٔ هموارشده (رامپ کسینوسی)
هر جاده یک متد at(x) دارد: ارتفاع جاده در مختصات طولی x [m].
"""
import numpy as np
from scipy.interpolate import CubicSpline

from config import ISO_GD0


class IsoRoad:
    """جادهٔ تصادفی ISO 8608: Gd(n) = Gd0 (n/n0)^(-w)، n0=0.1 [1/m]، w=2."""

    def __init__(self, cls='C', v=20.0, T=12.0, seed=0, L_extra=40.0,
                 n_min=0.01, n_max=20.0, dx=0.01, w=2.0, n0=0.1):
        self.cls = cls
        self.v = v
        Gd0 = ISO_GD0[cls]
        L = v * T + L_extra
        Nx = int(round(L / dx)) + 1
        x = np.arange(Nx) * dx

        # اطمینان از تفکیک فرکانسی: طول L باید بزرگ‌تر از بلندترین موج باشد
        n_fft = np.fft.rfftfreq(Nx, d=dx)           # [1/m]
        Gd_n = np.where((n_fft >= n_min) & (n_fft <= n_max),
                        Gd0 * (np.maximum(n_fft, n_min) / n0) ** (-w), 0.0)
        # دامنهٔ هرین (one-sided): |Z_j|² ∝ Gd(n_j)·Δn
        # v7 NOTE (m9): for an exact one-sided synthesis the amplitude
        # must be sqrt(Gd·L/2); the raw sqrt(Gd·Δn) under-scales by
        # ~Nx/sqrt(2).  This is intentionally NOT changed — the exact
        # variance renormalisation below (σ_theory/σ_meas) corrects the
        # level exactly, and altering the synthesis would perturb every
        # stored scenario/seed.  Guard against a degenerate all-zero
        # spectrum (e.g. misconfigured n_min/n_max) remains in place.
        dn = 1.0 / (Nx * dx)
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0, 2 * np.pi, size=n_fft.size)
        Z = np.sqrt(Gd_n * dn) * np.exp(1j * phase)
        Z[0] = 0.0
        z = np.fft.irfft(Z, n=Nx)

        # نرمال‌سازی دقیق واریانس به σ تئوری ∫Gd dn (بین n_min و n_max)
        n_grid = np.linspace(max(n_min, 1e-4), n_max, 20000)
        sigma_theory = np.sqrt(np.trapezoid(Gd0 * (n_grid / n0) ** (-w), n_grid))
        sigma_meas = np.std(z)
        if sigma_meas > 0:
            z *= (sigma_theory / sigma_meas)
        self.sigma = sigma_theory
        self.spline = CubicSpline(x, z)
        self.x = x
        self.z = z
        self.L = L

    def at(self, x):
        return self.spline(np.clip(x, 0.0, self.L))

    def profile(self, x0, v, T):
        """پروفایل ارتفاع برحسب زمان: zr(t) = at(x0 + v·t)"""
        return lambda t: self.at(x0 + v * np.asarray(t, dtype=float))


class BumpRoad:
    """برجستگی هَوِرساین: zr = h·sin²(π(x−x0)/L) برای x0 ≤ x ≤ x0+L"""

    def __init__(self, h=0.05, L=2.0, x0=30.0):
        self.h, self.L, self.x0 = h, L, x0

    def at(self, x):
        s = (np.asarray(x, dtype=float) - self.x0) / self.L
        inside = (s >= 0.0) & (s <= 1.0)
        return np.where(inside, self.h * np.sin(np.pi * s) ** 2, 0.0)

    def profile(self, x0, v, T):
        return lambda t: self.at(x0 + v * np.asarray(t, dtype=float))


class StepRoad:
    """پلهٔ هموارشده با رامپ کسینوسی به طول Lr (شیب محدود = فیزیکی)"""

    def __init__(self, h=0.03, x0=30.0, Lr=0.05):
        self.h, self.x0, self.Lr = h, x0, Lr

    def at(self, x):
        s = (np.asarray(x, dtype=float) - self.x0) / self.Lr
        s = np.clip(s, 0.0, 1.0)
        return self.h * 0.5 * (1.0 - np.cos(np.pi * s))

    def profile(self, x0, v, T):
        return lambda t: self.at(x0 + v * np.asarray(t, dtype=float))


def make_road(kind, sim: 'SimConfig', seed=None):
    """کارخانهٔ جاده بر اساس نام سناریو"""
    s = seed if seed is not None else sim.seed
    if kind.startswith('random'):
        cls = kind.replace('random', '') or 'C'
        return IsoRoad(cls=cls, v=sim.v, T=sim.T, seed=s)
    if kind == 'bump':
        return BumpRoad(h=0.05, L=2.0, x0=sim.v * 1.5)     # برخورد در t=1.5s
    if kind == 'step':
        return StepRoad(h=0.03, x0=sim.v * 1.5)            # برخورد در t=1.5s
    raise ValueError(kind)
