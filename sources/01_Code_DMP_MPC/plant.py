# -*- coding: utf-8 -*-
"""
plant.py — گیاه فیزیکی: Quarter-Car با اکچویتور مرتبهٔ اول (فراتر از پروپوزال)
حالت‌ها: X = [zs, żs, zu, żu, ua]   ورودی: ucmd   اغتشاش: zr(t)

  ms·z̈s = −ks(zs−zu) − cs(żs−żu) + ua
  mu·z̈u = +ks(zs−zu) + cs(żs−żu) + kt(zr−zu) − ua
  τa·u̇a = ucmd − ua ,  |ucmd| ≤ umax

انتگرال‌گیری: RK4 با ارزیابی جاده در هر زیرگام (اغتشاش پیوسته، نه پله‌ای).
"""
import numpy as np

from config import PlantParams


class QuarterCarPlant:
    def __init__(self, p: PlantParams = None):
        self.p = p or PlantParams()
        p = self.p
        self.ms, self.mu = p.ms, p.mu
        self.ks, self.cs, self.kt = p.ks, p.cs, p.kt
        self.tau_act = p.tau_act
        self.umax = p.umax

    # ------------------------------------------------------------------
    def f(self, t, X, ucmd, zr_of_t):
        """دینامیک پیوسته؛ zr_of_t: تابع زمان→ارتفاع جاده"""
        zs, zds, zu, zdu, ua = X
        zr = float(zr_of_t(t))
        # اشباع فرمان اکچویتور (الکترونیک قدرت)
        ucmd_sat = np.clip(ucmd, -self.umax, self.umax)
        Fs = -self.ks * (zs - zu) - self.cs * (zds - zdu) + ua
        a_s = Fs / self.ms
        Fu = (self.ks * (zs - zu) + self.cs * (zds - zdu)
              + self.kt * (zr - zu) - ua)
        a_u = Fu / self.mu
        ua_dot = (ucmd_sat - ua) / self.tau_act
        return np.array([zds, a_s, zdu, a_u, ua_dot])

    # ------------------------------------------------------------------
    def step(self, t, X, ucmd, zr_of_t, dt):
        """یک گام RK4 با جادهٔ پیوسته"""
        k1 = self.f(t, X, ucmd, zr_of_t)
        k2 = self.f(t + dt / 2, X + dt / 2 * k1, ucmd, zr_of_t)
        k3 = self.f(t + dt / 2, X + dt / 2 * k2, ucmd, zr_of_t)
        k4 = self.f(t + dt, X + dt * k3, ucmd, zr_of_t)
        Xn = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return Xn

    # ------------------------------------------------------------------
    @staticmethod
    def body_accel(X, p: PlantParams):
        """شتاب بدنهٔ لحظه‌ای از روی حالت (برای متریک‌ها)"""
        zs, zds, zu, zdu, ua = X
        return (-p.ks * (zs - zu) - p.cs * (zds - zdu) + ua) / p.ms

    # ------------------------------------------------------------------
    def matrices_cont(self):
        """ماتریس‌های فضای حالت پیوستهٔ ۵تایی (برای MPC و DARE)"""
        p = self.p
        A = np.array([
            [0, 1, 0, 0, 0],
            [-p.ks / p.ms, -p.cs / p.ms, p.ks / p.ms, p.cs / p.ms, 1 / p.ms],
            [0, 0, 0, 1, 0],
            [p.ks / p.mu, p.cs / p.mu, -(p.ks + p.kt) / p.mu, -p.cs / p.mu, -1 / p.mu],
            [0, 0, 0, 0, -1 / p.tau_act],
        ])
        B = np.array([[0], [0], [0], [0], [1 / p.tau_act]])
        G = np.array([[0], [0], [0], [p.kt / p.mu], [0]])
        return A, B, G
