# -*- coding: utf-8 -*-
"""
dmp.py — مولد مرجع پیش‌نگر DMP (نسخهٔ فراتر از پروپوزال)

ایده: از پنجرهٔ پیش‌نمای جاده، «رخداد» (برجستگی/چالهٔ معنادار) آشکار می‌شود؛
سپس یک DMP زمانی استاندارد Ijspeert روی «مسیر نمایشی» مهندسی‌شده
(برخاست کسینوسی → نگهداشت → بازگشت به صفر) آموزش می‌بیند و مرجع هموار
y_ref(t)، ẏ_ref(t) را با مشتق‌های دقیق تولید می‌کند. کنترلر MPC این مرجع
را در هزینهٔ ردیابی می‌بیند و «پیش‌چیدن» بدنه قبل از برخورد را اجرا می‌کند.

بهبودها نسبت به پروپوزال:
  1) رویداد-محور: DMP فقط وقتی مفید است فعال می‌شود (گیت انرژی پیش‌نما)
  2) یادگیری وزن‌ها با LSQ از روی نمایش (نه تنظیم دستی)
  3) پیوستگی مرجع: y0 هر رخداد = مقدار فعلی مرجع (پرش نداریم)
  4) مشتق‌های دقیق (y, v) از حالت DMP — نه تفاضل عددی خام
  5) سقف ایمنی هدف g و بازگشت خودکار به تراز ثابت
  6) پیش‌بینی آیندهٔ مرجع برای گره‌های افق MPC (projection)
"""
import numpy as np

from config import DmpConfig


class RoadEventDMP:
    def __init__(self, cfg: DmpConfig = None, dt: float = 0.02, v: float = 20.0):
        self.cfg = cfg or DmpConfig()
        self.dt = dt
        self.v = v
        c = self.cfg
        # توابع پایهٔ گاوسی با مرکزهای لگاریتمی (استاندارد Ijspeert)
        i = np.arange(c.n_basis)
        self.c = np.exp(-c.alpha_s * i / max(c.n_basis - 1, 1))
        dc = np.diff(self.c)
        self.h = 0.65 / np.maximum(dc ** 2, 1e-9)
        self.h = np.append(self.h, self.h[-1])
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, y: float = 0.0):
        c = self.cfg
        self.y = float(y)
        self.vy = 0.0
        self.s = 1.0
        self.active = False
        self.w = None
        self.tau = 1.0
        self.g = 0.0
        self.y0 = float(y)
        self.t_elapsed = 0.0
        self.n_triggers = 0
        self.pending = None          # رخداد در انتظار فعال‌سازی (فعال‌سازی با تأخیر)
        self._now = None

    # ------------------------------------------------------------------
    def _basis(self, s):
        s = np.asarray(s, dtype=float)
        if s.ndim == 0:                                   # اسکالر → (n_basis,)
            return np.exp(-self.h * (s - self.c) ** 2)
        return np.exp(-self.h[None, :] * (s[:, None] - self.c[None, :]) ** 2)

    def _forcing(self, s, w):
        """f_unit(s) = s·Σψw/Σψ — ضریب مقیاس (g−y0) در _deriv اعمال می‌شود"""
        psi = self._basis(s)
        spsi = psi.sum()
        if spsi < 1e-12:
            return 0.0
        return s * float(psi @ w) / spsi

    # ------------------------------------------------------------------
    def learn_from_demo(self, demo_unit, dt_demo, tau):
        """
        یادگیری وزن‌ها از نمایش «یکه» (y از 0 به 1) — پایدار از نظر عددی.
        f_target = τ²ÿ + αz(βz(1−y) − τẏ) و ویژگی‌های رگرسیون φ = s·ψ/Σψ.
        """
        c = self.cfg
        demo_unit = np.asarray(demo_unit, dtype=float)
        td = np.arange(demo_unit.size) * dt_demo
        dy = np.gradient(demo_unit, dt_demo)
        ddy = np.gradient(dy, dt_demo)
        s_j = np.exp(-c.alpha_s * td / tau)
        # علامت استاندارد Ijspeert: f = τ²ÿ − αz(βz(g−y) − τẏ)
        f_target = tau ** 2 * ddy - c.alpha_z * (c.beta_z * (1.0 - demo_unit) - tau * dy)
        Psi = self._basis(s_j)                                  # (M × n_basis)
        Phi = s_j[:, None] * Psi / np.maximum(Psi.sum(axis=1, keepdims=True), 1e-12)
        Gm = Phi.T @ Phi + 1e-6 * np.eye(self.c.size) * Phi.shape[0]
        w = np.linalg.solve(Gm, Phi.T @ f_target)
        # کیفیت برازش برای نظارت
        fit_err = float(np.sqrt(np.mean((Phi @ w - f_target) ** 2)) /
                        max(np.std(f_target), 1e-9))
        return w, fit_err

    # ------------------------------------------------------------------
    def _build_demo(self, y0, g, T_rise, T_hold):
        """
        نمایش مهندسی‌شدهٔ «یکه» (0→1): برخاست کسینوسی + نگهداشت.
        بازگشت به صفر خارج از DMP با افت آرام انجام می‌شود (τ کوچک = یادگیری پایدار).
        خروجی: (نمایش یکه، dt، τ کل)
        """
        dt_fine = 0.004
        n1 = max(int(round(T_rise / dt_fine)), 10)
        n2 = max(int(round(T_hold / dt_fine)), 5)
        seg1 = 0.5 * (1 - np.cos(np.pi * np.arange(1, n1 + 1) / n1))
        seg2 = np.ones(n2)
        demo = np.concatenate([seg1, seg2])
        return demo, dt_fine, n1 * dt_fine + n2 * dt_fine

    # ------------------------------------------------------------------
    def try_trigger(self, t_now, preview_t, preview_z, y_current):
        """
        آشکارسازی رخداد از پنجرهٔ پیش‌نما و زمان‌بندی فعال‌سازی DMP.
        فعال‌سازی واقعی طوری برنامه‌ریزی می‌شود که اوج مرجع دقیقاً هم‌زمان با
        رسیدن برجستگی باشد (T_rise ثانیه قبل از برخورد، برخاست شروع می‌شود).
        preview_t: زمان نسبی نمونه‌ها [s]، preview_z: ارتفاع جاده در همان نقاط [m]
        """
        c = self.cfg
        if self.active or self.pending is not None:
            return False
        pt = np.asarray(preview_t)
        pz = np.asarray(preview_z)
        win = (pt >= c.te_min)
        if not np.any(win):
            return False
        idx = np.where(win)[0]
        j = idx[np.argmax(np.abs(pz[idx]))]
        h_peak = float(pz[j])
        te = float(pt[j])
        if abs(h_peak) < c.z_thresh:
            return False
        # گیت هوشمند: فقط رخدادهایی که «به‌تنهایی قید کورس را می‌شکنند»
        # و «جداشده» هستند → DMP حداقل‌مداخلهٔ رفع‌قید می‌سازد؛ بقیه → بی‌کار
        above = np.abs(pz) >= 0.5 * abs(h_peak)
        above &= (pt >= max(pt[j] - 2.0, 0.0)) & (pt <= pt[j] + 2.0)
        T_b = max(above.sum() * (pt[1] - pt[0]) if pt.size > 1 else 0.2, 0.08)
        guard = (np.abs(pt - pt[j]) > 1.5 * T_b)
        if np.any(guard) and np.max(np.abs(pz[guard])) > c.isol_ratio * abs(h_peak):
            return False
        excess = abs(h_peak) - c.c_exc * c.zlim
        if excess <= 0.0:
            return False
        # ارتقا v2: رفع‌قید فقط «کسری که کورس نمی‌تواند جذب کند» را جبران می‌کند
        # نه کل اضافه‌ارتفاع → مداخلهٔ کمتر، هزینهٔ آسایش کمتر
        g_relief = float(np.clip(c.k_relief * excess, 0.0, c.g_abs_max))
        g = float(np.sign(h_peak) * g_relief)
        y0 = float(y_current)
        if abs(g - y0) < 0.002:          # حرکت بی‌معنا
            return False
        T_rise = min(te, c.stretch * T_b)
        if T_rise < 0.12:
            return False
        t_start = t_now + max(te - T_rise, 0.0)
        self.pending = {'t_start': t_start, 'g': g, 'T_rise': T_rise,
                        'T_hold': T_b}
        self.n_triggers += 1
        return True

    # ------------------------------------------------------------------
    def _activate(self):
        """فعال‌سازی واقعی DMP با y0 = مقدار فعلی مرجع (پیوستگی کامل)"""
        pd = self.pending
        y0 = self.y
        g = pd['g']
        if abs(g - y0) < 0.0015:
            self.pending = None
            return
        demo_u, dt_demo, tau = self._build_demo(y0, g, pd['T_rise'], pd['T_hold'])
        w, fit_err = self.learn_from_demo(demo_u, dt_demo, tau)
        if w is None or fit_err > 0.5:
            self.pending = None
            return
        self.w = w
        self.tau = tau
        self.g = g
        self.y0 = y0
        self.y = y0
        self.vy = 0.0
        self.s = 1.0
        self.t_elapsed = 0.0
        self.active = True
        self.pending = None
        self.last_fit_err = fit_err

    # ------------------------------------------------------------------
    def _deriv(self, st):
        """st = (s, y, v) → مشتق‌ها مطابق DMP استاندارد با مقیاس (g−y0)"""
        s, y, v = st
        c = self.cfg
        f = self._forcing(s, self.w) * (self.g - self.y0)
        dy = v / self.tau
        dv = (c.alpha_z * (c.beta_z * (self.g - y) - v) + f) / self.tau
        ds = -c.alpha_s * s / self.tau
        return np.array([ds, dy, dv])

    def _integrate(self, st, dt):
        k1 = self._deriv(st)
        k2 = self._deriv(st + 0.5 * dt * k1)
        return st + dt * k2            # RK2

    # ------------------------------------------------------------------
    def update(self, t_now=None):
        """یک گام DMP در نرخ کنترل (t_now برای زمان‌بندی فعال‌سازی)"""
        c = self.cfg
        if self.active:
            st = self._integrate((self.s, self.y, self.vy), self.dt)
            self.s, self.y, self.vy = st
            # کلیپ ایمنی مرجع (دفاع در عمق)
            gmax = 1.5 * self.cfg.g_abs_max
            self.y = float(np.clip(self.y, -gmax, gmax))
            self.vy = float(np.clip(self.vy, -0.5, 0.5))
            self.t_elapsed += self.dt
            if self.s < 0.005 or self.t_elapsed > 2.0 * self.tau:
                self.active = False
        elif self.pending is not None:
            if t_now is not None:
                self._now = t_now   # v7: never overwrite a valid _now with None
            if self._now is not None and t_now is not None \
                    and t_now >= self.pending['t_start']:
                self._activate()
            # در انتظار رخداد: مرجع ثابت می‌ماند (بدون افت)
        else:
            # بازگشت آرام مرجع به صفر در حالت بی‌کار (پیوستگی)
            self.y += -self.y * self.dt / c.y_ref_decay
            self.vy = -self.y / c.y_ref_decay
        return self.y, self.vy

    # ------------------------------------------------------------------
    def project(self, N, dt_m):
        """پیش‌بینی مرجع در گره‌های افق MPC: خروجی (N+1، y) و (N+1، v)"""
        st = np.array([self.s, self.y, self.vy])
        ys = np.empty(N + 1)
        vs = np.empty(N + 1)
        ys[0], vs[0] = st[1], st[2]
        if self.active:
            for k in range(1, N + 1):
                st = self._integrate(st, dt_m)
                ys[k], vs[k] = st[1], st[2]
        elif self.pending is not None and getattr(self, '_now', None) is not None:
            # فعال‌سازی معوقه: تا t_start ثابت، بعد دینامیک DMP
            tmp = RoadEventDMP.__new__(RoadEventDMP)
            tmp.__dict__.update(self.__dict__)
            tmp.pending = None
            tmp.active = False
            tmp.w = None
            tmp._act_failed = False          # v7 (M4): one LSQ attempt max
            n_wait = max(int(round((self.pending['t_start'] - self._now) / dt_m)), 0)
            for k in range(1, N + 1):
                if k <= n_wait:
                    ys[k] = ys[k - 1]
                    vs[k] = 0.0
                else:
                    if not tmp.active and not tmp._act_failed:
                        self._pending_activate_copy(tmp)
                    if not tmp.active:
                        # v7 FIX (M4): activation failed (fit_err>0.5 or
                        # negligible goal).  The old code integrated anyway
                        # and crashed with `psi @ None` inside the control
                        # loop — and retried the 40-basis LSQ at every
                        # remaining node.  Now: hold the reference constant
                        # (same convention as the waiting phase).
                        tmp._act_failed = True
                        ys[k] = ys[k - 1]
                        vs[k] = 0.0
                        continue
                    st2 = tmp._integrate((tmp.s, tmp.y, tmp.vy), dt_m)
                    tmp.s, tmp.y, tmp.vy = st2
                    ys[k], vs[k] = tmp.y, tmp.vy
        else:
            dec = np.exp(-dt_m / self.cfg.y_ref_decay)
            for k in range(1, N + 1):
                ys[k] = ys[k - 1] * dec
                vs[k] = -ys[k] / self.cfg.y_ref_decay
        return ys, vs

    def _pending_activate_copy(self, tmp):
        """فعال‌سازی روی نسخهٔ کپی برای projection (بدون دستکاری وضعیت اصلی)"""
        pd = self.pending
        y0 = tmp.y
        g = pd['g']
        if abs(g - y0) < 0.0015:
            tmp.active = False
            tmp.w = None
            return
        demo_u, dt_demo, tau = self._build_demo(y0, g, pd['T_rise'], pd['T_hold'])
        w, fit_err = self.learn_from_demo(demo_u, dt_demo, tau)
        if w is None or fit_err > 0.5:
            tmp.active = False
            tmp.w = None
            return
        tmp.w = w
        tmp.tau = tau
        tmp.g = g
        tmp.y0 = y0
        tmp.y = y0
        tmp.vy = 0.0
        tmp.s = 1.0
        tmp.t_elapsed = 0.0
        tmp.active = True
