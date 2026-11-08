# -*- coding: utf-8 -*-
"""
controller.py — کنترلر DMP + MPC (CasADi / IPOPT) — نسخهٔ فراتر از پروپوزال

رفع شش نقص اذعان‌شدهٔ پروپوزال:
  (1) انتگرال‌گیری: گسسته‌سازی دقیق ZOH با expm (به‌جای اویلر رو به جلو)
  (2) جریمهٔ Δu نسبت به «دستور اجراشدهٔ قبلی» (u_prev) نه گرهٔ افق
  (3) warm-start با جابه‌جایی مسیر بهینهٔ قبلی
  (4) زنجیرهٔ fallback: شکست IPOPT → نگهداشت فرمان قبلی → (۳ شکست پیوسته) LQR ایمن DARE
  (5) هم‌ترازی دقیق preview: جاده در گره‌های افق با درون‌یابی مکعبی
  (6) نرمال‌سازی همهٔ جملات هزینه با مقیاس فیزیکی + هزینهٔ ترمینال DARE
امکانات فراتر از پروپوزال:
  • اکچویتور مرتبهٔ اول داخل مدل MPC (حالت پنجم)
  • قید نرم کورس فنر با اسلاک L2 (پرهیز از ناپایداری NLP)
  • مرجع DMP با پیش‌بینی کامل روی افق (projection)
  • آمار زمان حل و نرخ موفقیت حل‌گر
"""
import time
import numpy as np
import casadi as ca
from scipy.linalg import expm, solve_discrete_are

from config import MpcConfig, DmpConfig
from dmp import RoadEventDMP


class DmpMpcController:
    def __init__(self, plant, mcfg: MpcConfig = None, dcfg: DmpConfig = None, v: float = 20.0):
        self.plant = plant
        self.p = plant.p
        self.mcfg = mcfg or MpcConfig()
        self.dcfg = dcfg or DmpConfig()
        self.v = v
        self.dmp = (RoadEventDMP(self.dcfg, dt=self.mcfg.dt_mpc, v=v)
                    if self.mcfg.use_dmp else None)
        self._build_model()
        self._build_opti()
        self._build_fallback()
        self._reset_state()

    # ------------------------------------------------------------------
    def _build_model(self):
        """ماتریس‌های مدل MPC (با tau_act_model) + گسسته‌سازی دقیق ZOH"""
        p, m = self.p, self.mcfg
        tau = m.tau_act_model
        A = np.array([
            [0, 1, 0, 0, 0],
            [-p.ks / p.ms, -p.cs / p.ms, p.ks / p.ms, p.cs / p.ms, 1 / p.ms],
            [0, 0, 0, 1, 0],
            [p.ks / p.mu, p.cs / p.mu, -(p.ks + p.kt) / p.mu, -p.cs / p.mu, -1 / p.mu],
            [0, 0, 0, 0, -1 / tau],
        ])
        B = np.array([[0.0], [0.0], [0.0], [0.0], [1 / tau]])
        G = np.array([[0.0], [0.0], [0.0], [p.kt / p.mu], [0.0]])
        dt = m.dt_mpc
        Mblk = np.zeros((7, 7))
        Mblk[:5, :5] = A * dt
        Mblk[:5, 5] = B[:, 0] * dt
        Mblk[:5, 6] = G[:, 0] * dt
        E = expm(Mblk)
        self.Ad, self.Bd, self.Gd = E[:5, :5], E[:5, 5:6], E[:5, 6:7]
        # --- گسسته‌سازی First-Order-Hold برای اغتشاش جاده (فراتر از پروپوزال) ---
        # w(τ) بین گره‌ها خطی: x+ = Ad·x + Gd·wk + Gt·(wk+1 − wk)
        # (هستهٔ صحیح e^{A(dt−s)}: پس از تعویض متغیر σ=dt−s)
        Ainv = np.linalg.inv(A)
        Ainv2 = Ainv @ Ainv
        Mw = Ainv2 @ ((self.Ad @ A * dt - self.Ad) + np.eye(5)) @ G
        self.G0 = self.Gd                 # ضریب w_k
        self.G1 = self.Gd - Mw / dt       # ضریب (w_{k+1} − w_k)
        self.A_c, self.B_c, self.G_c = A, B, G

    # ------------------------------------------------------------------
    def _terminal_P(self):
        """P از DARE برای هزینهٔ ترمینال.

        v7 NOTE: «تضمین پایداری» دقیق نیست — Q_term تقریبی است (ترم q_t
        لاستیک، مسیر مستقیم ks/ms و r_du در آن حذف شده‌اند) و فقط نقش
        پایدارکنندهٔ هیوریستیک برای افق‌های بلند با r کوچک دارد؛ این
        انتخاب تجربی در tune4 پایدار و مؤثر بوده است.
        """
        p, m = self.p, self.mcfg
        M_map = np.zeros((5, 5))
        M_map[0, 0] = 1.0
        M_map[1, 1] = 1.0
        M_map[2, 0] = 1.0
        M_map[2, 2] = -1.0
        d = np.zeros(5)
        d[0] = m.q_r / m.s_r ** 2 + m.q_s / m.s_s ** 2
        d[1] = m.q_rv / 0.2 ** 2 + m.q_a * (p.cs / p.ms) ** 2 / m.s_a ** 2
        d[2] = m.q_s / m.s_s ** 2
        Q_term = M_map.T @ np.diag(d) @ M_map
        R = np.array([[m.r_u / m.s_u ** 2]])
        P = solve_discrete_are(self.Ad, self.Bd, Q_term, R)
        self.M_map, self.P_dare = M_map, P
        return P

    # ------------------------------------------------------------------
    def _build_opti(self):
        p, m = self.p, self.mcfg
        N, dt = m.N, m.dt_mpc
        self.N, self.dt = N, dt
        P = self._terminal_P()
        K_fb = np.linalg.solve(np.array([[m.r_u / m.s_u ** 2]])
                               + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)
        self._K_fb_saved = K_fb

        opti = ca.Opti()
        X = opti.variable(N + 1, 5)
        U = opti.variable(N, 1)
        SL = opti.variable(N, 1)
        px0 = opti.parameter(5)
        puprev = opti.parameter(1)
        pref_y = opti.parameter(N + 1)
        pref_v = opti.parameter(N + 1)
        proad = opti.parameter(N + 1)
        # --- ارتقا v2: همهٔ وزن‌ها پارامتر آنلاین‌اند → زمان‌بندی DMP-گیت شدنی ---
        pw = opti.parameter(8)   # [q_r, q_rv, q_a, q_s, q_t, r_u, r_du, rho_susp]

        # دینامیک (ZOH فرمان + FOH جاده — هم‌ترازی دقیق preview)
        opti.subject_to(X[0, :].T == px0)
        for k in range(N):
            opti.subject_to(X[k + 1, :].T == self.Ad @ X[k, :].T
                            + self.Bd @ U[k]
                            + self.G0 * proad[k]
                            + self.G1 * (proad[k + 1] - proad[k]))

        # هزینه
        def a_of(x):
            return (-(p.ks) * (x[0] - x[2]) - p.cs * (x[1] - x[3]) + x[4]) / p.ms

        J = 0
        for k in range(1, N + 1):
            x = X[k, :]
            a_s = a_of(x.T)
            du = U[k - 1] - (U[k - 2] if k >= 2 else puprev)
            J += pw[0] * ((x[0] - pref_y[k]) / m.s_r) ** 2 \
                + pw[1] * ((x[1] - pref_v[k]) / 0.2) ** 2 \
                + pw[2] * (a_s / m.s_a) ** 2 \
                + pw[3] * ((x[0] - x[2]) / m.s_s) ** 2 \
                + pw[4] * ((x[2] - proad[k]) / m.s_t) ** 2 \
                + pw[5] * (U[k - 1] / m.s_u) ** 2 \
                + pw[6] * (du / m.s_u) ** 2 \
                + pw[7] * (SL[k - 1] / m.s_s) ** 2
        if m.use_terminal:
            eN = ca.vertcat(X[N, 0] - pref_y[N], X[N, 1] - pref_v[N],
                            X[N, 0] - X[N, 2], 0, 0)
            J += eN.T @ P @ eN
        opti.minimize(J)

        # قیدها
        opti.subject_to(opti.bounded(-p.umax, U, p.umax))
        opti.subject_to(opti.bounded(-p.umax, X[1:, 4], p.umax))
        opti.subject_to(opti.bounded(0, SL, 0.03))
        travel = X[1:, 0] - X[1:, 2]
        opti.subject_to(opti.bounded(-p.zlim - SL, travel, p.zlim + SL))

        opts = {'expand': True,
                'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': False,
                'ipopt.max_iter': m.max_iter, 'ipopt.tol': m.tol,
                'ipopt.acceptable_tol': m.acceptable_tol,
                'ipopt.mu_strategy': 'adaptive',
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.warm_start_bound_push': 1e-8,
                'ipopt.warm_start_mult_bound_push': 1e-8}
        opti.solver('ipopt', opts)
        self.opti, self._X, self._U, self._SL = opti, X, U, SL
        self._px0, self._puprev, self._pref_y, self._pref_v, self._proad = \
            px0, puprev, pref_y, pref_v, proad
        self._pw = pw

    # ------------------------------------------------------------------
    def _build_fallback(self):
        K = self._K_fb_saved
        self.K_fb = np.asarray(K).reshape(1, 5)

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.u_prev = 0.0
        self.prev_X = None
        self.prev_U = None
        self.n_fail = 0
        self.n_solves = 0
        self.n_success = 0
        self.n_fallback_steps = 0
        self.solve_ms = []
        self.statuses = []

    def reset(self):
        self._reset_state()
        if self.dmp is not None:
            self.dmp.reset()

    # ------------------------------------------------------------------
    def compute_control(self, t, X_meas, road, x_pos, v_used, T_preview=1.5):
        """یک گام کنترل؛ خروجی ucmd و دیکشنری اطلاعات تشخیصی"""
        m = self.mcfg
        N, dt = self.N, self.dt
        p = self.p

        # ---- 1) مرجع DMP (تشخیص رخداد + به‌روزرسانی + پیش‌بینی افق) ----
        if self.dmp is not None:
            n_pv = int(round(T_preview / dt)) + 1
            t_rel = np.arange(n_pv) * dt
            z_pv = np.asarray(road.at(x_pos + v_used * t_rel)).ravel()
            self.dmp.try_trigger(t, t_rel, z_pv, self.dmp.y)
            self.dmp.update(t)
            ref_y, ref_v = self.dmp.project(N, dt)
        else:
            ref_y = np.zeros(N + 1)
            ref_v = np.zeros(N + 1)

        # ---- 2) جاده در گره‌های افق (هم‌ترازی دقیق preview) ----
        node_x = x_pos + v_used * np.arange(N + 1) * dt
        w_nodes = np.asarray(road.at(node_x)).ravel()

        # ---- 3) تنظیم پارامترها (شامل زمان‌بندی وزن ردیابی DMP-گیت — ارتقا v2) ----
        dmp_engaged = bool(self.dmp is not None and
                           (self.dmp.active or self.dmp.pending is not None))
        w = [m.q_r if dmp_engaged else m.q_r_idle,
             m.q_rv, m.q_a, m.q_s, m.q_t, m.r_u, m.r_du, m.rho_susp]
        self.opti.set_value(self._pw, np.asarray(w, dtype=float))
        self.opti.set_value(self._px0, np.asarray(X_meas, dtype=float).ravel())
        self.opti.set_value(self._puprev, self.u_prev)
        self.opti.set_value(self._pref_y, ref_y)
        self.opti.set_value(self._pref_v, ref_v)
        self.opti.set_value(self._proad, w_nodes)

        # ---- 4) warm start ----
        if self.prev_X is not None:
            Xg = np.vstack([np.asarray(X_meas, float).reshape(1, 5),
                            self.prev_X[1:]])
            Ug = np.vstack([self.prev_U[1:], self.prev_U[-1:]])
        else:
            Xg = np.tile(np.asarray(X_meas, float).reshape(1, 5), (N + 1, 1))
            Ug = np.zeros((N, 1))
        self.opti.set_initial(self._X, Xg)
        self.opti.set_initial(self._U, Ug)
        self.opti.set_initial(self._SL, np.zeros((N, 1)))

        # ---- 5) حل ----
        t0 = time.perf_counter()
        mode = 'mpc'
        try:
            sol = self.opti.solve()
            ms = (time.perf_counter() - t0) * 1e3
            u0 = float(sol.value(self._U[0]))
            self.prev_X = np.array(sol.value(self._X))
            self.prev_U = np.array(sol.value(self._U)).reshape(N, 1)
            st = sol.stats().get('return_status', '?')
            ok = st in ('Solve_Succeeded', 'Solved_To_Acceptable_Level')
            self.n_fail = 0 if ok else self.n_fail + 1
            if ok:
                self.n_success += 1
            self.statuses.append(st)
            self.solve_ms.append(ms)
        except RuntimeError:
            ms = (time.perf_counter() - t0) * 1e3
            self.n_fail += 1
            self.statuses.append('Fail')
            self.solve_ms.append(ms)
            if self.n_fail >= m.fallback_after:
                mode = 'lqr'
            else:
                mode = 'hold'
                u0 = self.u_prev

        if mode == 'lqr':
            # LQR ایمن روی خطای ردیابی (fallback نهایی)
            e = (self.M_map @ np.asarray(X_meas, float).ravel()
                 - np.array([ref_y[0], ref_v[0], 0.0, 0.0, 0.0]))
            fb_val = np.dot(-self.K_fb, e)
            if hasattr(fb_val, '__len__'):
                fb_val = fb_val[0]
            u0 = float(np.clip(fb_val, -p.umax, p.umax))
            self.n_fallback_steps += 1
            self.prev_X = None
            self.prev_U = None

        u0 = float(np.clip(u0, -p.umax, p.umax))
        self.u_prev = u0
        self.n_solves += 1
        info = {'mode': mode, 'ref_y0': float(ref_y[0]),
                'dmp_active': bool(self.dmp.active) if self.dmp else False,
                'dmp_engaged': dmp_engaged}
        return u0, info
