# -*- coding: utf-8 -*-
"""
simulation.py — موتور شبیه‌سازی حلقه‌بسته
گیاه RK4 با 200Hz + کنترلر 50Hz + نویز سنسور اختیاری + خطای سرعت preview
"""
import numpy as np

from plant import QuarterCarPlant


class ClosedLoopSim:
    def __init__(self, plant: QuarterCarPlant, road, scfg, controller=None,
                 ctrl_dt: float = None, label: str = ''):
        self.plant = plant
        self.road = road
        self.s = scfg
        self.ctrl = controller
        self.ctrl_dt = ctrl_dt if ctrl_dt is not None else \
            (controller.dt if controller is not None else scfg.dt_sim)
        self.label = label

    # ------------------------------------------------------------------
    def run(self):
        s = self.s
        p = self.plant.p
        dt = s.dt_sim
        n = int(round(s.T / dt))
        ctrl_every = max(int(round(self.ctrl_dt / dt)), 1)
        v = s.v
        v_used = s.v_kalman if s.v_kalman is not None else v
        rng = np.random.default_rng(s.seed + 777)

        X = np.zeros(5)
        t = 0.0
        x_pos = 0.0
        ucmd = 0.0

        log = {k: np.zeros(n) for k in
               ('t', 'zs', 'zds', 'zu', 'zdu', 'ua', 'ucmd', 'zr', 'a_s')}
        info_modes = []
        ref0_log = np.zeros(n)

        for i in range(n):
            zr = float(self.road.at(x_pos))
            a_s = self.plant.body_accel(X, p)
            log['t'][i] = t
            log['zs'][i], log['zds'][i] = X[0], X[1]
            log['zu'][i], log['zdu'][i] = X[2], X[3]
            log['ua'][i], log['ucmd'][i] = X[4], ucmd
            log['zr'][i], log['a_s'][i] = zr, a_s

            if self.ctrl is not None and i % ctrl_every == 0:
                Xc = X.copy()
                if s.noise_sig is not None:
                    sz, sv, su = s.noise_sig
                    Xc[0] += rng.normal(0, sz)
                    Xc[1] += rng.normal(0, sv)
                    Xc[2] += rng.normal(0, sz)
                    Xc[3] += rng.normal(0, sv)
                    Xc[4] += rng.normal(0, su)
                ucmd, info = self.ctrl.compute_control(t, Xc, self.road,
                                                       x_pos, v_used)
                info_modes.append(info['mode'])
                ref0_log[i] = info['ref_y0']

            zr_of_t = (lambda tt, _x0=x_pos, _t0=t:
                       float(self.road.at(_x0 + v * (tt - _t0))))
            X = self.plant.step(t, X, ucmd, zr_of_t, dt)
            t += dt
            x_pos += v * dt

        log['travel'] = log['zs'] - log['zu']
        log['tire'] = log['zu'] - log['zr']
        log['info_modes'] = info_modes
        log['ref0'] = ref0_log
        log['label'] = self.label
        return log
