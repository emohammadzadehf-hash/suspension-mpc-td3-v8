# -*- coding: utf-8 -*-
"""
run_experiments.py — خط لولهٔ کامل آزمایش‌ها (نسخهٔ نهایی v2)
  E1 experiments.json : step / bump / bump8 / bump10 / randomB / randomC / randomD × {passive, mpc, dmp+mpc} × seeds
  E2 ablation.json    : حالت «مقاله» در برابر v1 و v2 (پیشرفت پله‌پله)
  E3 robustness.json  : عدم‌قطعیت پارامتر، نویز سنسور، خطای سرعت preview
  E4 speed.json       : sweep سرعت 10/20/30 m/s
  + sim_data.npz برای نمودارها

پیکربندی v2 (در صورت وجود results/tune4.json) خودکار بارگذاری می‌شود.
"""
import time
import os
import json
import dataclasses
import numpy as np

# v7 FIX (M8): fresh checkouts have no results/ yet
os.makedirs('results', exist_ok=True)

from config import PlantParams, MpcConfig, DmpConfig, SimConfig, save_json
from plant import QuarterCarPlant
from roads import make_road, BumpRoad
from controller import DmpMpcController
from simulation import ClosedLoopSim
from metrics import summarize, reduction, controller_stats


def load_tuned():
    """پیکربندی تنظیم‌شدهٔ نهایی — اولویت: tune4.json سپس tune3.json"""
    m, d = MpcConfig(use_dmp=True), DmpConfig()
    for cand in ('results/tune4.json', 'results/tune3.json'):
        if os.path.exists(cand):
            tj = json.load(open(cand))
            m = MpcConfig(**{k: v for k, v in tj['mpc'].items()})
            d = DmpConfig(**{k: v for k, v in tj['dmp'].items()})
            print('>> tuned config loaded from %s (score=%.4f)'
                  % (cand, tj['score']))
            break
    return m, d


plant = QuarterCarPlant(PlantParams())
V = 20.0
SCENARIOS = [('step', 4.0, [0]), ('bump', 4.5, [0]),
             ('bump8', 4.5, [0]), ('bump10', 4.5, [0]),
             ('randomB', 12.0, [0, 1, 2, 3]),
             ('randomC', 12.0, [0, 1, 2, 3]),
             ('randomD', 12.0, [0, 1, 2, 3])]


def make_road2(kind, sc, seed):
    if kind.startswith('bump'):
        h = 0.08 if kind == 'bump8' else (0.10 if kind == 'bump10' else 0.05)
        return BumpRoad(h=h, L=2.0, x0=sc.v * 1.5)
    return make_road(kind, sc, seed=seed)


def run_case(kind, T, seed, ctrl_mode, plant_sim=None, sc_override=None,
             keep_log=False, mcfg=None, dcfg=None):
    """یک ران کامل؛ ctrl_mode ∈ {passive, mpc, dmpmpc}"""
    sc = sc_override or SimConfig(v=V, dt_sim=0.005, T=T, seed=seed)
    road = make_road2(kind, sc, seed)
    psim = plant_sim or plant
    if ctrl_mode == 'passive':
        log = ClosedLoopSim(psim, road, sc, controller=None).run()
        sm = summarize(log)
        cs = {}
    else:
        mcfg = mcfg or (MpcConfig(use_dmp=(ctrl_mode == 'dmpmpc')))
        dcfg = dcfg or DmpConfig()
        ctrl = DmpMpcController(psim, mcfg, dcfg, v=sc.v)
        log = ClosedLoopSim(psim, road, sc, controller=ctrl).run()
        sm = summarize(log)
        cs = controller_stats(ctrl)
    out = {'summary': sm, 'controller': cs}
    if keep_log:
        out['_log'] = log
    return out


# ======================================================================
# v7 FIX (M8): the E1–E4 suite below is expensive (~35 min of IPOPT
# solves) and used to execute as an import side effect.  It is a script:
# importing it now raises instead of silently launching the pipeline.
if __name__ != '__main__':
    raise ImportError(
        "run_experiments.py is a script, not a library.  Run it directly: "
        "python run_experiments.py  (for programmatic use import "
        "run_case / load_tuned from it inside a __main__ block).")

print('==== E1: experiments (v2) ====')
V2_M, V2_D = load_tuned()
experiments = {}
t_start = time.time()
for kind, T, seeds in SCENARIOS:
    experiments[kind] = {}
    for seed in seeds:
        key = 'seed%d' % seed
        experiments[kind][key] = {}
        for mode in ('passive', 'mpc', 'dmpmpc'):
            r = run_case(kind, T, seed, mode,
                         mcfg=(V2_M if mode == 'dmpmpc'
                               else dataclasses.replace(V2_M, use_dmp=False)),
                         dcfg=V2_D)
            experiments[kind][key][mode] = r
        sp = experiments[kind][key]['passive']['summary']
        for mode in ('mpc', 'dmpmpc'):
            experiments[kind][key][mode]['reduction'] = reduction(
                experiments[kind][key][mode]['summary'], sp)
        a = experiments[kind][key]
        print('%-8s %s : passive=%.3f  mpc=%+.1f%%  dmp+mpc=%+.1f%%' %
              (kind, key, sp['a_rms'], a['mpc']['reduction']['a_rms'],
               a['dmpmpc']['reduction']['a_rms']))
save_json(experiments, 'results/experiments.json')
print('E1 done [%.0fs]' % (time.time() - t_start))

# ======================================================================
print('==== E2: ablation (paper-mode vs v1 vs v2) ====')
ablation = {}
for kind, T in (('randomC', 10.0), ('bump', 4.5), ('bump10', 4.5)):
    ablation[kind] = {}
    sc = SimConfig(v=V, dt_sim=0.005, T=T, seed=0)
    road = make_road2(kind, sc, 0)
    lp = ClosedLoopSim(plant, road, sc, controller=None).run()
    sp = summarize(lp)
    # حالت «مطابق مقاله»: افق کوتاه 0.2s، بدون ترمینال، اکچویتور ایده‌آل، بدون Δu
    paper_m = MpcConfig(use_dmp=False, N=20, dt_mpc=0.01, use_terminal=False,
                        tau_act_model=0.001, r_du=0.0)
    # پیکربندی v1: 50Hz، N=50، وزن‌های اولیه، q_r ثابت 0.5 (بدون زمان‌بندی)
    v1_m = MpcConfig(use_dmp=True, dt_mpc=0.02, N=50, max_iter=80,
                     q_r=0.5, q_r_idle=0.5, q_a=8.0, q_s=4.0, q_t=1.0,
                     r_u=0.02, r_du=0.05)
    v1_d = DmpConfig(k_relief=0.9, c_exc=0.75)
    for tag, mcfg, dcfg in (('paper_mode', paper_m, DmpConfig()),
                            ('v1_full', v1_m, v1_d),
                            ('v2_full_mpc',
                             dataclasses.replace(V2_M, use_dmp=False), V2_D),
                            ('v2_full_dmpmpc', V2_M, V2_D)):
        ctrl = DmpMpcController(plant, mcfg, dcfg, v=sc.v)
        lc = ClosedLoopSim(plant, road, sc, controller=ctrl).run()
        sm = summarize(lc)
        ablation[kind][tag] = {'summary': sm,
                               'reduction': reduction(sm, sp),
                               'controller': controller_stats(ctrl)}
        print('%-8s %-14s: a%+.1f%%  vdv%+.1f%%  trav_pk=%.4f  solve=%.1fms' %
              (kind, tag, ablation[kind][tag]['reduction']['a_rms'],
               ablation[kind][tag]['reduction'].get('vdv', 0.0),
               sm['travel_peak'],
               ablation[kind][tag]['controller'].get('solve_ms_mean', 0)))
save_json(ablation, 'results/ablation.json')

# ======================================================================
print('==== E3: robustness ====')
robustness = {}
cases = [
    ('nominal', PlantParams(), None, None),
    ('ms+30%', PlantParams(ms=PlantParams.ms * 1.3), None, None),
    ('ks-20%', PlantParams(ks=PlantParams.ks * 0.8), None, None),
    ('cs+30%', PlantParams(cs=PlantParams.cs * 1.3), None, None),
    ('kt-15%', PlantParams(kt=PlantParams.kt * 0.85), None, None),
    ('act_lag_x2', PlantParams(tau_act=0.030), None, None),
    ('sensor_noise', PlantParams(), (0.001, 0.02, 5.0), None),
    ('preview_speed_-10%', PlantParams(), None, 0.9),
]
for name, p_pert, noise, vk in cases:
    sc = SimConfig(v=V, dt_sim=0.005, T=10.0, seed=0,
                   noise_sig=noise, v_kalman=vk * V if vk else None)
    road = make_road2('randomC', sc, 0)
    plant_pert = QuarterCarPlant(p_pert)
    r = run_case('randomC', 10.0, 0, 'dmpmpc', plant_sim=plant_pert,
                 sc_override=sc, mcfg=V2_M, dcfg=V2_D)
    robustness[name] = r
    print('%-18s: a_rms=%.3f  trav_pk=%.4f  succ=%.2f' %
          (name, r['summary']['a_rms'], r['summary']['travel_peak'],
           r['controller'].get('success_rate', 0)))
save_json(robustness, 'results/robustness.json')

# ======================================================================
print('==== E4: speed sweep ====')
speed = {}
for v in (10.0, 20.0, 30.0):
    speed['v%02d' % int(v)] = {}
    for mode in ('passive', 'mpc', 'dmpmpc'):
        sc = SimConfig(v=v, dt_sim=0.005, T=12.0, seed=0)
        r = run_case('randomC', 12.0, 0, mode, sc_override=sc,
                     mcfg=(V2_M if mode == 'dmpmpc'
                           else dataclasses.replace(V2_M, use_dmp=False)),
                     dcfg=V2_D)
        speed['v%02d' % int(v)][mode] = r
    sp = speed['v%02d' % int(v)]['passive']['summary']
    for mode in ('mpc', 'dmpmpc'):
        speed['v%02d' % int(v)][mode]['reduction'] = reduction(
            speed['v%02d' % int(v)][mode]['summary'], sp)
    print('v=%2.0f : passive=%.3f  mpc=%+.1f%%  dmp+mpc=%+.1f%%' %
          (v, sp['a_rms'],
           speed['v%02d' % int(v)]['mpc']['reduction']['a_rms'],
           speed['v%02d' % int(v)]['dmpmpc']['reduction']['a_rms']))
save_json(speed, 'results/speed.json')

# ======================================================================
print('==== data for figures ====')
# bump: سه کنترلر — سری کامل 200Hz
data = {}
for mode in ('passive', 'mpc', 'dmpmpc'):
    r = run_case('bump', 4.5, 0, mode, keep_log=True)
    lg = r['_log']
    for kk in ('t', 'zr', 'zs', 'zu', 'a_s', 'ua', 'travel', 'tire'):
        data['bump_%s_%s' % (mode, kk)] = lg[kk]
    if mode == 'dmpmpc':
        data['bump_ref0'] = lg['ref0']
# randomC seed0
for mode in ('passive', 'mpc', 'dmpmpc'):
    r = run_case('randomC', 12.0, 0, mode, keep_log=True)
    lg = r['_log']
    for kk in ('t', 'zr', 'zs', 'zu', 'a_s', 'ua', 'travel', 'tire'):
        data['randomC_%s_%s' % (mode, kk)] = lg[kk]
np.savez_compressed('results/sim_data.npz', **data)
print('ALL DONE [%.0fs]' % (time.time() - t_start))
