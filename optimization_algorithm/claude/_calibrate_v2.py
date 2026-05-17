"""Calibrate expert constraint coefficients v2.

Key fixes from v1:
  1. Use production data 'phase' column (not sim_T phase which says ALL Cementite)
  2. H2 (cr_pearl) in sim_T is REVERSE-correlated with quality -> reduce to near-zero
  3. Focus on TS, EXT, cr_stage1, time_total as discriminators
  4. Keep coefficients reasonable
"""
import sys, os, copy, itertools

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import sim_T.sim_T as sim
from optimization_algorithm import expert_constraints as ec

df = pd.read_excel('工艺数据全.xlsx')

CANDIDATES = [
    (195, 'BEST'), (40, 'BEST'), (33, 'BEST'), (53, 'BEST'), (194, 'BEST'),
    (109, 'MEDIUM'), (107, 'MEDIUM'), (201, 'MEDIUM'), (203, 'MEDIUM'), (104, 'MEDIUM'),
    (173, 'WORST'), (177, 'WORST'), (179, 'WORST'), (181, 'WORST'), (165, 'WORST'),
]

def _sf(v):
    if pd.isna(v): return None
    try: return float(v)
    except: return None

def _np(v):
    if v is None: return None
    if v > 1.0: return v / 100.0
    return v

def run_sim(row):
    ort = _sf(row.get("ORT")) or 850
    for e, a in [("C_ELE","ELM_C"),("SI_ELE","ELM_SI"),("MN_ELE","ELM_MN"),
                  ("NI_ELE","ELM_NI"),("CR_ELE","ELM_CR")]:
        v = _sf(row.get(e))
        if v is not None: setattr(sim.basic_info, a, v/100.0)

    rolls, _ = sim.data_loader.load_roll_data()
    for i in range(1, 11):
        v = _sf(row.get('SPEED{}'.format(i)))
        if v is not None and v > 0:
            ov = rolls[i].roll_v
            rolls[i].roll_v = v
            rolls[i].t = rolls[i].t * (ov/v)
            rolls[i].step = int(rolls[i].t / sim._default_dt)
    ev = _sf(row.get("SPEED1"))
    if ev is not None and ev > 0:
        ov = rolls[0].roll_v; rolls[0].roll_v = ev
        rolls[0].t = rolls[0].t * (ov/ev)
        rolls[0].step = int(rolls[0].t / sim._default_dt)
    for i in range(1, 11):
        v = _sf(row.get('FAN{}'.format(i)))
        v = _np(v)
        if v is not None:
            rolls[i].fan_status = v
            rolls[i].fan_speed = rolls[i].fan_air_volume * v / rolls[i].fan_area

    state, _ = sim.run_full_simulation(rolls, tem1=ort, tem0=ort, dt=0.01)
    vals = ec.extract_from_state(state, sim.basic_info)
    vals['TS'] = row['TS']
    vals['EXT'] = row['EXT']
    vals['ORT'] = row['ORT']
    # USE PRODUCTION DATA PHASE (not sim_T which is broken for 82A)
    vals['prod_phase'] = row['phase']
    return vals

print("Running sim_T on 15 candidates...")
all_vals = {}
for idx, tier in CANDIDATES:
    print("  idx={} ({})...".format(idx, tier), end=' ', flush=True)
    all_vals[idx] = run_sim(df.iloc[idx])
    all_vals[idx]['tier'] = tier
    print("done")

# Show key values
print("\nKey constraint values:")
print("{:<4} {:>6} {:>6} {:>5} {:>6} {:>7} {:>7} {:>5} {:>10} {:>6}".format(
    'idx', 'tier', 'TS', 'EXT', 'S1_cr', 'S2_cr', 'crPost', 'time', 'prod_phase', 'cemF'))
for idx, v in sorted(all_vals.items()):
    print("{:<4} {:>6} {:>6.0f} {:>5.1f} {:>6.1f} {:>7.1f} {:>7.1f} {:>5.1f} {:>10} {:>6.3f}".format(
        idx, v['tier'], v['TS'], v['EXT'],
        v.get('cr_stage1') or 0, v.get('cr_pearl') or 0,
        v.get('cr_post', 0), v.get('time_total', 0),
        v['prod_phase'], v.get('cem_frac', 0)))

# ==============================================================
# COMPUTE SCORE with parameterized config
# ==============================================================
def score(vals, cfg, wb=0.3):
    pen = 0.0; bon = 0.0

    # H2: pearlite CR - LOW weight since sim_T cr_pearl is unreliable
    cp = vals.get('cr_pearl')
    if cp is not None:
        lo, hi = cfg['CR_PEARL_RANGE']
        m = cfg['M_H2']
        if cp < lo: pen += (lo-cp)**2 * m
        elif cp > hi: pen += (cp-hi)**2 * m
    else: pen += cfg['P_H2_NONE']

    # H3: low-T CR - all records ~4.8, never triggers
    c550 = vals.get('cr_550', 0)
    if c550 >= cfg['CR_550_LIMIT']:
        pen += (c550 - cfg['CR_550_LIMIT'])**2 * cfg['M_H3']

    # H4 hard check
    pf = vals.get('pearl_frac', 0)
    if pf < cfg['PEARL_FRAC_MIN']: return float('inf'), 0

    # S1/S2: dT - no discrimination, very low weight
    dtp = vals.get('max_dT_pearl', 0)
    if dtp > cfg['DT_PEARL_MAX']:
        pen += (dtp - cfg['DT_PEARL_MAX'])**2 * cfg['M_S1']
    dtt = vals.get('max_dT_total', 0)
    if dtt > cfg['DT_TOTAL_MAX']:
        pen += (dtt - cfg['DT_TOTAL_MAX'])**2 * cfg['M_S2']

    # S3: post CR - all ~4.2-4.4, no discrimination
    crp = vals.get('cr_post', 0)
    if crp > cfg['CR_POST_MAX']:
        pen += (crp - cfg['CR_POST_MAX'])**2 * cfg['M_S3']

    # S4: cementite from lever rule - very small for 82A
    cf = vals.get('cem_frac', 0)
    if cf > cfg['CEM_FRAC_MAX']:
        pen += (cf - cfg['CEM_FRAC_MAX'])**2 * cfg['M_S4']

    # Phase: use PRODUCTION DATA (discriminates BEST/MED=Ferrite vs WORST=Cementite)
    if vals.get('prod_phase') == 'Cementite':
        pen += cfg['P_CEMENTITE']

    # S7: TS - THE MAIN DISCRIMINATOR
    ts = vals.get('TS')
    if ts is not None:
        tlo, thi = cfg['TS_RANGE']
        if ts < tlo: pen += (tlo-ts)**2 * cfg['M_S7']
        elif ts > thi: pen += (ts-thi)**2 * cfg['M_S7']

    # S8: EXT - SECOND MAIN DISCRIMINATOR
    ext = vals.get('EXT')
    if ext is not None and ext < cfg['Z_MIN']:
        pen += (cfg['Z_MIN']-ext)**2 * cfg['M_S8']

    # S9: stage1 CR - MODERATE DISCRIMINATOR (BEST~17 vs WORST~14)
    cs1 = vals.get('cr_stage1')
    if cs1 is not None:
        s1lo, s1hi = cfg['CR_STAGE1_RANGE']
        if cs1 < s1lo: pen += (s1lo-cs1)**2 * cfg['M_S9']
        elif cs1 > s1hi: pen += (cs1-s1hi)**2 * cfg['M_S9']
    else: pen += cfg['P_S9_NONE']

    # S10: time - WEAK DISCRIMINATOR (BEST~84 vs WORST~90)
    tt = vals.get('time_total', 0)
    if tt > cfg['TIME_MAX']:
        pen += (tt - cfg['TIME_MAX'])**2 * cfg['M_S10']

    # Bonuses
    Tt = vals.get('T_trans')
    if Tt is not None and abs(Tt-cfg['T_TRANS_TARGET']) < cfg['T_TRANS_TOL']:
        bon += cfg['B_B1']
    S0 = vals.get('S0_est')
    if S0 is not None and cfg['S0_RANGE'][0] <= S0 <= cfg['S0_RANGE'][1]:
        bon += cfg['B_B2']
    if pf >= cfg['PEARL_BONUS_MIN']:
        bon += cfg['B_B4']

    return pen - wb * bon, pen, bon

def eval_cfg(cfg, av, wb=0.3):
    scores = {}
    for idx, v in av.items():
        c, p, b = score(v, cfg, wb)
        scores[idx] = (c, p, b)
    best = [scores[i][0] for i, _ in CANDIDATES[:5]]
    med  = [scores[i][0] for i, _ in CANDIDATES[5:10]]
    worst = [scores[i][0] for i, _ in CANDIDATES[10:]]
    bmax, mmin, mmax, wmin = max(best), min(med), max(med), min(worst)
    return {
        'perfect': bmax < mmin and mmax < wmin,
        'gap1': mmin - bmax, 'gap2': wmin - mmax,
        'best': best, 'med': med, 'worst': worst,
        'bmax': bmax, 'mmin': mmin, 'mmax': mmax, 'wmin': wmin,
        'brange': max(best)-min(best), 'mrange': max(med)-min(med), 'wrange': max(worst)-min(worst),
    }

# Current config
BASE = {
    'CR_PEARL_RANGE': (9.0, 12.0), 'M_H2': 5.0, 'P_H2_NONE': 50.0,
    'CR_550_LIMIT': 5.0, 'M_H3': 3.0,
    'PEARL_FRAC_MIN': 0.85,
    'DT_PEARL_MAX': 15.0, 'M_S1': 2.0,
    'DT_TOTAL_MAX': 20.0, 'M_S2': 1.0,
    'CR_POST_MAX': 3.0, 'M_S3': 5.0,
    'CEM_FRAC_MAX': 0.03, 'M_S4': 20.0,
    'P_CEMENTITE': 0.0,
    'TS_RANGE': (980, 1120), 'M_S7': 0.001,
    'Z_MIN': 38.0, 'M_S8': 5.0,
    'CR_STAGE1_RANGE': (20.0, 30.0), 'M_S9': 0.3, 'P_S9_NONE': 10.0,
    'TIME_MAX': 85.0, 'M_S10': 0.005,
    'T_TRANS_TARGET': 655.0, 'T_TRANS_TOL': 10.0, 'B_B1': 5.0,
    'S0_RANGE': (0.12, 0.17), 'B_B2': 3.0,
    'PEARL_BONUS_MIN': 0.90, 'B_B4': 5.0,
}

print("\n" + "=" * 80)
print("CURRENT config")
r = eval_cfg(BASE, all_vals)
print("Perfect: {}, gap1={:.1f}, gap2={:.1f}".format(r['perfect'], r['gap1'], r['gap2']))
print("BEST:   {}".format(['{:.1f}'.format(s) for s in r['best']]))
print("MEDIUM: {}".format(['{:.1f}'.format(s) for s in r['med']]))
print("WORST:  {}".format(['{:.1f}'.format(s) for s in r['worst']]))

# ==============================================================
# MANUAL TUNING (iterative, targeting clear separation)
# ==============================================================
# Design targets:
#   BEST: -5 to +10  (near zero, slightly negative from bonuses)
#   MEDIUM: +30 to +100  (TS penalty dominates)
#   WORST: +200 to +600  (TS + EXT + Cementite + S9 penalties)
#   Clear gap between tiers > 20

TUNED = BASE.copy()
# Reduce non-discriminating constraints
TUNED['M_H2'] = 0.1      # near-zero (sim_T cr_pearl is unreliable)
TUNED['P_H2_NONE'] = 2.0
TUNED['M_H3'] = 0.1      # never triggers anyway
TUNED['M_S1'] = 0.1      # no discrimination
TUNED['M_S2'] = 0.05     # no discrimination
TUNED['M_S3'] = 0.2      # very weak discrimination
TUNED['M_S4'] = 5.0      # cem_frac never exceeds threshold for 82A

# Main discriminators
TUNED['P_CEMENTITE'] = 50.0   # Strong signal from production data phase
TUNED['M_S7'] = 0.03          # TS=1200 -> 80^2*0.03=192, TS=1170 -> 50^2*0.03=75
TUNED['M_S8'] = 2.0           # EXT=34 -> 4^2*2=32, EXT=31 -> 7^2*2=98
TUNED['M_S9'] = 1.0           # cr_S1=14 -> 6^2*1=36, cr_S1=17 -> 3^2*1=9
TUNED['M_S10'] = 0.03         # time=90 -> 5^2*0.03=0.75

# Bonuses - keep modest
TUNED['B_B1'] = 2.0
TUNED['B_B2'] = 1.0
TUNED['B_B4'] = 2.0

print("\n" + "=" * 80)
print("TUNED config v1")
r = eval_cfg(TUNED, all_vals)
print("Perfect: {}, gap1={:.1f}, gap2={:.1f}".format(r['perfect'], r['gap1'], r['gap2']))
print("BEST:   {}".format(['{:.1f}'.format(s) for s in r['best']]))
print("MEDIUM: {}".format(['{:.1f}'.format(s) for s in r['med']]))
print("WORST:  {}".format(['{:.1f}'.format(s) for s in r['worst']]))

# ==============================================================
# GRID SEARCH for optimal key multipliers
# ==============================================================
best_cfg = None
best_metric = -1e9

for m_s7 in [0.02, 0.025, 0.03, 0.04, 0.05]:
    for m_s8 in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for m_s9 in [0.5, 0.8, 1.0, 1.2, 1.5]:
            for p_cem in [30, 40, 50, 60, 80]:
                for m_s10 in [0.01, 0.02, 0.03, 0.05]:
                    cfg = TUNED.copy()
                    cfg['M_S7'] = m_s7
                    cfg['M_S8'] = m_s8
                    cfg['M_S9'] = m_s9
                    cfg['P_CEMENTITE'] = p_cem
                    cfg['M_S10'] = m_s10
                    r = eval_cfg(cfg, all_vals)
                    if not r['perfect']: continue
                    # Desired: gap1>=20, gap2>=100, BEST in [-5,15], MED in [30,150], WORST in [200,600]
                    # Penalize if ranges are too extreme
                    metric = (r['gap1'] + r['gap2'] * 0.5
                              - max(0, max(r['best'])-15)*2   # BEST should be <15
                              + min(0, min(r['best'])+5)*2     # BEST should be >-5
                              - max(0, min(r['med'])-20)*1     # MED should be >20
                              - max(0, max(r['worst'])-800)*0.1)  # WORST not too crazy
                    if metric > best_metric:
                        best_metric = metric
                        best_cfg = cfg.copy()

if best_cfg:
    print("\n" + "=" * 80)
    print("OPTIMAL config from grid search")
    r = eval_cfg(best_cfg, all_vals)
    print("Perfect: {}, gap1={:.1f}, gap2={:.1f}".format(r['perfect'], r['gap1'], r['gap2']))
    print("BEST:   {}".format(['{:.1f}'.format(s) for s in r['best']]))
    print("MEDIUM: {}".format(['{:.1f}'.format(s) for s in r['med']]))
    print("WORST:  {}".format(['{:.1f}'.format(s) for s in r['worst']]))
    print("Spreads: BEST={:.1f}, MED={:.1f}, WORST={:.1f}".format(r['brange'], r['mrange'], r['wrange']))

# ==============================================================
# DETAILED BREAKDOWN with final config
# ==============================================================
FINAL = best_cfg if best_cfg else TUNED
print("\n" + "=" * 80)
print("FINAL DETAILED SCORES")
print("=" * 80)
for tier_name, indices in [('BEST', CANDIDATES[:5]), ('MEDIUM', CANDIDATES[5:10]), ('WORST', CANDIDATES[10:])]:
    print("\n--- {} ---".format(tier_name))
    for idx, _ in indices:
        v = all_vals[idx]
        c, p, b = score(v, FINAL)
        # Manual breakdown
        parts = []
        ts = v['TS']
        if ts < 980: parts.append("TS_lo={:.0f}".format((980-ts)**2*FINAL['M_S7']))
        elif ts > 1120: parts.append("TS_hi={:.0f}".format((ts-1120)**2*FINAL['M_S7']))
        ext = v['EXT']
        if ext < 38: parts.append("EXT={:.0f}".format((38-ext)**2*FINAL['M_S8']))
        cs1 = v.get('cr_stage1')
        if cs1 and cs1 < 20: parts.append("S9={:.0f}".format((20-cs1)**2*FINAL['M_S9']))
        if v.get('prod_phase') == 'Cementite': parts.append("Cem={:.0f}".format(FINAL['P_CEMENTITE']))
        cp = v.get('cr_pearl')
        if cp:
            if cp < 9: parts.append("H2={:.0f}".format((9-cp)**2*FINAL['M_H2']))
            elif cp > 12: parts.append("H2={:.0f}".format((cp-12)**2*FINAL['M_H2']))
        crp = v.get('cr_post', 0)
        if crp > 3: parts.append("S3={:.0f}".format((crp-3)**2*FINAL['M_S3']))
        tt = v.get('time_total', 0)
        if tt > FINAL['TIME_MAX']: parts.append("S10={:.0f}".format((tt-FINAL['TIME_MAX'])**2*FINAL['M_S10']))
        print("  idx={}: TS={:.0f} EXT={:.1f} ph={} S1cr={:.1f} time={:.0f}s | pen={:.0f} bon={:.0f} cost={:.1f} | {}".format(
            idx, ts, ext, v['prod_phase'], cs1 or 0, tt or 0, p, b, c, " + ".join(parts)))

# Print final coefficients table
print("\n" + "=" * 80)
print("FINAL COEFFICIENTS SUMMARY (apply to expert_constraints.py)")
print("=" * 80)
print("M_H2 = {:.1f}     # pearlite zone CR (reduced: sim_T unreliable)".format(FINAL['M_H2']))
print("M_H3 = {:.1f}     # low-T CR (never triggers)".format(FINAL['M_H3']))
print("M_S1 = {:.2f}    # pearlite dT (no discrimination)".format(FINAL['M_S1']))
print("M_S2 = {:.2f}    # overall dT (no discrimination)".format(FINAL['M_S2']))
print("M_S3 = {:.1f}     # post-transformation CR (weak discrimination)".format(FINAL['M_S3']))
print("M_S4 = {:.0f}     # cementite fraction (never exceeds threshold)".format(FINAL['M_S4']))
print("P_CEMENTITE = {:.0f}  # Cementite phase flat penalty (production data phase)".format(FINAL['P_CEMENTITE']))
print("M_S7 = {:.3f}   # TS deviation [MAIN DISCRIMINATOR]".format(FINAL['M_S7']))
print("M_S8 = {:.1f}     # EXT below Z_min [MAIN DISCRIMINATOR]".format(FINAL['M_S8']))
print("M_S9 = {:.1f}     # stage1 CR [MODERATE DISCRIMINATOR]".format(FINAL['M_S9']))
print("M_S10 = {:.3f}   # total time [WEAK DISCRIMINATOR]".format(FINAL['M_S10']))
print("W_BONUS = 0.3")
