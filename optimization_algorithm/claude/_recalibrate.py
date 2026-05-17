"""Recalibrate constraint coefficients to hit target score ranges.

Targets: BEST 0-9, MEDIUM 10-29, WORST 30-40
Method: 30 labeled records from best_process_data.txt, grid search over key multipliers.
Switches TS/EXT to LINEAR penalties for more controllable separation.
"""
import sys, os

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import sim_T.sim_T as sim

df = pd.read_excel('工艺数据全.xlsx')

# 30 labeled indices from best_process_data.txt
BEST_IDX  = [195, 40, 53, 59, 33, 194, 196, 35, 51, 26]
MED_IDX   = [99, 15, 98, 100, 27, 101, 102, 103, 8, 6]
WORST_IDX = [132, 159, 133, 166, 164, 134, 208, 135, 213, 136]

ALL_IDX = BEST_IDX + MED_IDX + WORST_IDX

# Quick sim on all 30
def _sf(v):
    if pd.isna(v): return None
    try: return float(v)
    except: return None

def _np(v):
    if v is None: return None
    if v > 1.0: return v / 100.0
    return v

def sim_one(idx):
    row = df.iloc[idx]
    ort = _sf(row.get("ORT")) or 850
    for e, a in [("C_ELE","ELM_C"),("SI_ELE","ELM_SI"),("MN_ELE","ELM_MN"),
                  ("NI_ELE","ELM_NI"),("CR_ELE","ELM_CR")]:
        v = _sf(row.get(e))
        if v is not None: setattr(sim.basic_info, a, v/100.0)
    rolls, _ = sim.data_loader.load_roll_data()
    for i in range(1, 11):
        v = _sf(row.get('SPEED{}'.format(i)))
        if v is not None and v > 0:
            ov = rolls[i].roll_v; rolls[i].roll_v = v
            rolls[i].t = rolls[i].t*(ov/v)
            rolls[i].step = int(rolls[i].t / sim._default_dt)
    ev = _sf(row.get("SPEED1"))
    if ev is not None and ev > 0:
        ov = rolls[0].roll_v; rolls[0].roll_v = ev
        rolls[0].t = rolls[0].t*(ov/ev)
        rolls[0].step = int(rolls[0].t / sim._default_dt)
    for i in range(1, 11):
        v = _sf(row.get('FAN{}'.format(i)))
        v = _np(v)
        if v is not None:
            rolls[i].fan_status = v
            rolls[i].fan_speed = rolls[i].fan_air_volume * v / rolls[i].fan_area
    state, _ = sim.run_full_simulation(rolls, tem1=ort, tem0=ort, dt=0.01)
    return state

# Collect constraint values (using direct extraction, not expert_constraints functions)
print("Running sim_T on 30 labeled records...")
raw_data = {}
for idx in ALL_IDX:
    state = sim_one(idx)
    T0 = np.array(state.history_T_0[-1], dtype=np.float64)
    T1 = np.array(state.history_T_1[-1], dtype=np.float64)
    t = np.array(state.history_time, dtype=np.float64)
    A1 = float(sim.basic_info.A1)
    Bs = 550.0

    # Use T0 (full length) for temperature ranges, then compute CR on the masked segments
    def avg_cr_temp(T_subset, t_subset):
        if len(T_subset) < 2: return None
        dt_arr = np.diff(t_subset); dT_arr = np.diff(T_subset)
        cr = np.abs(dT_arr / np.where(dt_arr > 0, dt_arr, np.inf))
        return float(np.mean(cr))

    # Stage 1: T > A1
    m1 = T0 >= A1
    cr_s1 = avg_cr_temp(T0[m1], t[m1]) if m1.sum() >= 2 else None
    # Stage 2: A1 ~ Bs
    m2 = (T0 <= A1) & (T0 >= Bs)
    cr_s2 = avg_cr_temp(T0[m2], t[m2]) if m2.sum() >= 2 else None
    # T < 550 cooling rate (MAX)
    m_low = T0 < 550.0
    if m_low.sum() >= 2:
        dt_low = np.diff(t[m_low]); dT_low = np.diff(T0[m_low])
        cr_550 = float(np.max(np.abs(dT_low / np.where(dt_low > 0, dt_low, np.inf))))
    else:
        cr_550 = 0.0
    # dT
    dT_arr = np.abs(T0 - T1)
    max_dT = float(np.max(dT_arr))
    # dT in pearlite zone
    m2_full = (T0 <= A1) & (T0 >= Bs)
    max_dTp = float(np.max(dT_arr[m2_full])) if m2_full.sum() > 0 else max_dT
    # T_end
    T_end = float(T0[-1])
    # time_total
    time_tot = float(t[-1])
    # pearl_frac
    pf = float(state.pearlite_0[-1][-1]) if len(state.pearlite_0[-1]) > 0 else 0.0
    # cem_frac
    f_ferrite = float(state.ferrite_final_0[-1])
    C_pct = sim.basic_info.ELM_C * 100
    cem = max(0.0, (C_pct - 0.77) / 5.9) if f_ferrite < 0.01 else 0.0
    # S0_est
    if cr_s2 and cr_s2 > 0:
        S0 = float(np.exp(np.log(0.197) - 0.38 * (np.log(max(cr_s2, 0.5)) - np.log(4.2))))
    else:
        S0 = None
    # T_trans
    pearl_hist = np.array(state.pearlite_0[-1], dtype=np.float64)
    if len(pearl_hist) > 0 and pearl_hist[-1] > 0.5:
        hi = int(np.argmax(pearl_hist > 0.5))
        T_trans = float(T0[hi]) if hi < len(T0) else None
    else:
        T_trans = None

    row = df.iloc[idx]
    raw_data[idx] = {
        'TS': row['TS'], 'EXT': row['EXT'], 'ORT': row['ORT'],
        'cr_s1': cr_s1, 'cr_s2': cr_s2, 'cr_550': cr_550,
        'max_dT': max_dT, 'max_dTp': max_dTp, 'T_end': T_end,
        'time_tot': time_tot, 'pearl_frac': pf, 'cem_frac': cem,
        'S0': S0, 'T_trans': T_trans, 'phase': row['phase'],
    }
    print("  idx={} done".format(idx), end=' ' if idx != ALL_IDX[-1] else '\n', flush=True)

# ================================================================
# SCORE FUNCTION with parameterized coefficients
# ================================================================
def compute_score(v, cfg):
    pen = 0.0; bon = 0.0

    # H2: cr_pearl
    cp = v['cr_s2']
    if cp is not None:
        lo, hi = cfg['CR_PEARL']
        if cp < lo: pen += (lo-cp)**2 * cfg['M_H2']
        elif cp > hi: pen += (cp-hi)**2 * cfg['M_H2']
    else: pen += cfg['P_H2_NONE']

    # H3: cr_550
    if v['cr_550'] >= cfg['CR_550']:
        pen += (v['cr_550'] - cfg['CR_550'])**2 * cfg['M_H3']

    # H4
    if v['pearl_frac'] < cfg['PEARL_FRAC']: return 99999

    # S1, S2
    if v['max_dTp'] > cfg['DT_PEARL']:
        pen += (v['max_dTp'] - cfg['DT_PEARL'])**2 * cfg['M_S1']
    if v['max_dT'] > cfg['DT_TOTAL']:
        pen += (v['max_dT'] - cfg['DT_TOTAL'])**2 * cfg['M_S2']

    # S4: cem_frac linear
    if v['cem_frac'] > cfg['CEM_THR']:
        pen += (v['cem_frac'] - cfg['CEM_THR']) * cfg['M_S4']

    # S7: TS (quadratic for spread amplification)
    ts = v['TS']
    if ts < cfg['TS_LO']: pen += (cfg['TS_LO'] - ts)**2 * cfg['M_S7']
    elif ts > cfg['TS_HI']: pen += (ts - cfg['TS_HI'])**2 * cfg['M_S7']

    # S8: EXT (quadratic)
    ext = v['EXT']
    if ext < cfg['Z_MIN']: pen += (cfg['Z_MIN'] - ext)**2 * cfg['M_S8']

    # S9: cr_stage1 (quadratic)
    cs1 = v['cr_s1']
    if cs1 is not None:
        s1lo, s1hi = cfg['CR_STAGE1']
        if cs1 < s1lo: pen += (s1lo - cs1)**2 * cfg['M_S9']
        elif cs1 > s1hi: pen += (cs1 - s1hi)**2 * cfg['M_S9']
    else: pen += cfg['P_S9_NONE']

    # S10: time
    if v['time_tot'] > cfg['TIME_MAX']:
        pen += (v['time_tot'] - cfg['TIME_MAX'])**2 * cfg['M_S10']

    # S11: T_end
    if v['T_end'] > cfg['T_END_MAX']:
        pen += (v['T_end'] - cfg['T_END_MAX'])**2 * cfg['M_S11']

    # Bonuses
    Tt = v['T_trans']
    if Tt is not None and abs(Tt - cfg['T_TRANS_TARGET']) < cfg['T_TRANS_TOL']:
        bon += cfg['B_B1']
    S0 = v['S0']
    if S0 is not None and cfg['S0_LO'] <= S0 <= cfg['S0_HI']:
        bon += cfg['B_B2']
    if v['pearl_frac'] >= cfg['PEARL_BONUS']:
        bon += cfg['B_B4']

    return pen - cfg['W_BONUS'] * bon

# ================================================================
# GRID SEARCH
# ================================================================
# Use linear TS/EXT for tight control over tier separation
BASE_CFG = {
    # Ranges
    'CR_PEARL': (9.0, 12.0), 'CR_550': 5.0, 'PEARL_FRAC': 0.85,
    'DT_PEARL': 15.0, 'DT_TOTAL': 20.0,
    'CEM_THR': 0.012,
    'TS_LO': 980, 'TS_HI': 1120, 'Z_MIN': 38.0,
    'CR_STAGE1': (20.0, 30.0),
    'TIME_MAX': 85.0, 'T_END_MAX': 400.0,
    'T_TRANS_TARGET': 655.0, 'T_TRANS_TOL': 10.0,
    'S0_LO': 0.12, 'S0_HI': 0.17, 'PEARL_BONUS': 0.90,
    # Multipliers to search
    'M_H2': 0.1, 'P_H2_NONE': 5.0,
    'M_H3': 0.1,
    'M_S1': 0.1, 'M_S2': 0.05,
    'M_S4': 500.0,    # cem_frac linear
    'M_S7': 0.004,    # TS quadratic — KEY parameter
    'M_S8': 0.3,      # EXT quadratic — KEY parameter
    'M_S9': 0.15,     # cr_stage1 quadratic
    'P_S9_NONE': 5.0,
    'M_S10': 0.005,
    'M_S11': 0.5,
    'B_B1': 2.0, 'B_B2': 1.0, 'B_B4': 2.0,
    'W_BONUS': 0.3,
}

# Search grid over key parameters
best_cfg = None
best_ok = 0

for m_s7 in [0.002, 0.003, 0.004, 0.005, 0.006, 0.008]:
    for m_s8 in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        for m_s9 in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
            for m_s4 in [500, 800, 1000, 1200, 1500, 2000]:
                cfg = BASE_CFG.copy()
                cfg['M_S7'] = m_s7
                cfg['M_S8'] = m_s8
                cfg['M_S9'] = m_s9
                cfg['M_S4'] = m_s4

                best_scores = [compute_score(raw_data[i], cfg) for i in BEST_IDX]
                med_scores  = [compute_score(raw_data[i], cfg) for i in MED_IDX]
                worst_scores = [compute_score(raw_data[i], cfg) for i in WORST_IDX]

                # Check target ranges
                ok = 0
                if all(0 <= s <= 9 for s in best_scores): ok += 1
                if all(10 <= s <= 29 for s in med_scores): ok += 1
                if all(30 <= s <= 40 for s in worst_scores): ok += 1

                if ok >= best_ok:
                    # Tie-break: prefer smaller within-tier spread
                    spread = (max(best_scores)-min(best_scores) +
                              max(med_scores)-min(med_scores) +
                              max(worst_scores)-min(worst_scores))
                    if ok > best_ok or (ok == best_ok and (best_cfg is None or spread < best_spread)):
                        best_ok = ok
                        best_cfg = cfg.copy()
                        best_spread = spread
                        best_cfg['_best'] = best_scores
                        best_cfg['_med'] = med_scores
                        best_cfg['_worst'] = worst_scores

print("\n" + "=" * 70)
print("GRID SEARCH RESULTS")
print("=" * 70)
if best_cfg:
    print("Tiers satisfied: {}/3".format(best_ok))
    print("BEST scores:   {}".format(['{:.1f}'.format(s) for s in best_cfg['_best']]))
    print("MEDIUM scores: {}".format(['{:.1f}'.format(s) for s in best_cfg['_med']]))
    print("WORST scores:  {}".format(['{:.1f}'.format(s) for s in best_cfg['_worst']]))
    print("\nKey multipliers:")
    print("  M_S7 (TS linear) = {:.3f}".format(best_cfg['M_S7']))
    print("  M_S8 (EXT linear) = {:.1f}".format(best_cfg['M_S8']))
    print("  M_S9 (cr_stage1 quad) = {:.2f}".format(best_cfg['M_S9']))
    print("  M_S4 (cem_frac linear) = {:.0f}".format(best_cfg['M_S4']))

# If all 3 tiers satisfied, also print full config
if best_ok == 3:
    print("\n" + "=" * 70)
    print("FINAL CALIBRATED COEFFICIENTS")
    print("=" * 70)
    for k, v in sorted(best_cfg.items()):
        if not k.startswith('_'):
            print("  {} = {}".format(k, v))
else:
    print("\nBest result: {}/3 tiers satisfied".format(best_ok))
    print("Relaxing constraints...")
    # Show which records violate
    for label, indices, lo, hi in [('BEST', BEST_IDX, 0, 9), ('MEDIUM', MED_IDX, 10, 29), ('WORST', WORST_IDX, 30, 40)]:
        scores = [compute_score(raw_data[i], best_cfg) for i in indices]
        violators = [(i, s) for i, s in zip(indices, scores) if s < lo or s > hi]
        if violators:
            print("  {} violations: {}".format(label, [(i, '{:.1f}'.format(s)) for i, s in violators]))
