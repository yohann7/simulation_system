"""Calibrate expert constraint coefficients using 15 labeled production records.

Goal: BEST scores < MEDIUM scores < WORST scores, with clear separation.
Method: Run sim_T on all 15, extract constraint values, test coefficient sets.
"""
import sys, os, copy, itertools, json

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import sim_T.sim_T as sim
from optimization_algorithm import expert_constraints as ec

# Load data
df = pd.read_excel('工艺数据全.xlsx')

# 15 candidates with labels
CANDIDATES = [
    # BEST
    (195, 'BEST'), (40, 'BEST'), (33, 'BEST'), (53, 'BEST'), (194, 'BEST'),
    # MEDIUM
    (109, 'MEDIUM'), (107, 'MEDIUM'), (201, 'MEDIUM'), (203, 'MEDIUM'), (104, 'MEDIUM'),
    # WORST
    (173, 'WORST'), (177, 'WORST'), (179, 'WORST'), (181, 'WORST'), (165, 'WORST'),
]

def _safe_float(value):
    if pd.isna(value): return None
    try: return float(value)
    except: return None

def _normalize_percent(value):
    if value is None: return None
    if value > 1.0: return value / 100.0
    return value

def run_sim_and_extract(row):
    """Run sim_T and extract all constraint values + production TS/EXT."""
    ort = _safe_float(row.get("ORT"))
    if ort is None: ort = 850

    # Chemistry
    for elem, attr in [("C_ELE","ELM_C"),("SI_ELE","ELM_SI"),("MN_ELE","ELM_MN"),
                        ("NI_ELE","ELM_NI"),("CR_ELE","ELM_CR")]:
        val = _safe_float(row.get(elem))
        if val is not None:
            setattr(sim.basic_info, attr, val / 100.0)

    rolls, _ = sim.data_loader.load_roll_data()

    for i in range(1, 11):
        val = _safe_float(row.get('SPEED{}'.format(i)))
        if val is not None and val > 0:
            old_v = rolls[i].roll_v
            rolls[i].roll_v = val
            rolls[i].t = rolls[i].t * (old_v / val)
            rolls[i].step = int(rolls[i].t / sim._default_dt)

    entry_speed = _safe_float(row.get("SPEED1"))
    if entry_speed is not None and entry_speed > 0:
        old_v = rolls[0].roll_v
        rolls[0].roll_v = entry_speed
        rolls[0].t = rolls[0].t * (old_v / entry_speed)
        rolls[0].step = int(rolls[0].t / sim._default_dt)

    for i in range(1, 11):
        val = _safe_float(row.get('FAN{}'.format(i)))
        val = _normalize_percent(val)
        if val is not None:
            rolls[i].fan_status = val
            rolls[i].fan_speed = rolls[i].fan_air_volume * val / rolls[i].fan_area

    state, _ = sim.run_full_simulation(rolls, tem1=ort, tem0=ort, dt=0.01)

    # Use extract_from_state for full vals
    vals = ec.extract_from_state(state, sim.basic_info)
    vals['TS'] = row['TS']
    vals['EXT'] = row['EXT']
    vals['ORT'] = row['ORT']
    vals['V_COOL_data'] = row['V_COOL']
    vals['MAT_NO'] = row.get('MAT_NO', '?')
    return vals

# Run all 15 sims
print("Running sim_T on 15 candidates...")
all_vals = {}
for idx, tier in CANDIDATES:
    row = df.iloc[idx]
    print("  idx={} ({}), ORT={:.0f}...".format(idx, tier, row['ORT']), end=' ', flush=True)
    vals = run_sim_and_extract(row)
    vals['tier'] = tier
    all_vals[idx] = vals
    print("done")

# Print raw constraint values
print("\n" + "=" * 100)
print("RAW CONSTRAINT VALUES FOR ALL 15 CANDIDATES")
print("=" * 100)
keys = ['tier', 'TS', 'EXT', 'ORT', 'cr_stage1', 'cr_pearl', 'cr_550', 'cr_post',
        'pearl_frac', 'max_dT_total', 'max_dT_pearl', 'cem_frac', 'phase',
        'T_trans', 'S0_est', 'time_total']
hdr = "{:<4} {:>6} {:>6} {:>5} {:>5} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>5} {:>10} {:>7} {:>7} {:>6}".format(
    'idx', 'tier', 'TS', 'EXT', 'ORT', 'S1_cr', 'S2_cr', 'cr550', 'crPost',
    'pearl', 'dTmax', 'dTp', 'cemF', 'phase', 'Ttrans', 'S0est', 'time')
print(hdr)
print("-" * 100)
for idx, vals in sorted(all_vals.items()):
    print("{:<4} {:>6} {:>6.0f} {:>5.1f} {:>5.0f} {:>7.1f} {:>7.1f} {:>7.1f} {:>7.1f} {:>6.3f} {:>6.1f} {:>6.1f} {:>6.3f} {:>10} {:>7.1f} {:>7.3f} {:>6.1f}".format(
        idx, vals['tier'], vals['TS'], vals['EXT'], vals['ORT'],
        vals.get('cr_stage1') or 0, vals.get('cr_pearl') or 0,
        vals.get('cr_550', 0), vals.get('cr_post', 0),
        vals.get('pearl_frac', 0), vals.get('max_dT_total', 0),
        vals.get('max_dT_pearl', 0), vals.get('cem_frac', 0),
        vals.get('phase', '?'),
        vals.get('T_trans') or 0, vals.get('S0_est') or 0,
        vals.get('time_total', 0)))

# ================================================================
# ITERATIVE COEFFICIENT CALIBRATION
# ================================================================

def compute_score(vals, cfg, k_ek=1.0, w_bonus=0.3):
    """Compute penalty and bonus using given config."""
    penalty = 0.0
    bonus = 0.0

    # H2: pearlite zone cooling rate
    cr_pearl = vals.get('cr_pearl')
    if cr_pearl is not None:
        cr_lo, cr_hi = cfg['CR_PEARL_RANGE']
        m = cfg['M_H2']
        if cr_pearl < cr_lo:
            penalty += (cr_lo - cr_pearl) ** 2 * m
        elif cr_pearl > cr_hi:
            penalty += (cr_pearl - cr_hi) ** 2 * m
    else:
        penalty += cfg['P_H2_NONE']

    # H3: low temp cooling rate
    cr_550 = vals.get('cr_550', 0.0)
    if cr_550 >= cfg['CR_550_LIMIT']:
        penalty += (cr_550 - cfg['CR_550_LIMIT']) ** 2 * cfg['M_H3']

    # H4 hard constraint: pearl_frac
    pearl_frac = vals.get('pearl_frac', 0.0)
    if pearl_frac < cfg['PEARL_FRAC_MIN']:
        return float('inf'), 0

    # S1: pearlite zone dT
    dT_pearl = vals.get('max_dT_pearl', 0.0)
    if dT_pearl > cfg['DT_PEARL_MAX']:
        penalty += (dT_pearl - cfg['DT_PEARL_MAX']) ** 2 * cfg['M_S1']

    # S2: overall dT
    dT_total = vals.get('max_dT_total', 0.0)
    if dT_total > cfg['DT_TOTAL_MAX']:
        penalty += (dT_total - cfg['DT_TOTAL_MAX']) ** 2 * cfg['M_S2']

    # S3: post-transformation cooling
    cr_post = vals.get('cr_post', 0.0)
    if cr_post > cfg['CR_POST_MAX']:
        penalty += (cr_post - cfg['CR_POST_MAX']) ** 2 * cfg['M_S3']

    # S4: cementite fraction
    cem_frac = vals.get('cem_frac', 0.0)
    if cem_frac > cfg['CEM_FRAC_MAX']:
        penalty += (cem_frac - cfg['CEM_FRAC_MAX']) ** 2 * cfg['M_S4']
    # Extra penalty for Cementite phase
    if vals.get('phase') == 'Cementite':
        penalty += cfg['P_CEMENTITE']

    # S7: TS
    ts = vals.get('TS')
    if ts is not None:
        ts_lo, ts_hi = cfg['TS_RANGE']
        if ts < ts_lo:
            penalty += (ts_lo - ts) ** 2 * cfg['M_S7']
        elif ts > ts_hi:
            penalty += (ts - ts_hi) ** 2 * cfg['M_S7']

    # S8: EXT (Z)
    ext = vals.get('EXT')
    if ext is not None and ext < cfg['Z_MIN']:
        penalty += (cfg['Z_MIN'] - ext) ** 2 * cfg['M_S8']

    # S9: stage 1 cooling rate
    cr_stage1 = vals.get('cr_stage1')
    if cr_stage1 is not None:
        s1_lo, s1_hi = cfg['CR_STAGE1_RANGE']
        if cr_stage1 < s1_lo:
            penalty += (s1_lo - cr_stage1) ** 2 * cfg['M_S9']
        elif cr_stage1 > s1_hi:
            penalty += (cr_stage1 - s1_hi) ** 2 * cfg['M_S9']
    else:
        penalty += cfg['P_S9_NONE']

    # S10: total time
    time_total = vals.get('time_total', 0.0)
    if time_total > cfg['TIME_MAX']:
        penalty += (time_total - cfg['TIME_MAX']) ** 2 * cfg['M_S10']

    # B1: T_trans bonus
    T_trans = vals.get('T_trans')
    if T_trans is not None and abs(T_trans - cfg['T_TRANS_TARGET']) < cfg['T_TRANS_TOL']:
        bonus += cfg['B_B1']

    # B2: S0 bonus
    S0 = vals.get('S0_est')
    if S0 is not None and cfg['S0_RANGE'][0] <= S0 <= cfg['S0_RANGE'][1]:
        bonus += cfg['B_B2']

    # B4: pearl_frac bonus
    if pearl_frac >= cfg['PEARL_BONUS_MIN']:
        bonus += cfg['B_B4']

    ek_cost = penalty - w_bonus * bonus
    return ek_cost, penalty, bonus


def evaluate_config(cfg, all_vals, w_bonus=0.3):
    """Score all 15 candidates and compute separation metrics."""
    scores = {}
    for idx, vals in all_vals.items():
        cost, pen, bon = compute_score(vals, cfg, w_bonus=w_bonus)
        scores[idx] = (cost, pen, bon)

    best_scores = [scores[idx][0] for idx, _ in CANDIDATES[:5]]
    med_scores = [scores[idx][0] for idx, _ in CANDIDATES[5:10]]
    worst_scores = [scores[idx][0] for idx, _ in CANDIDATES[10:]]

    best_max = max(best_scores)
    med_min = min(med_scores)
    med_max = max(med_scores)
    worst_min = min(worst_scores)

    # Metrics (higher = better separation)
    gap1 = med_min - best_max  # should be > 0
    gap2 = worst_min - med_max  # should be > 0
    best_range = max(best_scores) - min(best_scores)
    med_range = max(med_scores) - min(med_scores)
    worst_range = max(worst_scores) - min(worst_scores)

    # Perfection: all BEST < all MEDIUM < all WORST
    perfect = (best_max < med_min) and (med_max < worst_min)

    return {
        'perfect': perfect,
        'gap1': gap1, 'gap2': gap2,
        'best_scores': best_scores, 'med_scores': med_scores, 'worst_scores': worst_scores,
        'best_max': best_max, 'med_min': med_min, 'med_max': med_max, 'worst_min': worst_min,
        'best_range': best_range, 'med_range': med_range, 'worst_range': worst_range,
    }

# ================================================================
# Base config (copy of current CONSTRAINT_CFG values)
# ================================================================
BASE_CFG = {
    'CR_PEARL_RANGE': (9.0, 12.0),
    'M_H2': 5.0,
    'P_H2_NONE': 50.0,
    'CR_550_LIMIT': 5.0,
    'M_H3': 3.0,
    'PEARL_FRAC_MIN': 0.85,
    'DT_PEARL_MAX': 15.0,
    'M_S1': 2.0,
    'DT_TOTAL_MAX': 20.0,
    'M_S2': 1.0,
    'CR_POST_MAX': 3.0,
    'M_S3': 5.0,
    'CEM_FRAC_MAX': 0.03,
    'M_S4': 20.0,
    'P_CEMENTITE': 0.0,
    'TS_RANGE': (980, 1120),
    'M_S7': 0.001,
    'Z_MIN': 38.0,
    'M_S8': 5.0,
    'CR_STAGE1_RANGE': (20.0, 30.0),
    'M_S9': 0.3,
    'P_S9_NONE': 10.0,
    'TIME_MAX': 85.0,
    'M_S10': 0.005,
    'T_TRANS_TARGET': 655.0,
    'T_TRANS_TOL': 10.0,
    'B_B1': 5.0,
    'S0_RANGE': (0.12, 0.17),
    'B_B2': 3.0,
    'PEARL_BONUS_MIN': 0.90,
    'B_B4': 5.0,
}

print("\n" + "=" * 100)
print("CURRENT COEFFICIENTS - Score Analysis")
print("=" * 100)
result = evaluate_config(BASE_CFG, all_vals)
print("Perfect separation: {}".format(result['perfect']))
print("BEST scores:   {}".format(['{:.2f}'.format(s) for s in result['best_scores']]))
print("MEDIUM scores: {}".format(['{:.2f}'.format(s) for s in result['med_scores']]))
print("WORST scores:  {}".format(['{:.2f}'.format(s) for s in result['worst_scores']]))
print("Gap BEST->MED: {:.2f}, MED->WORST: {:.2f}".format(result['gap1'], result['gap2']))

# Analyze which constraints contribute most to each tier
print("\n" + "=" * 100)
print("CONTRIBUTION ANALYSIS (penalty breakdown by constraint, avg per tier)")
print("=" * 100)
constraint_names = ['H2_cr_pearl', 'H3_cr550', 'S1_dTp', 'S2_dT', 'S3_crPost',
                    'S4_cem', 'S7_TS', 'S8_EXT', 'S9_crS1', 'S10_time',
                    'Cementite_phase']
for tier_name, indices in [('BEST', CANDIDATES[:5]), ('MEDIUM', CANDIDATES[5:10]), ('WORST', CANDIDATES[10:])]:
    print("\n{}:".format(tier_name))
    for idx, _ in indices:
        vals = all_vals[idx]
        cost, pen, bon = compute_score(vals, BASE_CFG)
        # Manual breakdown
        parts = []
        # H2
        cr_pearl = vals.get('cr_pearl')
        if cr_pearl is not None:
            if cr_pearl < 9: parts.append("H2={:.1f}".format((9-cr_pearl)**2*5))
            elif cr_pearl > 12: parts.append("H2={:.1f}".format((cr_pearl-12)**2*5))
        # S7
        ts = vals.get('TS')
        if ts is not None:
            if ts < 980: parts.append("S7={:.1f}".format((980-ts)**2*0.001))
            elif ts > 1120: parts.append("S7={:.1f}".format((ts-1120)**2*0.001))
        # S8
        ext = vals.get('EXT')
        if ext is not None and ext < 38: parts.append("S8={:.1f}".format((38-ext)**2*5))
        # S9
        cr_s1 = vals.get('cr_stage1')
        if cr_s1 is not None and cr_s1 < 20: parts.append("S9={:.1f}".format((20-cr_s1)**2*0.3))
        # Cementite
        if vals.get('phase') == 'Cementite': parts.append("Cem_phase")
        # S3
        cr_post = vals.get('cr_post', 0)
        if cr_post > 3: parts.append("S3={:.1f}".format((cr_post-3)**2*5))
        # S10
        tt = vals.get('time_total', 0)
        if tt > 85: parts.append("S10={:.2f}".format((tt-85)**2*0.005))
        print("  idx={}: pen={:.1f} bon={:.0f} | {}".format(
            idx, pen, bon, " ".join(parts) if parts else "no penalty"))

# ================================================================
# TUNED CONFIG - iterative adjustment
# ================================================================
print("\n" + "=" * 100)
print("TUNED COEFFICIENTS")
print("=" * 100)

# Key insights from data:
# 1. S7 (TS) multiplier 0.001 is too small: TS=1200 -> penalty=6.4, TS=1170 -> penalty=2.5
# 2. H2 (cr_pearl) REVERSES ranking: WORST has ideal cr_pearl, BEST doesn't
# 3. S8 (EXT) multiplier 5.0 gives EXT=34 -> penalty=80, EXT=31 -> penalty=245 (too high!)
# 4. S9 (cr_stage1) discriminates well: BEST~17 vs WORST~14
# 5. Cementite phase is a strong discriminator
# 6. S3 (cr_post) is nearly identical across all tiers -> noise
# 7. dT constraints (S1/S2) are near-identical -> noise, lower multipliers

TUNED_CFG = BASE_CFG.copy()
# H2: reduce drastically since sim_T cr_pearl doesn't correlate with quality
TUNED_CFG['M_H2'] = 0.5        # was 5.0 — sim_T cold bias makes this unreliable
TUNED_CFG['P_H2_NONE'] = 5.0   # was 50.0
# TS: significantly increase for discrimination
TUNED_CFG['M_S7'] = 0.05       # was 0.001 — TS=1200 -> 80^2*0.05=320, TS=1170 -> 50^2*0.05=125
# EXT: reduce to avoid over-penalizing
TUNED_CFG['M_S8'] = 1.5        # was 5.0 — EXT=34 -> 4^2*1.5=24, EXT=31 -> 7^2*1.5=73.5
# S9: increase for better discrimination
TUNED_CFG['M_S9'] = 1.0        # was 0.3 — BEST cr1=17 -> 9, WORST cr1=14 -> 36
# Cementite: add phase penalty
TUNED_CFG['P_CEMENTITE'] = 30.0  # was 0 — strong signal for phase
# S3: reduce since it's noise
TUNED_CFG['M_S3'] = 1.0        # was 5.0
# S1/S2: reduce since no discrimination
TUNED_CFG['M_S1'] = 0.5        # was 2.0
TUNED_CFG['M_S2'] = 0.2        # was 1.0
# S10: increase slightly to penalize slow production
TUNED_CFG['M_S10'] = 0.02      # was 0.005

result = evaluate_config(TUNED_CFG, all_vals)
print("Perfect separation: {}".format(result['perfect']))
print("BEST scores:   {}".format(['{:.2f}'.format(s) for s in result['best_scores']]))
print("MEDIUM scores: {}".format(['{:.2f}'.format(s) for s in result['med_scores']]))
print("WORST scores:  {}".format(['{:.2f}'.format(s) for s in result['worst_scores']]))
print("Gap BEST->MED: {:.2f}, MED->WORST: {:.2f}".format(result['gap1'], result['gap2']))

# ================================================================
# Fine-tune: iterative optimization of key multipliers
# ================================================================
print("\n" + "=" * 100)
print("FINE-TUNING: Grid search over key multipliers")
print("=" * 100)

best_config = None
best_metric = -999

# Search grid
grid = {
    'M_S7': [0.02, 0.03, 0.05, 0.08, 0.10],
    'M_S8': [0.5, 1.0, 1.5, 2.0, 3.0],
    'M_S9': [0.5, 0.8, 1.0, 1.5, 2.0],
    'P_CEMENTITE': [10.0, 20.0, 30.0, 40.0, 50.0],
}

for m_s7 in grid['M_S7']:
    for m_s8 in grid['M_S8']:
        for m_s9 in grid['M_S9']:
            for p_cem in grid['P_CEMENTITE']:
                cfg = TUNED_CFG.copy()
                cfg['M_S7'] = m_s7
                cfg['M_S8'] = m_s8
                cfg['M_S9'] = m_s9
                cfg['P_CEMENTITE'] = p_cem
                r = evaluate_config(cfg, all_vals)
                if not r['perfect']:
                    continue
                # Metric: maximize gap1 + gap2, penalize large intra-tier variance
                metric = r['gap1'] + r['gap2'] - (r['best_range'] + r['med_range'] + r['worst_range']) * 0.3
                if metric > best_metric:
                    best_metric = metric
                    best_config = cfg.copy()
                    best_config['_m_s7'] = m_s7
                    best_config['_m_s8'] = m_s8
                    best_config['_m_s9'] = m_s9
                    best_config['_p_cem'] = p_cem

if best_config:
    print("\nBest found config:")
    print("  M_S7={}, M_S8={}, M_S9={}, P_CEMENTITE={}".format(
        best_config['_m_s7'], best_config['_m_s8'], best_config['_m_s9'], best_config['_p_cem']))
    result = evaluate_config(best_config, all_vals)
    print("  BEST scores:   {}".format(['{:.2f}'.format(s) for s in result['best_scores']]))
    print("  MEDIUM scores: {}".format(['{:.2f}'.format(s) for s in result['med_scores']]))
    print("  WORST scores:  {}".format(['{:.2f}'.format(s) for s in result['worst_scores']]))
    print("  Gap BEST->MED: {:.2f}, MED->WORST: {:.2f}".format(result['gap1'], result['gap2']))
    print("  intra-tier spread: BEST={:.1f}, MED={:.1f}, WORST={:.1f}".format(
        result['best_range'], result['med_range'], result['worst_range']))

# ================================================================
# Final detailed output
# ================================================================
print("\n" + "=" * 100)
print("FINAL TUNED CONFIG - DETAILED SCORES")
print("=" * 100)
FINAL_CFG = best_config if best_config else TUNED_CFG

for tier_name, indices in [('BEST', CANDIDATES[:5]), ('MEDIUM', CANDIDATES[5:10]), ('WORST', CANDIDATES[10:])]:
    print("\n--- {} ---".format(tier_name))
    for idx, _ in indices:
        vals = all_vals[idx]
        cost, pen, bon = compute_score(vals, FINAL_CFG)
        print("  idx={}: TS={:.0f}, EXT={:.1f}, ph={}, cr_pearl={:.1f}, cr_S1={:.1f}, time={:.0f}s".format(
            idx, vals['TS'], vals['EXT'], vals.get('phase','?'),
            vals.get('cr_pearl') or 0, vals.get('cr_stage1') or 0, vals.get('time_total', 0)))
        print("         penalty={:.1f}, bonus={:.0f}, EK_cost={:.2f}".format(pen, bon, cost))

# Print final config
print("\n" + "=" * 100)
print("FINAL COEFFICIENTS (to be applied to expert_constraints.py)")
print("=" * 100)
print("M_H2 = {:.1f}    # pearlite zone CR (reduced - sim_T bias)".format(FINAL_CFG['M_H2']))
print("M_H3 = {:.1f}    # low-T CR".format(FINAL_CFG['M_H3']))
print("M_S1 = {:.1f}    # pearlite zone dT".format(FINAL_CFG['M_S1']))
print("M_S2 = {:.1f}    # overall dT".format(FINAL_CFG['M_S2']))
print("M_S3 = {:.1f}    # post-transformation CR".format(FINAL_CFG['M_S3']))
print("M_S4 = {:.1f}    # cementite fraction".format(FINAL_CFG['M_S4']))
print("P_CEMENTITE = {:.0f}  # Cementite phase flat penalty".format(FINAL_CFG['P_CEMENTITE']))
print("M_S7 = {:.3f}   # TS deviation".format(FINAL_CFG['M_S7']))
print("M_S8 = {:.1f}    # EXT (Z) below min".format(FINAL_CFG['M_S8']))
print("M_S9 = {:.1f}    # stage1 CR".format(FINAL_CFG['M_S9']))
print("M_S10 = {:.3f}   # total time".format(FINAL_CFG['M_S10']))
print("W_BONUS = 0.3")
