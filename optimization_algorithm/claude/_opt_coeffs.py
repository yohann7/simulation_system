"""
Meta-optimization v3: PO brute-force search for optimal TIER_CFG parameters.
Evaluates tier weights against cached sim_T results from 30 labeled records.

Search space (4 params): k, W_T1, W_T2, W_T3
"""
import sys, os, time, warnings, json
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'optimization_algorithm'))
os.chdir(PROJECT_ROOT)

import sim_T.sim_T as sim
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'sim_T'))
import calculate_all_sim_T as calc
_orig = calc.run_all_simulations
def _single(*a,**kw): kw['n_workers']=1; return _orig(*a,**kw)
calc.run_all_simulations = _single

import numpy as np, pandas as pd
import expert_constraints as ec
import PO

# ============================================================
# 1. Data caching (30 labeled records)
# ============================================================

df = pd.read_excel('工艺数据全.xlsx')
BEST = [195,40,53,59,33,194,196,35,51,26]
MED = [99,15,98,100,27,101,102,103,8,6]
WORST = [132,159,133,166,164,134,208,135,213,136]

def _sf(v):
    if pd.isna(v): return None
    try: return float(v)
    except: return None

def _np(v):
    if v is None: return None
    if v > 1.0: return v / 100.0
    return v

def sim_one(row):
    """Simulate one data row, return (vals_dict, speeds_list, ort, fans_list)."""
    ort = _sf(row.get("ORT")) or 850
    for e, a in [("C_ELE","ELM_C"),("SI_ELE","ELM_SI"),("MN_ELE","ELM_MN"),
                 ("NI_ELE","ELM_NI"),("CR_ELE","ELM_CR")]:
        v = _sf(row.get(e))
        if v is not None: setattr(sim.basic_info, a, v / 100.0)

    rolls, _ = sim.data_loader.load_roll_data()

    for i in range(1, 11):
        v = _sf(row.get('SPEED%d' % i))
        if v is not None and v > 0:
            rolls[i].roll_v = v
    ev = _sf(row.get("SPEED1"))
    if ev is not None and ev > 0:
        rolls[0].roll_v = ev

    for i in range(1, 11):
        v = _sf(row.get('FAN%d' % i))
        v = _np(v)
        if v is not None:
            rolls[i].fan_status = v
            rolls[i].fan_speed = rolls[i].fan_air_volume * v / rolls[i].fan_area

    state, rt = sim.run_full_simulation(rolls, tem1=ort, tem0=ort, dt=0.01)
    vals = ec.extract_from_state(state, sim.basic_info, rt)
    speeds = [_sf(row.get('SPEED%d' % i)) or 1.0 for i in range(1, 11)]
    fans = [_sf(row.get('FAN%d' % i)) or 0.0 for i in range(1, 11)]
    return vals, speeds, ort, fans

# Cache all 30 records
print("=" * 60)
print("Stage 1: Caching sim_T for 30 labeled records...")
print("=" * 60)
cache = {}
for idx in BEST + MED + WORST:
    print("  idx=%d..." % idx, end=' ', flush=True)
    row = df.iloc[idx]
    cache[idx] = sim_one(row)
    print("done")
print("Caching complete.\n")

def score_all_records():
    """Score all cached records with current TIER_CFG. Returns tier dicts."""
    tiers = {'B': [], 'M': [], 'W': []}
    for idx in BEST:
        v, spd, ort, fan = cache[idx]
        p, _ = ec.evaluate_constraints(v, speeds=spd, ort=ort, fans=fan)
        tiers['B'].append(p)
    for idx in MED:
        v, spd, ort, fan = cache[idx]
        p, _ = ec.evaluate_constraints(v, speeds=spd, ort=ort, fans=fan)
        tiers['M'].append(p)
    for idx in WORST:
        v, spd, ort, fan = cache[idx]
        p, _ = ec.evaluate_constraints(v, speeds=spd, ort=ort, fans=fan)
        tiers['W'].append(p)
    return tiers

t0 = score_all_records()
print("Initial tier scores with current TIER_CFG:")
print("  BEST:  [%s]" % ', '.join('%.1f' % s for s in sorted(t0['B'])))
print("  MED:   [%s]" % ', '.join('%.1f' % s for s in sorted(t0['M'])))
print("  WORST: [%s]" % ', '.join('%.1f' % s for s in sorted(t0['W'])))
b_max, m_min, m_max, w_min, w_max = max(t0['B']), min(t0['M']), max(t0['M']), min(t0['W']), max(t0['W'])
print("  Ranges: BEST max=%.1f, MED [%.1f, %.1f], WORST [%.1f, %.1f]" % (b_max, m_min, m_max, w_min, w_max))
overlaps = []
if b_max >= m_min: overlaps.append("BEST-MED")
if m_max >= w_min: overlaps.append("MED-WORST")
print("  Overlaps: %s" % (overlaps if overlaps else "none"))
print()

# ============================================================
# 2. Coefficient search space (4 params: k, W_T1, W_T2, W_T3)
# ============================================================

COEFF_NAMES = ['k', 'W_T1', 'W_T2', 'W_T3']

COEFF_LB = np.array([1.5, 0.5, 0.3, 0.1], dtype=np.float64)
COEFF_UB = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float64)

COEFF_CURRENT = np.array([
    ec.TIER_CFG['k'], ec.TIER_CFG['W_T1'],
    ec.TIER_CFG['W_T2'], ec.TIER_CFG['W_T3'],
], dtype=np.float64)

print("Coefficient search space:")
for i, name in enumerate(COEFF_NAMES):
    print("  %s: current=%.2f, bounds=[%.2f, %.2f]" % (name, COEFF_CURRENT[i], COEFF_LB[i], COEFF_UB[i]))
print()

# ============================================================
# 3. Coefficient cost function
# ============================================================

def _apply_coeffs(coeffs):
    """Apply coefficient vector to ec.TIER_CFG. Returns dict of old values."""
    old = dict(ec.TIER_CFG)
    ec.TIER_CFG['k'] = float(coeffs[0])
    ec.TIER_CFG['W_T1'] = float(coeffs[1])
    ec.TIER_CFG['W_T2'] = float(coeffs[2])
    ec.TIER_CFG['W_T3'] = float(coeffs[3])
    return old

def _restore_coeffs(old):
    """Restore ec.TIER_CFG from saved values."""
    for k, v in old.items():
        ec.TIER_CFG[k] = v

def evaluate_coefficients(coeff_vector):
    """
    Evaluate a coefficient set against all 30 cached records.
    Returns: (cost, breakdown) — cost lower = better, 0 = perfect separation.
    """
    coeffs = np.asarray(coeff_vector, dtype=np.float64).flatten()
    old = _apply_coeffs(coeffs)

    try:
        tiers = {'B': [], 'M': [], 'W': []}

        for idx in BEST:
            v, spd, ort, fan = cache[idx]
            p, _ = ec.evaluate_constraints(v, speeds=spd, ort=ort, fans=fan)
            tiers['B'].append(p)

        for idx in MED:
            v, spd, ort, fan = cache[idx]
            p, _ = ec.evaluate_constraints(v, speeds=spd, ort=ort, fans=fan)
            tiers['M'].append(p)

        for idx in WORST:
            v, spd, ort, fan = cache[idx]
            p, _ = ec.evaluate_constraints(v, speeds=spd, ort=ort, fans=fan)
            tiers['W'].append(p)

        b_arr = np.array(tiers['B'])
        m_arr = np.array(tiers['M'])
        w_arr = np.array(tiers['W'])

        b_max = float(np.max(b_arr))
        m_min = float(np.min(m_arr))
        m_max = float(np.max(m_arr))
        w_min = float(np.min(w_arr))
        w_max = float(np.max(w_arr))

        cost = 0.0

        # Tier range violations
        if b_max > 9:     cost += (b_max - 9) ** 2
        if m_min < 10:    cost += (10 - m_min) ** 2
        if m_max > 29:    cost += (m_max - 29) ** 2
        if w_min < 30:    cost += (30 - w_min) ** 2
        if w_max > 40:    cost += (w_max - 40) ** 2

        # Overlap violations
        if b_max >= m_min: cost += (b_max - m_min + 1) ** 2 * 10.0
        if m_max >= w_min: cost += (m_max - w_min + 1) ** 2 * 10.0

        # Within-tier spread
        cost += float(np.std(b_arr)) * 0.1
        cost += float(np.std(m_arr)) * 0.1
        cost += float(np.std(w_arr)) * 0.1

        breakdown = {
            'cost': cost, 'b_max': b_max, 'm_min': m_min, 'm_max': m_max,
            'w_min': w_min, 'w_max': w_max,
            'b_overlap': b_max >= m_min, 'm_overlap': m_max >= w_min,
        }
        return cost, breakdown

    finally:
        _restore_coeffs(old)


def coeff_batch_cost(X_batch, batch_tag=""):
    X_arr = np.asarray(X_batch, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    n = X_arr.shape[0]
    costs = np.full(n, float('inf'), dtype=np.float64)

    for i in range(n):
        x = np.clip(X_arr[i], COEFF_LB, COEFF_UB)
        cost, breakdown = evaluate_coefficients(x)
        costs[i] = cost

        if i == 0 or (i < 3 and batch_tag in ("Init", "Iter 1 Explore")):
            print(f"  [{batch_tag}] coeff[{i}] cost={cost:.1f} "
                  f"B_max={breakdown['b_max']:.1f} M=[{breakdown['m_min']:.1f},{breakdown['m_max']:.1f}] "
                  f"W=[{breakdown['w_min']:.1f},{breakdown['w_max']:.1f}] "
                  f"overlap={breakdown['b_overlap'] or breakdown['m_overlap']}")

    return costs


def coeff_single_cost(x):
    cost, _ = evaluate_coefficients(x)
    return cost


# ============================================================
# 4. Run PO to find optimal coefficients
# ============================================================

print("=" * 60)
print("Stage 2: PO meta-optimization of TIER_CFG")
print("=" * 60)

baseline_cost, baseline_bd = evaluate_coefficients(COEFF_CURRENT)
print("Current TIER_CFG cost: %.1f" % baseline_cost)
print("  B_max=%.1f M=[%.1f,%.1f] W=[%.1f,%.1f]" % (
    baseline_bd['b_max'], baseline_bd['m_min'], baseline_bd['m_max'],
    baseline_bd['w_min'], baseline_bd['w_max']))
print()

nSol = 20
MaxIter = 50
dim = len(COEFF_NAMES)
lb, ub = COEFF_LB.copy(), COEFF_UB.copy()

np.random.seed(12345)

_orig_repair_speeds = PO._repair_speeds
PO._repair_speeds = lambda X, lb, ub: np.clip(X, lb, ub)

try:
    best_coeffs, best_cost, conv = PO.puma_optimize(
        nSol=nSol, MaxIter=MaxIter, lb=lb, ub=ub, dim=dim,
        CostFunction=coeff_single_cost,
        BatchCostFunction=coeff_batch_cost,
        patience=15,
    )
finally:
    PO._repair_speeds = _orig_repair_speeds

print("\nBest coefficient cost: %.2f (current: %.2f)" % (best_cost, baseline_cost))
print()

# ============================================================
# 5. Analyze best coefficients
# ============================================================

print("=" * 60)
print("Stage 3: Analyze best coefficients")
print("=" * 60)

old_final = _apply_coeffs(best_coeffs)
final_tiers = score_all_records()

print("Best TIER_CFG found:")
for i, name in enumerate(COEFF_NAMES):
    delta = best_coeffs[i] - COEFF_CURRENT[i]
    pct = delta / COEFF_CURRENT[i] * 100 if COEFF_CURRENT[i] > 0 else 0
    print("  %s = %.4f (was %.4f, %+.0f%%)" % (name, best_coeffs[i], COEFF_CURRENT[i], pct))

print()
print("Tier scores with best TIER_CFG:")
print("  BEST:  [%s]" % ', '.join('%.1f' % s for s in sorted(final_tiers['B'])))
print("  MED:   [%s]" % ', '.join('%.1f' % s for s in sorted(final_tiers['M'])))
print("  WORST: [%s]" % ', '.join('%.1f' % s for s in sorted(final_tiers['W'])))

b_arr = np.array(final_tiers['B'])
m_arr = np.array(final_tiers['M'])
w_arr = np.array(final_tiers['W'])
b_max, m_min, m_max, w_min, w_max = float(np.max(b_arr)), float(np.min(m_arr)), float(np.max(m_arr)), float(np.min(w_arr)), float(np.max(w_arr))
print("  Ranges: BEST [%.1f, %.1f], MED [%.1f, %.1f], WORST [%.1f, %.1f]" % (
    float(np.min(b_arr)), b_max, m_min, m_max, w_min, w_max))
overlaps = []
if b_max >= m_min: overlaps.append("BEST-MED(%.1f>=%.1f)" % (b_max, m_min))
if m_max >= w_min: overlaps.append("MED-WORST(%.1f>=%.1f)" % (m_max, w_min))
print("  Overlaps: %s" % (overlaps if overlaps else "none"))

criteria_met = sum([
    b_max <= 9,
    m_min >= 10 and m_max <= 29,
    w_min >= 30 and w_max <= 40,
    b_max < m_min,
    m_max < w_min,
])
print("  Criteria met: %d/5" % criteria_met)

_restore_coeffs(old_final)

# ============================================================
# 6. Persist to expert_constraints.py
# ============================================================

print()
print("=" * 60)
print("Stage 4: Persist best TIER_CFG")
print("=" * 60)

if criteria_met >= 3:
    import re
    src_path = os.path.join(PROJECT_ROOT, 'optimization_algorithm', 'expert_constraints.py')

    with open(src_path, 'r', encoding='utf-8') as f:
        src = f.read()

    # Update TIER_CFG values
    for i, k in enumerate(COEFF_NAMES):
        pattern = r'("{}"\s*:\s*)[\d.]+'.format(k)
        replacement = r'\g<1>{}'.format(round(best_coeffs[i], 4))
        src = re.sub(pattern, replacement, src)

    with open(src_path, 'w', encoding='utf-8') as f:
        f.write(src)

    # Also apply in memory
    _apply_coeffs(best_coeffs)

    print("Persisted to %s" % src_path)
    print("Updated: k=%.4f, W_T1=%.4f, W_T2=%.4f, W_T3=%.4f" % tuple(best_coeffs))
else:
    print("NOT persisted: only %d/5 criteria met (need >=3)" % criteria_met)

print()
print("=" * 60)
print("Meta-optimization complete.")
print("=" * 60)

conv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_opt_conv.json')
with open(conv_path, 'w') as f:
    json.dump({
        'convergence': [float(c) for c in conv],
        'best_cost': float(best_cost),
        'best_TIER_CFG': {COEFF_NAMES[i]: float(best_coeffs[i]) for i in range(len(COEFF_NAMES))},
        'criteria_met': criteria_met,
    }, f, indent=2)
print("Convergence data saved to %s" % conv_path)
