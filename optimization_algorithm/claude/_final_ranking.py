"""Score ALL production data with current constraint system + sim_T, pick 10 diverse per tier."""
import sys, os

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import sim_T.sim_T as sim
from optimization_algorithm import expert_constraints as ec

df = pd.read_excel('工艺数据全.xlsx')
print("Total records: {}".format(len(df)))

# ============================================================
# STEP 1: Hard filter
# ============================================================
valid_mask = np.ones(len(df), dtype=bool)
filtered = []

for idx, row in df.iterrows():
    reasons = []
    # pearl_frac
    pf = 1.0 - row['fraction'] / 100.0
    if pf < 0.85:
        reasons.append("pearl_frac<0.85")
    # SPEED > 0
    speeds = []
    bad = False
    for i in range(1, 11):
        s = row.get('SPEED{}'.format(i), np.nan)
        if pd.isna(s) or s <= 0.01:
            reasons.append("SPEED{}=0/NaN".format(i)); bad = True; break
        speeds.append(s)
    if not bad:
        for i in range(1, len(speeds)):
            r = max(speeds[i]/speeds[i-1], speeds[i-1]/speeds[i])
            if r > 1.5:
                reasons.append("ratio>1.5"); break
    # missing key data
    for c in ['TS', 'EXT', 'ORT', 'V_COOL', 'spacing']:
        if pd.isna(row.get(c)): reasons.append("{}_NaN".format(c))
    if reasons:
        valid_mask[idx] = False
        filtered.append((idx, reasons))

df_valid = df[valid_mask].copy()
print("Filtered: {}, Survivors: {}".format(len(df)-len(df_valid), len(df_valid)))

# ============================================================
# STEP 2: Run sim_T on all survivors
# ============================================================
def _sf(v):
    if pd.isna(v): return None
    try: return float(v)
    except: return None

def _np(v):
    if v is None: return None
    if v > 1.0: return v / 100.0
    return v

def sim_one(idx_row):
    idx, row = idx_row
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
    vals = ec.extract_from_state(state, sim.basic_info)
    vals['TS'] = row['TS']
    vals['EXT'] = row['EXT']
    vals['ORT'] = row['ORT']
    vals['MAT_NO'] = row.get('MAT_NO', '?')
    return vals

print("\nRunning sim_T on {} records...".format(len(df_valid)))
all_vals = {}
for i, (idx, row) in enumerate(df_valid.iterrows()):
    if (i+1) % 25 == 0:
        print("  {}/{}...".format(i+1, len(df_valid)), end=' ', flush=True)
    try:
        all_vals[idx] = sim_one((idx, row))
    except Exception as e:
        print("FAIL idx={}: {}".format(idx, e))
print("done.")

# ============================================================
# STEP 3: Score all records using current constraint system
# ============================================================
scored = []
for idx, vals in all_vals.items():
    feasible, penalty, bonus, details = ec.evaluate_constraints(
        vals, pred_TS=vals['TS'], pred_Z=vals['EXT']
    )
    ek_cost = ec.compute_total_score(feasible, penalty, bonus, mp_cost=0.0)
    scored.append({
        'idx': idx,
        'score': ek_cost if ek_cost != float('inf') else 99999,
        'penalty': penalty,
        'bonus': bonus,
        'feasible': feasible,
        'TS': vals['TS'], 'EXT': vals['EXT'], 'ORT': vals['ORT'],
        'cr_pearl': vals.get('cr_pearl'), 'cr_stage1': vals.get('cr_stage1'),
        'time_total': vals.get('time_total'), 'T_end': vals.get('T_end'),
        'pearl_frac': vals.get('pearl_frac'),
        'max_dT': vals.get('max_dT_total'), 'max_dT_pearl': vals.get('max_dT_pearl'),
        'S0': vals.get('S0_est'), 'T_trans': vals.get('T_trans'),
        'cem_frac': vals.get('cem_frac'),
        'phase': df.iloc[idx]['phase'],
        'MAT_NO': vals['MAT_NO'],
    })

scored.sort(key=lambda x: x['score'])
n = len(scored)
print("\nScored {} records".format(n))
print("Score range: {:.2f} ~ {:.2f}".format(scored[0]['score'], scored[-1]['score']))

# ============================================================
# STEP 4: Pick 10 diverse per tier
# ============================================================

def pick_diverse(candidates, n_pick=10):
    """Pick n_pick diverse records from candidates (sorted by score).
    Diversity ensures spread across ORT levels and fan strategy types.
    """
    if len(candidates) <= n_pick:
        return candidates

    # Categorize each candidate
    for c in candidates:
        row = df.iloc[c['idx']]
        # ORT category
        ort = c['ORT']
        if ort < 870: c['ort_cat'] = 'low'
        elif ort < 890: c['ort_cat'] = 'mid'
        else: c['ort_cat'] = 'high'
        # Fan strategy
        fan1 = row.get('FAN1', 0) if pd.notna(row.get('FAN1')) else 0
        if fan1 >= 90: c['fan_cat'] = 'all_high'
        elif fan1 >= 30: c['fan_cat'] = 'mixed'
        else: c['fan_cat'] = 'fan1_low'
        # Speed level (average of SPEED1-10)
        spd_avg = np.mean([row['SPEED{}'.format(j)] for j in range(1,11)
                          if pd.notna(row.get('SPEED{}'.format(j)))])
        c['spd_avg'] = spd_avg
        if spd_avg < 1.0: c['spd_cat'] = 'slow'
        elif spd_avg < 1.2: c['spd_cat'] = 'medium'
        else: c['spd_cat'] = 'fast'

    # Greedy selection: pick best from each diversity category, then fill remaining
    picks = []
    used_idx = set()

    # Try to get coverage of ort_cat x fan_cat combos first
    combos = {}
    for c in candidates:
        key = (c['ort_cat'], c['fan_cat'])
        if key not in combos:
            combos[key] = []
        combos[key].append(c)

    # Pick best from each combo
    for key, items in combos.items():
        for c in items:
            if c['idx'] not in used_idx:
                picks.append(c)
                used_idx.add(c['idx'])
                break

    # Fill remaining from best not yet picked
    for c in candidates:
        if len(picks) >= n_pick:
            break
        if c['idx'] not in used_idx:
            picks.append(c)
            used_idx.add(c['idx'])

    # Sort by score within picks
    picks.sort(key=lambda x: x['score'])
    return picks[:n_pick]

# BEST: top 25% of scores (take best 10 diverse from this pool)
best_pool = scored[:max(25, n//4)]
best10 = pick_diverse(best_pool, 10)

# MEDIUM: middle of the pack
med_start = n//2 - 12
med_pool = scored[med_start:med_start+25]
med10 = pick_diverse(med_pool, 10)

# WORST: bottom 25%
worst_pool = scored[-max(25, n//4):]
# For worst, take the actual worst but ensure diversity
worst10 = pick_diverse(worst_pool, 10)

print("\nSelected: BEST={}, MEDIUM={}, WORST={}".format(
    [x['idx'] for x in best10],
    [x['idx'] for x in med10],
    [x['idx'] for x in worst10],
))

# ============================================================
# STEP 5: Write best_process_data.txt
# ============================================================
def extract_row_params(idx):
    row = df.iloc[idx]
    params = {}
    params['ORT'] = int(row['ORT']) if pd.notna(row['ORT']) else '?'
    for j in range(1, 11):
        v = row.get('SPEED{}'.format(j))
        params['SPEED{}'.format(j)] = '{:.3f}'.format(v) if pd.notna(v) else 'NaN'
    for j in range(1, 11):
        v = row.get('FAN{}'.format(j))
        if j <= 6:
            params['FAN{}'.format(j)] = '{:.0f}'.format(v) if pd.notna(v) else 'NaN'
        else:
            params['FAN{}'.format(j)] = 'None'  # Not collected
    return params

def write_group(f, title, group):
    f.write(title + '\n')
    f.write('=' * 80 + '\n\n')
    for i, c in enumerate(group):
        row = df.iloc[c['idx']]
        p = extract_row_params(c['idx'])
        f.write("┌─────────────────────────────────────────────────────────────┐\n")
        f.write("│ {} #{}  idx={}  MAT_NO={}  Score={:.2f}\n".format(
            title.split()[0], i+1, c['idx'], c['MAT_NO'], c['score']))
        f.write("├─────────────────────────────────────────────────────────────┤\n")
        f.write("│ TS={:.0f} MPa [980-1120]  EXT={:.1f}% [>=38]  ORT={:.0f} C  Phase={}\n".format(
            c['TS'], c['EXT'], c['ORT'], c['phase']))
        f.write("│ S2_cr={:.1f} C/s  S1_cr={:.1f} C/s  dTmax={:.1f} C  dTp={:.1f} C  T_end={:.0f} C\n".format(
            c['cr_pearl'] or 0, c['cr_stage1'] or 0, c['max_dT'], c['max_dT_pearl'], c['T_end'] or 0))
        f.write("│ pearl_frac={:.3f}  S0={:.3f} um  cem_frac={:.4f}  time={:.0f}s\n".format(
            c['pearl_frac'], c['S0'] or 0, c['cem_frac'], c['time_total'] or 0))
        f.write("│ penalty={:.1f}  bonus={:.0f}  EK_cost={:.2f}\n".format(
            c['penalty'], c['bonus'], c['score']))
        f.write("├─────────────────────────────────────────────────────────────┤\n")
        f.write("│ ORT:{}\n".format(p['ORT']))
        spd_str = '  '.join(['SPEED{}:{}'.format(j, p['SPEED{}'.format(j)]) for j in range(1,6)])
        f.write("│ {}\n".format(spd_str))
        spd_str = '  '.join(['SPEED{}:{}'.format(j, p['SPEED{}'.format(j)]) for j in range(6,11)])
        f.write("│ {}\n".format(spd_str))
        fan_str = '  '.join(['FAN{}:{}'.format(j, p['FAN{}'.format(j)]) for j in range(1,6)])
        f.write("│ {}\n".format(fan_str))
        fan_str = '  '.join(['FAN{}:{}'.format(j, p['FAN{}'.format(j)]) for j in range(6,11)])
        f.write("│ {}\n".format(fan_str))
        f.write("└─────────────────────────────────────────────────────────────┘\n\n")

    # Summary table
    f.write("\nSummary:\n")
    f.write("{:<3} {:>4} {:>8} {:>6} {:>5} {:>6} {:>6} {:>6} {:>5} {:>5} {:>6}\n".format(
        'No', 'idx', 'Score', 'TS', 'EXT', 'S1_cr', 'S2_cr', 'dTmax', 'ORT', 'Ph', 'T_end'))
    for i, c in enumerate(group):
        f.write("{:<3} {:>4} {:>8.1f} {:>6.0f} {:>5.1f} {:>6.1f} {:>6.1f} {:>6.1f} {:>5.0f} {:>5} {:>6.0f}\n".format(
            i+1, c['idx'], c['score'], c['TS'], c['EXT'],
            c['cr_stage1'] or 0, c['cr_pearl'] or 0, c['max_dT'],
            c['ORT'], c['phase'][0] if c['phase'] else '?', c['T_end'] or 0))
    f.write('\n')


out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_process_data.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("生产数据专家知识评价 —— BEST/MEDIUM/WORST 各10条\n")
    f.write("═══════════════════════════════════════════════════════════════\n")
    f.write("分析日期: 2026-05-15\n")
    f.write("数据来源: 工艺数据全.xlsx (219条 → 硬约束过滤 {} 条 → {} 条仿真评分)\n".format(
        len(df)-len(df_valid), len(df_valid)))
    f.write("评分方法: expert_constraints.py 当前全部约束 (H2-H4, S1-S2, S4-S5, S7-S11, B1-B2, B4)\n")
    f.write("多样性: 每等级内按 ORT 区间 × 风机策略 分层选取\n")
    f.write("\n")

    write_group(f, "一、BEST 10 —— 与专家知识最吻合的工艺参数", best10)
    write_group(f, "二、MEDIUM 10 —— 中等吻合度的工艺参数", med10)
    write_group(f, "三、WORST 10 —— 与专家知识偏离最大的工艺参数", worst10)

    # Tier analysis
    f.write("\n═══════════════════════════════════════════════════════════════\n")
    f.write("四、三级对比分析\n")
    f.write("═══════════════════════════════════════════════════════════════\n\n")

    for label, grp in [('BEST', best10), ('MEDIUM', med10), ('WORST', worst10)]:
        ts_vals = [c['TS'] for c in grp]
        ext_vals = [c['EXT'] for c in grp]
        s1_vals = [c['cr_stage1'] for c in grp if c['cr_stage1']]
        s2_vals = [c['cr_pearl'] for c in grp if c['cr_pearl']]
        ort_vals = [c['ORT'] for c in grp]
        scores = [c['score'] for c in grp]
        phases = [c['phase'] for c in grp]
        f.write("{}: Score={:.0f}~{:.0f}, TS={:.0f}~{:.0f}, EXT={:.0f}~{:.0f}, ".format(
            label, min(scores), max(scores),
            min(ts_vals), max(ts_vals), min(ext_vals), max(ext_vals)))
        f.write("S1_cr={:.1f}~{:.1f}, S2_cr={:.1f}~{:.1f}, ORT={:.0f}~{:.0f}\n".format(
            min(s1_vals), max(s1_vals), min(s2_vals), max(s2_vals),
            min(ort_vals), max(ort_vals)))
        f.write("  Phases: Ferrite={}, Cementite={}\n".format(
            phases.count('Ferrite'), phases.count('Cementite')))

print("\nDone. Output: {}".format(out_path))
