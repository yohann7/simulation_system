"""Find 5 best / 5 medium / 5 worst candidates from production data.

Criteria (from plan.txt expert knowledge, NOT from expert_constraints.py):
  - TS: target 1050+-70 [980, 1120], center=1050
  - EXT (Z): >= 38
  - ORT: ideal 910-930, acceptable 890-940
  - V_COOL: ideal 9-12 C/s
  - S0: ideal 0.12-0.17 um
  - Phase: Ferrite > Cementite
  - Speed ratio: <= 1.5, smooth transitions preferred
  - FAN1, FAN2: should be high (close to 100%)
  - pearl_frac: >= 0.85 (fraction is proeutectoid %)
"""
import sys, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

df = pd.read_excel(os.path.join(PROJECT_ROOT, '工艺数据全.xlsx'))

def distance_to_target(val, lo, hi):
    """Distance to interval [lo, hi]. 0 = inside, positive = how far outside."""
    if pd.isna(val):
        return 999
    if lo <= val <= hi:
        return 0.0
    if val < lo:
        return lo - val
    return val - hi

def evaluate_record(row):
    """Return (rank_score, details_dict) for a record. Lower score = better."""
    details = {}

    # --- TS: 980-1120, center 1050 ---
    ts = row['TS']
    ts_dist = distance_to_target(ts, 980, 1120)
    # weight: TS is the most important, heavy penalty for being outside
    if ts_dist == 0:
        ts_score = 0
        details['TS_ok'] = True
    else:
        ts_score = ts_dist * 0.1  # e.g. TS=1200 -> 80*0.1=8
        details['TS_ok'] = False
    details['TS'] = ts
    details['TS_dist'] = ts_dist

    # --- EXT: >= 38 ---
    ext = row['EXT']
    if ext >= 38:
        ext_score = 0
        details['EXT_ok'] = True
    else:
        ext_score = (38 - ext) * 2.0
        details['EXT_ok'] = False
    details['EXT'] = ext

    # --- ORT: ideal 910-930, acceptable 890-940 ---
    ort = row['ORT']
    ort_dist_ideal = distance_to_target(ort, 910, 930)
    ort_dist_acc = distance_to_target(ort, 890, 940)
    ort_score = ort_dist_ideal * 0.05
    details['ORT'] = ort
    details['ORT_in_ideal'] = (ort_dist_ideal == 0)
    details['ORT_in_acc'] = (ort_dist_acc == 0)

    # --- V_COOL: 9-12 ---
    vcool = row['V_COOL']
    vcool_dist = distance_to_target(vcool, 9, 12)
    vcool_score = vcool_dist * 0.5
    details['V_COOL'] = vcool
    details['V_COOL_in_ideal'] = (vcool_dist == 0)

    # --- S0: 0.12-0.17 ---
    spacing = row['spacing']
    s0_dist = distance_to_target(spacing, 0.12, 0.17)
    s0_score = s0_dist * 50
    details['S0'] = spacing
    details['S0_in_ideal'] = (s0_dist == 0)

    # --- Phase: Ferrite=0, Cementite=+penalty ---
    phase = row['phase']
    if phase == 'Cementite':
        phase_score = 5.0
        details['phase_ok'] = False
    else:
        phase_score = 0
        details['phase_ok'] = True
    details['phase'] = phase

    # --- pearl_frac (from fraction) ---
    frac = row['fraction']
    pearl_frac = 1.0 - frac / 100.0
    details['pearl_frac'] = pearl_frac
    if pearl_frac < 0.85:
        pearl_score = 50  # heavy penalty
        details['pearl_ok'] = False
    else:
        pearl_score = 0
        details['pearl_ok'] = True

    # --- Speed ratio: check H7 (<= 1.5) and smoothness ---
    speeds = []
    for i in range(1, 11):
        s = row.get('SPEED{}'.format(i), np.nan)
        if pd.notna(s) and s > 0.01:
            speeds.append(s)

    speed_score = 0
    max_ratio = 1.0
    jump_count = 0
    for i in range(1, len(speeds)):
        r = max(speeds[i]/speeds[i-1], speeds[i-1]/speeds[i])
        max_ratio = max(max_ratio, r)
        if r > 1.5:
            speed_score += 20  # H7 violation
        # S5: jump > 25%
        ratio_change = abs(speeds[i]/speeds[i-1] - 1)
        if ratio_change > 0.25:
            jump_count += 1
            speed_score += ratio_change * 2.0
    details['max_speed_ratio'] = max_ratio
    details['speed_jumps'] = jump_count
    details['speed_count'] = len(speeds)

    # --- Fan: FAN1 should be high (>=50 ideal), FAN2 >= 90 ---
    fan_score = 0
    fan1 = row.get('FAN1', np.nan)
    fan2 = row.get('FAN2', np.nan)
    if pd.notna(fan1) and fan1 < 50:
        fan_score += (50 - fan1) * 0.02
    if pd.notna(fan2) and fan2 < 90:
        fan_score += (90 - fan2) * 0.05
    details['FAN1'] = fan1 if pd.notna(fan1) else -1
    details['FAN2'] = fan2 if pd.notna(fan2) else -1

    # --- Total score ---
    total = (ts_score + ext_score + ort_score + vcool_score +
             s0_score + phase_score + pearl_score + speed_score + fan_score)
    details['total_score'] = total
    return total, details

# Evaluate all records
results = []
for idx, row in df.iterrows():
    score, details = evaluate_record(row)
    if details['pearl_ok']:  # only include records with pearl_frac >= 85%
        results.append((idx, score, details))

results.sort(key=lambda x: x[1])

print("Evaluated {} records (pearl_frac >= 0.85)".format(len(results)))
print()

# === BEST 5: lowest scores (best match to expert knowledge) ===
print("=" * 70)
print("BEST 5 candidates (best match to expert recommendations)")
print("=" * 70)
print("{:>3} {:>4} {:>8} {:>6} {:>5} {:>7} {:>7} {:>5} {:>10} {:>6} {:>6} {:>6}".format(
    'No', 'idx', 'Score', 'TS', 'EXT', 'V_COOL', 'S0', 'ORT', 'Phase', 'FAN1', 'FAN2', 'SpdR'))
print("-" * 70)
best5 = []
for i, (idx, score, d) in enumerate(results[:5]):
    best5.append((idx, score, d))
    print("{:>3} {:>4} {:>8.2f} {:>6.0f} {:>5.1f} {:>7.1f} {:>7.3f} {:>5.0f} {:>10} {:>6.0f} {:>6.0f} {:>6.2f}".format(
        i+1, idx, score, d['TS'], d['EXT'], d['V_COOL'], d['S0'], d['ORT'],
        d['phase'], d['FAN1'], d['FAN2'], d['max_speed_ratio']))

# === MEDIUM 5: around median ===
n = len(results)
med_start = n // 2 - 2
print()
print("=" * 70)
print("MEDIUM 5 candidates (median range, typical production)")
print("=" * 70)
print("{:>3} {:>4} {:>8} {:>6} {:>5} {:>7} {:>7} {:>5} {:>10} {:>6} {:>6} {:>6}".format(
    'No', 'idx', 'Score', 'TS', 'EXT', 'V_COOL', 'S0', 'ORT', 'Phase', 'FAN1', 'FAN2', 'SpdR'))
print("-" * 70)
med5 = []
for i, (idx, score, d) in enumerate(results[med_start:med_start+5]):
    med5.append((idx, score, d))
    print("{:>3} {:>4} {:>8.2f} {:>6.0f} {:>5.1f} {:>7.1f} {:>7.3f} {:>5.0f} {:>10} {:>6.0f} {:>6.0f} {:>6.2f}".format(
        i+1, idx, score, d['TS'], d['EXT'], d['V_COOL'], d['S0'], d['ORT'],
        d['phase'], d['FAN1'], d['FAN2'], d['max_speed_ratio']))

# === WORST 5: highest scores (worst match) ===
print()
print("=" * 70)
print("WORST 5 candidates (worst match to expert recommendations)")
print("=" * 70)
print("{:>3} {:>4} {:>8} {:>6} {:>5} {:>7} {:>7} {:>5} {:>10} {:>6} {:>6} {:>6}".format(
    'No', 'idx', 'Score', 'TS', 'EXT', 'V_COOL', 'S0', 'ORT', 'Phase', 'FAN1', 'FAN2', 'SpdR'))
print("-" * 70)
worst5 = []
for i, (idx, score, d) in enumerate(results[-5:]):
    worst5.append((idx, score, d))
    print("{:>3} {:>4} {:>8.2f} {:>6.0f} {:>5.1f} {:>7.1f} {:>7.3f} {:>5.0f} {:>10} {:>6.0f} {:>6.0f} {:>6.2f}".format(
        i+1, idx, score, d['TS'], d['EXT'], d['V_COOL'], d['S0'], d['ORT'],
        d['phase'], d['FAN1'], d['FAN2'], d['max_speed_ratio']))

# === DETAILED breakdown for BEST 5 ===
print()
print("=" * 70)
print("BEST 5 - Detailed criteria breakdown")
print("=" * 70)
for i, (idx, score, d) in enumerate(best5):
    row = df.iloc[idx]
    print("\n--- Best candidate #{} (idx={}, score={:.2f}) ---".format(i+1, idx, score))
    print("  TS={:.0f} [980-1120]: {}".format(d['TS'], 'OK' if d['TS_ok'] else 'OUT'))
    print("  EXT={:.1f}% [>=38]: {}".format(d['EXT'], 'OK' if d['EXT_ok'] else 'LOW'))
    print("  ORT={:.0f} [ideal 910-930]: {}".format(d['ORT'], 'IN' if d['ORT_in_ideal'] else ('in [890,940]' if d['ORT_in_acc'] else 'OUT')))
    print("  V_COOL={:.1f} [9-12]: {}".format(d['V_COOL'], 'IN' if d['V_COOL_in_ideal'] else 'OUT'))
    print("  S0={:.3f} [0.12-0.17]: {}".format(d['S0'], 'IN' if d['S0_in_ideal'] else 'OUT'))
    print("  Phase={} [Ferrite ideal]: {}".format(d['phase'], 'OK' if d['phase_ok'] else 'Cementite'))
    print("  pearl_frac={:.3f} [>=0.85]: {}".format(d['pearl_frac'], 'OK' if d['pearl_ok'] else 'LOW'))
    print("  FAN1={:.0f}%, FAN2={:.0f}% [high=good]".format(d['FAN1'], d['FAN2']))
    print("  Speed ratio max={:.2f} [<=1.5 ideal], jumps={}".format(d['max_speed_ratio'], d['speed_jumps']))
    # print speeds
    spd_str = ", ".join(["SPEED{}={:.3f}".format(j+1, row['SPEED{}'.format(j+1)])
                         for j in range(10) if pd.notna(row.get('SPEED{}'.format(j+1)))])
    print("  Speeds: " + spd_str)
    fan_str = ", ".join(["FAN{}={:.0f}%".format(j+1, row['FAN{}'.format(j+1)])
                         for j in range(6) if pd.notna(row.get('FAN{}'.format(j+1)))])
    print("  Fans: " + fan_str)
    print("  MAT_NO: {}".format(row.get('MAT_NO', '?')))

# === Save candidate indices for sim_T verification ===
print()
print("=" * 70)
print("SUMMARY OF INDICES")
print("=" * 70)
print("Best 5:    {}".format([idx for idx, _, _ in best5]))
print("Medium 5:  {}".format([idx for idx, _, _ in med5]))
print("Worst 5:   {}".format([idx for idx, _, _ in worst5]))
