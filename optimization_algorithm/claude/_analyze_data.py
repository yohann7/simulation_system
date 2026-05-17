"""Analyze production data against plan.txt expert knowledge to find best/medium/worst."""
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

df = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '工艺数据全.xlsx'))

def score_record(row):
    penalty = 0.0
    bonus = 0.0
    violations = []
    goods = []

    ts = row['TS']
    ext = row['EXT']
    vcool = row['V_COOL']
    spacing = row['spacing']
    ort = row['ORT']
    phase = row['phase']
    fraction = row['fraction']

    # H4: pearl_frac >= 85%
    pearl_frac = 1.0 - fraction / 100.0
    if pearl_frac < 0.85:
        penalty += 1000
        violations.append("H4:pearl_frac<85%")

    # TS [980, 1120] (plan 3.4.1)
    if 980 <= ts <= 1120:
        bonus += 5.0
        goods.append("TS={:.0f} in [980,1120]".format(ts))
    elif ts < 980:
        penalty += (980 - ts) ** 2 * 0.005
        violations.append("TS={:.0f}<980".format(ts))
    else:
        penalty += (ts - 1120) ** 2 * 0.003
        violations.append("TS={:.0f}>1120".format(ts))

    # EXT >= 38% (plan 3.4.1)
    if ext >= 38:
        goods.append("EXT={:.1f}%>=38%".format(ext))
        if ext >= 42:
            bonus += 2.0
    else:
        penalty += (38 - ext) ** 2 * 3.0
        violations.append("EXT={:.1f}%<38%".format(ext))

    # V_COOL [9, 12] (plan 3.5.1, 3.5.2)
    if 9 <= vcool <= 12:
        bonus += 5.0
        goods.append("V_COOL={:.1f} in [9,12]".format(vcool))
    elif vcool < 9:
        penalty += (9 - vcool) ** 2 * 3.0
        violations.append("V_COOL={:.1f}<9".format(vcool))
    else:
        penalty += (vcool - 12) ** 2 * 2.0
        violations.append("V_COOL={:.1f}>12".format(vcool))

    # S0 [0.12, 0.17] (plan 3.5.3)
    if 0.12 <= spacing <= 0.17:
        bonus += 3.0
        goods.append("S0={:.3f} in [0.12,0.17]".format(spacing))
    elif spacing < 0.12:
        penalty += (0.12 - spacing) ** 2 * 500
    else:
        penalty += (spacing - 0.17) ** 2 * 300

    # ORT [890, 940] (plan 3.5.1)
    if 890 <= ort <= 940:
        goods.append("ORT={:.0f} in [890,940]".format(ort))
        if 910 <= ort <= 930:
            bonus += 2.0
    elif ort < 890:
        penalty += (890 - ort) ** 2 * 0.05
        violations.append("ORT={:.0f}<890".format(ort))
    else:
        penalty += (ort - 940) ** 2 * 0.05

    # Phase: Ferrite better than Cementite (plan 3.7.3)
    if phase == 'Cementite':
        penalty += 3.0
        if fraction > 1.0:
            penalty += fraction * 2.0
        violations.append("Cementite(frac={:.2f}%)".format(fraction))
    else:
        goods.append("Ferrite")

    # H7: Speed ratio <= 1.5 (plan 3.8.1)
    speeds = []
    for i in range(1, 11):
        s = row.get('SPEED{}'.format(i), np.nan)
        if pd.notna(s) and s > 0:
            speeds.append(s)

    speed_violations = 0
    for i in range(1, len(speeds)):
        r = max(speeds[i]/speeds[i-1], speeds[i-1]/speeds[i])
        if r > 1.5:
            speed_violations += 1
            penalty += (r - 1.5) ** 2 * 10
    if speed_violations > 0:
        violations.append("H7:{} speed ratio violations".format(speed_violations))

    # S5: Speed jumps <= 25% (plan 3.8.1)
    for i in range(1, len(speeds)):
        ratio = abs(speeds[i] / speeds[i-1] - 1)
        if ratio > 0.25:
            penalty += min((ratio - 0.25) ** 2 * 5.0, 30.0)

    # Fan strategy (plan 3.5.2): front fans should be high
    fan1 = row.get('FAN1', 0)
    fan2 = row.get('FAN2', 0)
    if pd.notna(fan1) and pd.notna(fan2):
        if fan1 >= 50 and fan2 >= 90:
            goods.append("FAN1={:.0f}%,FAN2={:.0f}% high".format(fan1, fan2))
        elif fan1 < 30:
            penalty += 2.0
            violations.append("FAN1={:.0f}% too low".format(fan1))

    ek_cost = penalty - 0.3 * bonus
    return ek_cost, penalty, bonus, violations, goods, pearl_frac

results = []
for idx, row in df.iterrows():
    cost, penalty, bonus, violations, goods, pearl_frac = score_record(row)
    results.append({
        'idx': idx, 'MAT_NO': row.get('MAT_NO', ''),
        'score': cost, 'penalty': penalty, 'bonus': bonus,
        'TS': row['TS'], 'EXT': row['EXT'], 'V_COOL': row['V_COOL'],
        'spacing': row['spacing'], 'ORT': row['ORT'],
        'phase': row['phase'], 'fraction': row['fraction'],
        'pearl_frac': pearl_frac,
        'violations': violations, 'goods': goods,
    })

results.sort(key=lambda x: x['score'])

print("Total records: {}".format(len(results)))

# BEST
print("\n" + "=" * 80)
print("BEST (Lowest Score)")
print("=" * 80)
best = results[0]
for k in ['idx', 'MAT_NO', 'score', 'penalty', 'bonus', 'TS', 'EXT', 'V_COOL', 'spacing', 'ORT', 'phase', 'fraction', 'pearl_frac']:
    print("  {}: {}".format(k, best[k]))
print("  Violations: {}".format(best['violations']))
print("  Goods: {}".format(best['goods']))

# MEDIUM
valid = [r for r in results if r['penalty'] < 100]
medium = valid[len(valid)//2]
print("\n" + "=" * 80)
print("MEDIUM (Median Score)")
print("=" * 80)
for k in ['idx', 'MAT_NO', 'score', 'penalty', 'bonus', 'TS', 'EXT', 'V_COOL', 'spacing', 'ORT', 'phase', 'fraction', 'pearl_frac']:
    print("  {}: {}".format(k, medium[k]))
print("  Violations: {}".format(medium['violations']))
print("  Goods: {}".format(medium['goods']))

# WORST
worst = valid[-1]
print("\n" + "=" * 80)
print("WORST (Highest Score, still feasible)")
print("=" * 80)
for k in ['idx', 'MAT_NO', 'score', 'penalty', 'bonus', 'TS', 'EXT', 'V_COOL', 'spacing', 'ORT', 'phase', 'fraction', 'pearl_frac']:
    print("  {}: {}".format(k, worst[k]))
print("  Violations: {}".format(worst['violations']))
print("  Goods: {}".format(worst['goods']))

# TOP 10
print("\n" + "=" * 80)
print("TOP 10 RANKING")
print("=" * 80)
print("{:>4} {:>4} {:>8} {:>6} {:>6} {:>7} {:>8} {:>5} {:>12} {:>6}".format(
    'Rank', 'idx', 'Score', 'TS', 'EXT', 'V_COOL', 'S0', 'ORT', 'Phase', 'frac%'))
for rank, r in enumerate(results[:10], 1):
    print("{:>4} {:>4} {:>8.2f} {:>6.0f} {:>6.1f} {:>7.1f} {:>8.3f} {:>5.0f} {:>12} {:>6.2f}".format(
        rank, r['idx'], r['score'], r['TS'], r['EXT'],
        r['V_COOL'], r['spacing'], r['ORT'], r['phase'], r['fraction']))

print("\nSelected: BEST idx={}, MEDIUM idx={}, WORST idx={}".format(best['idx'], medium['idx'], worst['idx']))
