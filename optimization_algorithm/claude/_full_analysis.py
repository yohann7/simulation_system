"""Step 1: Filter all records through hard constraints, then run sim_T on survivors.

Hard filters (remove clearly invalid / severe-error data only):
  - pearl_frac < 0.85 (sorbite standard not met)
  - Any SPEED1-10 = 0 or NaN (broken sensor / impossible operation)
  - Adjacent speed ratio > 1.5 (H7 mechanical constraint)
  - Missing TS, EXT, ORT, V_COOL, spacing

Step 2: Run sim_T on all survivors.
Step 3: Evaluate all dimensions vs expert knowledge.
Step 4: Rank and pick 5 best / 5 medium / 5 worst.
"""
import sys, os

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import sim_T.sim_T as sim

df = pd.read_excel('工艺数据全.xlsx')
print("Total records: {}".format(len(df)))

# ============================================================
# STEP 1: Hard filter
# ============================================================
valid_mask = np.ones(len(df), dtype=bool)
filter_reasons = []

for idx, row in df.iterrows():
    reasons = []

    # Pearlite fraction >= 85%
    frac = row['fraction']
    pearl_frac = 1.0 - frac / 100.0
    if pearl_frac < 0.85:
        reasons.append("pearl_frac={:.3f}<0.85".format(pearl_frac))

    # All SPEED1-10 must be > 0
    speeds = []
    speed_ok = True
    for i in range(1, 11):
        s = row.get('SPEED{}'.format(i), np.nan)
        if pd.isna(s) or s <= 0.01:
            speed_ok = False
            reasons.append("SPEED{}={}".format(i, s))
            break
        speeds.append(s)

    if speed_ok:
        # H7: speed ratio <= 1.5
        for i in range(1, len(speeds)):
            r = max(speeds[i]/speeds[i-1], speeds[i-1]/speeds[i])
            if r > 1.5:
                reasons.append("speed_ratio[{},{}]={:.2f}>1.5".format(i, i+1, r))
                break

    # Missing key data
    for col in ['TS', 'EXT', 'ORT', 'V_COOL', 'spacing']:
        if pd.isna(row.get(col)):
            reasons.append("{} is NaN".format(col))

    if reasons:
        valid_mask[idx] = False
        filter_reasons.append((idx, reasons))

df_valid = df[valid_mask].copy()
n_filtered = len(df) - len(df_valid)
print("Filtered out: {} records".format(n_filtered))
print("Survivors: {} records".format(len(df_valid)))

# Show filter reasons for removed records
if n_filtered <= 30:
    for idx, reasons in filter_reasons:
        print("  idx={}: {}".format(idx, "; ".join(reasons)))
else:
    # Summary of filter reasons
    from collections import Counter
    reason_counts = Counter()
    for idx, reasons in filter_reasons:
        for r in reasons:
            cat = r.split(':')[0].split('[')[0].split('=')[0].strip()
            reason_counts[cat] += 1
    print("Filter reason summary: {}".format(dict(reason_counts)))

if len(df_valid) == 0:
    print("No survivors! Relaxing filters...")
    sys.exit(1)

# ============================================================
# STEP 2: Run sim_T on all survivors
# ============================================================
def _safe_float(value):
    if pd.isna(value): return None
    try: return float(value)
    except: return None

def _normalize_percent(value):
    if value is None: return None
    if value > 1.0: return value / 100.0
    return value

def sim_one_record(idx_row):
    """Run sim_T on a single record, return analysis dict."""
    idx, row = idx_row
    ort = _safe_float(row.get("ORT"))
    if ort is None: ort = 850

    # Chemistry
    for elem, attr in [("C_ELE","ELM_C"),("SI_ELE","ELM_SI"),("MN_ELE","ELM_MN"),
                        ("NI_ELE","ELM_NI"),("CR_ELE","ELM_CR")]:
        val = _safe_float(row.get(elem))
        if val is not None:
            setattr(sim.basic_info, attr, val / 100.0)

    A1 = (727 - 10.7*sim.basic_info.ELM_MN - 16.9*sim.basic_info.ELM_NI
          + 16*sim.basic_info.ELM_CR + 29.1*sim.basic_info.ELM_SI)

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

    time = np.array(state.history_time)
    T0 = np.array(state.history_T_0[-1])
    T1 = np.array(state.history_T_1[-1])
    Bs = 550.0

    def avg_cr(mask):
        if mask.sum() < 2: return None
        dt_arr = np.diff(time[mask])
        dT_arr = np.diff(T0[mask])
        cr = np.abs(dT_arr / np.where(dt_arr > 0, dt_arr, np.inf))
        return float(np.mean(cr))

    mask1 = T0 >= A1
    mask2 = (T0 <= A1) & (T0 >= Bs)
    mask3 = T0 < Bs

    cr1 = avg_cr(mask1)
    cr2 = avg_cr(mask2)
    cr3 = avg_cr(mask3)

    dT_arr = np.abs(T0 - T1)
    max_dT = float(np.max(dT_arr))
    max_dT_pearl = float(np.max(dT_arr[mask2])) if mask2.sum() > 0 else np.nan

    return {
        'idx': idx,
        'cr1': cr1, 'cr2': cr2, 'cr3': cr3,
        'max_dT': max_dT, 'max_dT_pearl': max_dT_pearl,
        'A1': A1, 'T_start': float(T0[0]), 'T_end': float(T0[-1]),
        'time_total': float(time[-1]),
    }

print("\nRunning sim_T on {} records...".format(len(df_valid)))
sim_results = {}
for i, (idx, row) in enumerate(df_valid.iterrows()):
    if (i+1) % 20 == 0 or i == 0:
        print("  {}/{}...".format(i+1, len(df_valid)), end=' ', flush=True)
    try:
        r = sim_one_record((idx, row))
        sim_results[idx] = r
    except Exception as e:
        print("FAIL idx={}: {}".format(idx, e))
        sim_results[idx] = {'idx': idx, 'error': str(e)}
print("done.")

# ============================================================
# STEP 3: Comprehensive evaluation
# ============================================================
def evaluate(r, row):
    """Evaluate one record against all expert criteria. Lower = better match."""
    scores = {}
    details = {}

    ts = row['TS']
    ext = row['EXT']
    ort = row['ORT']
    vcool = row['V_COOL']
    spacing = row['spacing']
    phase = row['phase']
    cr2 = r.get('cr2')
    cr1 = r.get('cr1')
    cr3 = r.get('cr3')
    max_dT = r.get('max_dT', np.nan)
    max_dT_pearl = r.get('max_dT_pearl', np.nan)
    pearl_frac = 1.0 - row['fraction'] / 100.0

    # TS: distance from [980, 1120]
    if 980 <= ts <= 1120:
        scores['TS'] = 0.0
    elif ts < 980:
        scores['TS'] = (980 - ts) * 0.1
    else:
        scores['TS'] = (ts - 1120) * 0.1
    details['TS'] = ts

    # EXT: >= 38
    if ext >= 38:
        scores['EXT'] = 0.0
    else:
        scores['EXT'] = (38 - ext) * 1.0
    details['EXT'] = ext

    # ORT: ideal 910-930, acceptable 890-940
    if 910 <= ort <= 930:
        scores['ORT'] = 0.0
    elif 890 <= ort <= 940:
        scores['ORT'] = 0.5
    else:
        scores['ORT'] = 1.0 + min(abs(ort-890), abs(ort-940)) * 0.02
    details['ORT'] = ort

    # Phase
    if phase == 'Ferrite':
        scores['Phase'] = 0.0
    else:
        scores['Phase'] = 3.0
    details['Phase'] = phase

    # S0
    if 0.12 <= spacing <= 0.17:
        scores['S0'] = 0.0
    else:
        scores['S0'] = distance_to_interval(spacing, 0.12, 0.17) * 30
    details['S0'] = spacing

    # Stage 2 cooling rate (sim_T): ideal 9-12
    if cr2 is not None:
        if 9 <= cr2 <= 12:
            scores['S2_cr'] = 0.0
        elif 8 <= cr2 <= 15:
            scores['S2_cr'] = 0.5
        else:
            scores['S2_cr'] = 1.0 + min(abs(cr2-9), abs(cr2-12)) * 0.3
        details['S2_cr'] = cr2
    else:
        scores['S2_cr'] = 5.0
        details['S2_cr'] = None

    # Stage 1 (sim_T)
    if cr1 is not None:
        if cr1 >= 20:
            scores['S1_cr'] = 0.0
        elif cr1 >= 10:
            scores['S1_cr'] = 0.5
        else:
            scores['S1_cr'] = 1.0
        details['S1_cr'] = cr1
    else:
        scores['S1_cr'] = 3.0
        details['S1_cr'] = None

    # Stage 3 (sim_T)
    if cr3 is not None:
        if cr3 <= 3:
            scores['S3_cr'] = 0.0
        elif cr3 <= 5:
            scores['S3_cr'] = 0.3
        else:
            scores['S3_cr'] = 1.0
        details['S3_cr'] = cr3
    else:
        scores['S3_cr'] = 3.0
        details['S3_cr'] = None

    # dT
    if not np.isnan(max_dT):
        if max_dT <= 20:
            scores['dT'] = 0.0
        else:
            scores['dT'] = (max_dT - 20) * 0.2
        details['max_dT'] = max_dT
    else:
        scores['dT'] = 2.0

    # dT pearlite
    if not np.isnan(max_dT_pearl):
        if max_dT_pearl <= 15:
            scores['dTp'] = 0.0
        else:
            scores['dTp'] = (max_dT_pearl - 15) * 0.3
        details['max_dT_pearl'] = max_dT_pearl
    else:
        scores['dTp'] = 1.0

    total = sum(scores.values())
    details['total'] = total
    details['scores'] = scores
    return total, details

def distance_to_interval(val, lo, hi):
    if lo <= val <= hi: return 0.0
    if val < lo: return lo - val
    return val - hi

# Evaluate all
eval_results = []
for idx, row in df_valid.iterrows():
    r = sim_results.get(idx)
    if r is None or 'error' in r:
        continue
    total, details = evaluate(r, row)
    details['idx'] = idx
    details['MAT_NO'] = row.get('MAT_NO', '?')
    eval_results.append((total, details))

eval_results.sort(key=lambda x: x[0])
n_eval = len(eval_results)
print("\nEvaluated {} records".format(n_eval))
print("Score range: {:.2f} ~ {:.2f}".format(eval_results[0][0], eval_results[-1][0]))

# ============================================================
# STEP 4: Pick 5 best / 5 medium / 5 worst
# ============================================================
best5 = eval_results[:5]
med_start = n_eval // 2 - 2
med5 = eval_results[med_start:med_start+5]
worst5 = eval_results[-5:]

def print_group(title, group):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    hdr = "{:<3} {:>4} {:>8} {:>6} {:>5} {:>7} {:>7} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5} {:>6} {:>10}".format(
        'No', 'idx', 'Score', 'TS', 'EXT', 'S2_cr', 'S1_cr', 'S3_cr', 'dTmax', 'dTp',
        'ORT', 'S0', 'Ph', 'VCOOL', 'MAT_NO')
    print("-" * 110)
    print(hdr)
    print("-" * 110)
    for i, (total, d) in enumerate(group):
        s = d['scores']
        print("{:<3} {:>4} {:>8.2f} {:>6.0f} {:>5.1f} {:>7.1f} {:>7.1f} {:>6.1f} {:>6.1f} {:>6.1f} {:>5.0f} {:>5.3f} {:>5} {:>6.1f} {:>10}".format(
            i+1, d['idx'], d['total'],
            d['TS'], d['EXT'],
            d['S2_cr'] if d['S2_cr'] else 0,
            d['S1_cr'] if d['S1_cr'] else 0,
            d['S3_cr'] if d['S3_cr'] else 0,
            d['max_dT'], d['max_dT_pearl'] if not np.isnan(d['max_dT_pearl']) else 0,
            d['ORT'], d['S0'], 'F' if d['Phase']=='Ferrite' else 'C',
            df.iloc[d['idx']]['V_COOL'],
            str(d['MAT_NO'])[:10]))

    # Score breakdown
    print("\n  Score breakdown (per-dimension sub-scores):")
    for i, (total, d) in enumerate(group):
        s = d['scores']
        parts = ["{}={:.2f}".format(k, v) for k, v in sorted(s.items())]
        print("  #{} idx={}: {}".format(i+1, d['idx'], ", ".join(parts)))

    # Expert verdict
    print("\n  Expert verdict per record:")
    for i, (total, d) in enumerate(group):
        comments = []
        if d['TS'] is not None:
            if 980 <= d['TS'] <= 1120: comments.append("TS in target")
            else: comments.append("TS={:.0f} out of [980,1120]".format(d['TS']))
        if d['EXT'] is not None:
            if d['EXT'] >= 38: comments.append("EXT OK")
            else: comments.append("EXT={:.1f}<38".format(d['EXT']))
        if d['S2_cr'] is not None:
            if 9 <= d['S2_cr'] <= 12: comments.append("S2 IDEAL")
            elif d['S2_cr'] < 8: comments.append("S2 SLOW({:.1f})".format(d['S2_cr']))
            elif d['S2_cr'] > 12: comments.append("S2 FAST({:.1f})".format(d['S2_cr']))
        if d['Phase'] == 'Cementite': comments.append("Cementite!")
        if d['max_dT_pearl'] is not None and not np.isnan(d['max_dT_pearl']):
            if d['max_dT_pearl'] > 15: comments.append("dTp HIGH")
        print("  #{} idx={}: {}".format(i+1, d['idx'], ", ".join(comments)))

print_group("BEST 5 (best overall match to expert knowledge)", best5)
print_group("MEDIUM 5 (median range)", med5)
print_group("WORST 5 (worst overall match to expert knowledge)", worst5)

# Print full process parameters for all 15
print("\n" + "=" * 110)
print("FULL PROCESS PARAMETERS FOR ALL 15 CANDIDATES")
print("=" * 110)
for label, group in [("BEST", best5), ("MEDIUM", med5), ("WORST", worst5)]:
    print("\n--- {} ---".format(label))
    for i, (total, d) in enumerate(group):
        idx = d['idx']
        row = df.iloc[idx]
        print("\n  #{}. idx={} (score={:.2f}, TS={:.0f}, EXT={:.1f}, MAT_NO={})".format(
            i+1, idx, d['total'], d['TS'], d['EXT'], row.get('MAT_NO', '?')))
        print("  ORT:{}".format(int(row['ORT'])))
        spd = " ".join(["SPEED{}:{:.3f}".format(j, row['SPEED{}'.format(j)]) for j in range(1,11)])
        print("  " + spd)
        fan_parts = []
        for j in range(1, 11):
            f = row.get('FAN{}'.format(j))
            if pd.notna(f):
                fan_parts.append("FAN{}:{:.0f}".format(j, f))
        print("  " + " ".join(fan_parts))
