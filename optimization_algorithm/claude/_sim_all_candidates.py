"""Run sim_T on all 15 candidates and evaluate temperature/cooling vs expert knowledge.

Expert criteria from plan.txt:
  Stage 1 (T > A1):         20-30 C/s ideal, >=10 C/s minimum
  Stage 2 (A1 ~ 550C):      9-12 C/s ideal, 8-15 acceptable
  Stage 3 (T < 550C):       1-3 C/s ideal, <5 C/s required
  Max dT overall:           <= 20 C
  Max dT pearlite zone:     <= 15 C
"""
import sys, os

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import sim_T.sim_T as sim

df = pd.read_excel('工艺数据全.xlsx')

CANDIDATES = {
    'BEST':   [3, 4, 22, 23, 189],
    'MEDIUM': [103, 28, 106, 112, 19],
    'WORST':  [159, 181, 173, 165, 123],
}

def _safe_float(value):
    if pd.isna(value): return None
    try: return float(value)
    except (TypeError, ValueError): return None

def _normalize_percent(value):
    if value is None: return None
    if value > 1.0: return value / 100.0
    return value

def apply_and_sim(row):
    """Run sim_T on one record, return (state, A1)."""
    ort = _safe_float(row.get("ORT"))
    if ort is None: ort = 850

    # Set chemistry
    for elem, attr in [("C_ELE","ELM_C"),("SI_ELE","ELM_SI"),("MN_ELE","ELM_MN"),
                        ("NI_ELE","ELM_NI"),("CR_ELE","ELM_CR")]:
        val = _safe_float(row.get(elem))
        if val is not None:
            setattr(sim.basic_info, attr, val / 100.0)

    sim.basic_info.A1 = (727 - 10.7*sim.basic_info.ELM_MN - 16.9*sim.basic_info.ELM_NI
                          + 16*sim.basic_info.ELM_CR + 29.1*sim.basic_info.ELM_SI)

    rolls, _ = sim.data_loader.load_roll_data()

    # Speeds
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

    # Fans
    for i in range(1, 11):
        val = _safe_float(row.get('FAN{}'.format(i)))
        val = _normalize_percent(val)
        if val is not None:
            rolls[i].fan_status = val
            rolls[i].fan_speed = rolls[i].fan_air_volume * val / rolls[i].fan_area

    state, _ = sim.run_full_simulation(rolls, tem1=ort, tem0=ort, dt=0.01)
    return state, sim.basic_info.A1


def analyze_cooling(state, A1, label):
    """Analyze 3-stage cooling rates and dT from simulation state."""
    time = np.array(state.history_time)
    T0 = np.array(state.history_T_0[-1])
    T1 = np.array(state.history_T_1[-1])
    Bs = 550.0

    def avg_cooling_rate(mask):
        if mask.sum() < 2: return None, None, None
        idx = np.where(mask)[0]
        dt = np.diff(time[mask])
        dT = np.diff(T0[mask])
        cr = np.abs(dT / np.where(dt > 0, dt, np.inf))
        return np.mean(cr), time[idx[0]], time[idx[-1]]

    # Stage 1: T > A1
    mask1 = T0 >= A1
    cr1, t1_s, t1_e = avg_cooling_rate(mask1)
    T1_s = T0[mask1][0] if mask1.sum() > 0 else None
    T1_e = T0[mask1][-1] if mask1.sum() > 0 else None
    dur1 = t1_e - t1_s if (t1_s is not None and t1_e is not None) else None

    # Stage 2: A1 ~ Bs
    mask2 = (T0 <= A1) & (T0 >= Bs)
    cr2, t2_s, t2_e = avg_cooling_rate(mask2)
    T2_s = T0[mask2][0] if mask2.sum() > 0 else None
    T2_e = T0[mask2][-1] if mask2.sum() > 0 else None
    dur2 = t2_e - t2_s if (t2_s is not None and t2_e is not None) else None

    # Stage 3: T < Bs
    mask3 = T0 < Bs
    cr3, t3_s, t3_e = avg_cooling_rate(mask3)
    T3_s = T0[mask3][0] if mask3.sum() > 0 else None
    T3_e = T0[mask3][-1] if mask3.sum() > 0 else None
    dur3 = t3_e - t3_s if (t3_s is not None and t3_e is not None) else None

    # dT
    dT_arr = np.abs(T0 - T1)
    max_dT = np.max(dT_arr)
    max_dT_pearl = np.max(dT_arr[mask2]) if mask2.sum() > 0 else np.nan

    # Judge each stage
    def judge_stage1(cr):
        if cr is None: return 'N/A'
        if cr >= 20: return 'IDEAL'
        if cr >= 10: return 'OK'
        return 'SLOW'

    def judge_stage2(cr):
        if cr is None: return 'N/A'
        if 9 <= cr <= 12: return 'IDEAL'
        if 8 <= cr <= 15: return 'OK'
        if cr < 8: return 'SLOW'
        return 'FAST'

    def judge_stage3(cr):
        if cr is None: return 'N/A'
        if cr <= 3: return 'IDEAL'
        if cr <= 5: return 'OK'
        return 'FAST'

    def judge_dT(v, limit):
        if np.isnan(v): return 'N/A'
        return 'OK' if v <= limit else 'HIGH'

    return {
        'label': label,
        'A1': A1,
        'time_total': time[-1],
        'T_start': T0[0], 'T_end': T0[-1],
        # Stage 1
        'S1_cr': cr1, 'S1_dur': dur1, 'S1_T_range': (T1_s, T1_e),
        'S1_judge': judge_stage1(cr1),
        # Stage 2
        'S2_cr': cr2, 'S2_dur': dur2, 'S2_T_range': (T2_s, T2_e),
        'S2_judge': judge_stage2(cr2),
        # Stage 3
        'S3_cr': cr3, 'S3_dur': dur3, 'S3_T_range': (T3_s, T3_e),
        'S3_judge': judge_stage3(cr3),
        # dT
        'max_dT': max_dT, 'dT_judge': judge_dT(max_dT, 20),
        'max_dT_pearl': max_dT_pearl, 'dTp_judge': judge_dT(max_dT_pearl, 15),
    }

# Run all 15 simulations
all_results = {}
for category, indices in CANDIDATES.items():
    for idx in indices:
        row = df.iloc[idx]
        label = "{}/idx={}".format(category, idx)
        print("Simulating {} (ORT={:.0f}, MAT_NO={})...".format(label, row['ORT'], row.get('MAT_NO','?')), end=' ', flush=True)
        try:
            state, A1 = apply_and_sim(row)
            r = analyze_cooling(state, A1, label)
            r['TS'] = row['TS']
            r['EXT'] = row['EXT']
            r['ORT'] = row['ORT']
            r['V_COOL_data'] = row['V_COOL']
            r['S0'] = row['spacing']
            r['phase'] = row['phase']
            r['fraction'] = row['fraction']
            all_results[label] = r
            print("done (S1_cr={:.1f}, S2_cr={:.1f}, S3_cr={:.1f})".format(
                r['S1_cr'] or 0, r['S2_cr'] or 0, r['S3_cr'] or 0))
        except Exception as e:
            print("FAILED: {}".format(e))
            all_results[label] = {'label': label, 'error': str(e)}

# Print comprehensive comparison table
print()
print("=" * 115)
print("SIM_T VERIFICATION: 3-STAGE COOLING vs EXPERT KNOWLEDGE (plan 3.5.2)")
print("=" * 115)

# Header
hdr = "{:<20} {:>6} {:>6} {:>7} {:>6} {:>7} {:>6} {:>7} {:>6} {:>6} {:>6} {:>6}".format(
    'Candidate', 'TS', 'EXT', 'S1_cr', 'S1', 'S2_cr', 'S2', 'S3_cr', 'S3', 'dTmax', 'dTp', 'VCOOL')
print(hdr)
print("-" * 115)

# Expert targets
print("{:<20} {:>6} {:>6} {:>7} {:>6} {:>7} {:>6} {:>7} {:>6} {:>6} {:>6} {:>6}".format(
    'Expert target', '980-1120', '>=38', '20-30', 'IDEAL', '9-12', 'IDEAL', '1-3', 'IDEAL', '<=20', '<=15', '9-12'))
print("-" * 115)

for category in ['BEST', 'MEDIUM', 'WORST']:
    for idx in CANDIDATES[category]:
        label = "{}/idx={}".format(category, idx)
        r = all_results.get(label)
        if r is None or 'error' in r:
            print("{:<20} ERROR".format(label))
            continue
        print("{:<20} {:>6.0f} {:>6.1f} {:>7.1f} {:>6} {:>7.1f} {:>6} {:>7.1f} {:>6} {:>6.1f} {:>6.1f} {:>6.1f}".format(
            label, r['TS'], r['EXT'],
            r['S1_cr'] or 0, r['S1_judge'],
            r['S2_cr'] or 0, r['S2_judge'],
            r['S3_cr'] or 0, r['S3_judge'],
            r['max_dT'], r['max_dT_pearl'] if not np.isnan(r['max_dT_pearl']) else 0,
            r['V_COOL_data']))
    if category != 'WORST':
        print("-" * 115)

# Summary verdict per candidate
print()
print("=" * 115)
print("OVERALL VERDICT per candidate")
print("=" * 115)
for category in ['BEST', 'MEDIUM', 'WORST']:
    print("\n--- {} ---".format(category))
    for idx in CANDIDATES[category]:
        label = "{}/idx={}".format(category, idx)
        r = all_results.get(label)
        if r is None or 'error' in r:
            print("  {}: SIM FAILED".format(label))
            continue

        # Count IDEAL/OK/FAIL per stage
        stages = []
        for s, judge_key in [('S1', 'S1_judge'), ('S2', 'S2_judge'), ('S3', 'S3_judge')]:
            j = r[judge_key]
            cr = r['{}_cr'.format(s)]
            if cr is None:
                stages.append("{}:N/A".format(s))
            else:
                stages.append("{}:{:.1f}({})".format(s, cr, j))

        dT_status = "dT:{:.1f}({})".format(r['max_dT'], r['dT_judge'])
        dTp_status = "dTp:{:.1f}({})".format(r['max_dT_pearl'], r['dTp_judge'])

        # Count IDEAL stages
        ideal_count = sum(1 for s in ['S1_judge', 'S2_judge', 'S3_judge'] if r[s] == 'IDEAL')
        ok_count = sum(1 for s in ['S1_judge', 'S2_judge', 'S3_judge'] if r[s] in ('IDEAL', 'OK'))

        print("  idx={}: TS={:.0f} EXT={:.1f} | {} | {} {} | IDEAL stages={}/3, OK+IDEAL={}/3".format(
            idx, r['TS'], r['EXT'],
            ", ".join(stages), dT_status, dTp_status,
            ideal_count, ok_count))
