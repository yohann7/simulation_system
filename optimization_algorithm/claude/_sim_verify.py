"""Run sim_T on selected 3 records and analyze cooling behavior vs expert knowledge."""
import sys, os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import sim_T.sim_T as sim

# Load data
df = pd.read_excel('工艺数据全.xlsx')

# Selected indices from analysis
selected = {
    'BEST': 3,
    'MEDIUM': 185,
    'WORST': 87,
}

DEFAULT_DT = 0.01

def _safe_float(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _normalize_percent(value):
    if value is None:
        return None
    if value > 1.0:
        return value / 100.0
    return value

def _apply_process_row_to_sim(row, rolls):
    c_wt = _safe_float(row.get("C_ELE"))
    si_wt = _safe_float(row.get("SI_ELE"))
    mn_wt = _safe_float(row.get("MN_ELE"))
    ni_wt = _safe_float(row.get("NI_ELE"))
    cr_wt = _safe_float(row.get("CR_ELE"))

    if c_wt is not None:
        sim.basic_info.ELM_C = c_wt / 100.0
    if si_wt is not None:
        sim.basic_info.ELM_SI = si_wt / 100.0
    if mn_wt is not None:
        sim.basic_info.ELM_MN = mn_wt / 100.0
    if ni_wt is not None:
        sim.basic_info.ELM_NI = ni_wt / 100.0
    if cr_wt is not None:
        sim.basic_info.ELM_CR = cr_wt / 100.0

    sim.basic_info.A1 = (
        727
        - 10.7 * sim.basic_info.ELM_MN
        - 16.9 * sim.basic_info.ELM_NI
        + 16 * sim.basic_info.ELM_CR
        + 29.1 * sim.basic_info.ELM_SI
    )

    speed_map = {
        "SPEED1": 1, "SPEED2": 2, "SPEED3": 3, "SPEED4": 4, "SPEED5": 5,
        "SPEED6": 6, "SPEED7": 7, "SPEED8": 8, "SPEED9": 9, "SPEED10": 10,
    }
    for col, idx in speed_map.items():
        val = _safe_float(row.get(col))
        if val is not None and val > 0:
            old_v = rolls[idx].roll_v
            rolls[idx].roll_v = val
            rolls[idx].t = rolls[idx].t * (old_v / val)
            rolls[idx].step = int(rolls[idx].t / sim._default_dt)

    entry_speed = _safe_float(row.get("SPEED1"))
    if entry_speed is not None and entry_speed > 0:
        old_v = rolls[0].roll_v
        rolls[0].roll_v = entry_speed
        rolls[0].t = rolls[0].t * (old_v / entry_speed)
        rolls[0].step = int(rolls[0].t / sim._default_dt)

    fan_map = {f"FAN{i}": i for i in range(1, 11)}
    for col, idx in fan_map.items():
        val = _safe_float(row.get(col))
        val = _normalize_percent(val)
        if val is not None:
            rolls[idx].fan_status = val
            rolls[idx].fan_speed = rolls[idx].fan_air_volume * val / rolls[idx].fan_area

def _format_time_col(t, suffix):
    return f"{float(t):.2f}{suffix}"

def run_single_sim(row, tem1=850, tem0=830):
    rolls, num_rolls = sim.data_loader.load_roll_data()
    _apply_process_row_to_sim(row, rolls)
    state, roll_start_time = sim.run_full_simulation(rolls, tem1=tem1, tem0=tem0, dt=DEFAULT_DT)
    return state, roll_start_time

def simulate_and_analyze(row, label):
    print("\n" + "=" * 80)
    print("{} (idx={}, MAT_NO={})".format(label, row.name, row.get('MAT_NO', '?')))
    print("  ORT={:.0f}, TS={:.0f}, EXT={:.1f}, V_COOL={:.1f}, S0={:.3f}, phase={}, fraction={:.2f}%".format(
        row['ORT'], row['TS'], row['EXT'], row['V_COOL'], row['spacing'], row['phase'], row['fraction']))
    print("=" * 80)

    ort_val = _safe_float(row.get("ORT"))
    if ort_val is None:
        ort_val = 850
    tem1, tem0 = ort_val, ort_val

    print("  Initial T: tem1={:.0f}, tem0={:.0f}".format(tem1, tem0))
    print("  A1 = {:.1f}C".format(sim.basic_info.A1))

    state, roll_start_time = run_single_sim(row, tem1=tem1, tem0=tem0)

    time_arr = np.array(state.history_time)
    t0_arr = np.array(state.history_T_0[-1])
    t1_arr = np.array(state.history_T_1[-1])

    print("  Sim time range: {:.1f}s ~ {:.1f}s ({} points)".format(
        time_arr[0], time_arr[-1], len(time_arr)))

    # Stage analysis
    A1 = sim.basic_info.A1
    Bs = 550.0

    # Stage 1: T > A1
    mask1 = t0_arr >= A1
    if mask1.sum() >= 2:
        dt1 = np.diff(time_arr[mask1])
        dT1 = np.diff(t0_arr[mask1])
        cr1 = np.abs(dT1 / np.where(dt1 > 0, dt1, np.inf))
        avg_cr1 = np.mean(cr1)
        t_start = time_arr[mask1][0]
        t_end = time_arr[mask1][-1]
        t_start_t = t0_arr[mask1][0]
        t_end_t = t0_arr[mask1][-1]
        print("\n  Stage 1 (T > A1={:.0f}C): t={:.1f}s~{:.1f}s, T={:.1f}C~{:.1f}C".format(
            A1, t_start, t_end, t_start_t, t_end_t))
        print("    Duration: {:.1f}s, Avg cooling rate: {:.1f} C/s (plan: 20-30 ideal, >10 min)".format(
            t_end - t_start, avg_cr1))
        if avg_cr1 >= 20:
            print("    [OK] Fast cooling through austenite zone")
        elif avg_cr1 >= 10:
            print("    [WARN] Acceptable but below ideal range")
        else:
            print("    [BAD] Too slow, risk of grain boundary cementite network")
        stage1_cr = avg_cr1
    else:
        print("\n  Stage 1 (T > A1): insufficient data")
        stage1_cr = None

    # Stage 2: A1 to Bs (pearlite zone)
    mask2 = (t0_arr <= A1) & (t0_arr >= Bs)
    if mask2.sum() >= 2:
        dt2 = np.diff(time_arr[mask2])
        dT2 = np.diff(t0_arr[mask2])
        cr2 = np.abs(dT2 / np.where(dt2 > 0, dt2, np.inf))
        avg_cr2 = np.mean(cr2)
        t_start = time_arr[mask2][0]
        t_end = time_arr[mask2][-1]
        t_start_t = t0_arr[mask2][0]
        t_end_t = t0_arr[mask2][-1]
        print("\n  Stage 2 (A1={:.0f}C ~ Bs={:.0f}C): t={:.1f}s~{:.1f}s, T={:.1f}C~{:.1f}C".format(
            A1, Bs, t_start, t_end, t_start_t, t_end_t))
        print("    Duration: {:.1f}s, Avg cooling rate: {:.1f} C/s (plan: 9-12 ideal)".format(
            t_end - t_start, avg_cr2))
        if 9 <= avg_cr2 <= 12:
            print("    [OK] In ideal pearlite transformation range")
        elif 8 <= avg_cr2 <= 15:
            print("    [WARN] In acceptable range")
        else:
            print("    [BAD] Outside acceptable range")
        stage2_cr = avg_cr2
    else:
        print("\n  Stage 2 (pearlite zone): insufficient data")
        stage2_cr = None

    # Stage 3: Below Bs
    mask3 = t0_arr < Bs
    if mask3.sum() >= 2:
        dt3 = np.diff(time_arr[mask3])
        dT3 = np.diff(t0_arr[mask3])
        cr3 = np.abs(dT3 / np.where(dt3 > 0, dt3, np.inf))
        avg_cr3 = np.mean(cr3)
        t_start = time_arr[mask3][0]
        t_end = time_arr[mask3][-1]
        t_start_t = t0_arr[mask3][0]
        t_end_t = t0_arr[mask3][-1]
        print("\n  Stage 3 (T < Bs={:.0f}C): t={:.1f}s~{:.1f}s, T={:.1f}C~{:.1f}C".format(
            Bs, t_start, t_end, t_start_t, t_end_t))
        print("    Duration: {:.1f}s, Avg cooling rate: {:.1f} C/s (plan: 1-3 ideal, <5 required)".format(
            t_end - t_start, avg_cr3))
        if avg_cr3 <= 3:
            print("    [OK] Slow cooling, safe for residual transformation")
        elif avg_cr3 <= 5:
            print("    [WARN] Acceptable but above ideal")
        else:
            print("    [BAD] >5 C/s, risk of martensite/bainite")
        stage3_cr = avg_cr3
    else:
        print("\n  Stage 3 (below Bs): insufficient data")
        stage3_cr = None

    # dT between overlap/non-overlap
    dT_arr = np.abs(t0_arr - t1_arr)
    max_dt = np.max(dT_arr)
    if mask2.sum() > 0:
        max_dT_pearl = np.max(dT_arr[mask2])
    else:
        max_dT_pearl = np.nan

    print("\n  Temperature uniformity:")
    print("    Max dT (overall): {:.1f}C (plan: <=20)".format(max_dt))
    if not np.isnan(max_dT_pearl):
        print("    Max dT (pearlite zone): {:.1f}C (plan: <=15)".format(max_dT_pearl))

    # Key temperature-time points
    print("\n  Temperature profile (T0 non-overlap surface):")
    for t_target in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70]:
        if t_target <= time_arr[-1]:
            idx_t = min(np.searchsorted(time_arr, t_target), len(t0_arr) - 1)
            print("    t={:.0f}s: T0={:.1f}C, T1={:.1f}C, dT={:.1f}C".format(
                t_target, t0_arr[idx_t], t1_arr[idx_t], abs(t0_arr[idx_t] - t1_arr[idx_t])))

    # Speeds and fans
    print("\n  Process parameters:")
    print("    ORT={:.0f}".format(row['ORT']))
    for i in range(1, 11):
        s = row.get('SPEED{}'.format(i))
        print("    SPEED{}={:.3f}".format(i, s) if pd.notna(s) else "    SPEED{}=NaN".format(i))
    for i in range(1, 8):
        f = row.get('FAN{}'.format(i))
        print("    FAN{}={:.0f}%".format(i, f) if pd.notna(f) else "    FAN{}=NaN".format(i))

    return {
        'stage1_cr': stage1_cr, 'stage2_cr': stage2_cr, 'stage3_cr': stage3_cr,
        'max_dT': max_dt, 'max_dT_pearl': max_dT_pearl,
    }

print("Production Data Analysis with sim_T Verification")
print("Based on plan.txt 3.5.2 three-stage cooling curve")

for label, idx in selected.items():
    row = df.iloc[idx]
    result = simulate_and_analyze(row, label)
