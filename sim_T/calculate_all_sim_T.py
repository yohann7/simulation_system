"""
批量温度仿真 —— 逐条生产数据进行温度仿真并输出 all_sim_T.csv。

核心仿真逻辑调用 sim_T.py 的函数，本模块仅负责：
- 工艺数据加载与校验
- 逐条/并行仿真调度
- 结果拼接与 CSV 输出
"""

import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import sim_T as sim

# 仿真时间步长默认值
DEFAULT_DT = 0.01

TEMP_START_COL = 0
TEMP_END_COL = "ALL"  # "ALL" 表示截取到最后一个温度列

# 默认保留的工艺列
DEFAULT_KEEP_COLUMNS = [
    "C_ELE", "SI_ELE", "MN_ELE", "P_ELE", "S_ELE",
    "CR_ELE", "NI_ELE", "CU_ELE", "TS",
]


# ═══════════════════════════════════════════════════════════════
# 数据加载与预处理
# ═══════════════════════════════════════════════════════════════


def load_process_data(excel_path):
    """读取工艺数据到 DataFrame。"""
    return pd.read_excel(excel_path)


def delete_wrong_rows(df):
    """删除数据缺失的异常行（关键列为 0 的行）。"""
    if df is None or df.empty:
        return df

    element_cols = [col for col in df.columns if "_ELE" in col]
    speed_cols = [f"SPEED{i}" for i in range(1, 11) if f"SPEED{i}" in df.columns]
    check_cols = element_cols + speed_cols
    if "TS" in df.columns:
        check_cols.append("TS")

    if not check_cols:
        return df.copy()

    mask = (df[check_cols] != 0).all(axis=1)
    return df.loc[mask].copy()


# ═══════════════════════════════════════════════════════════════
# 温度列排序与截取
# ═══════════════════════════════════════════════════════════════


def _sort_temp_columns(temp_columns):
    """按时间数值排序温度列。"""
    return sorted(temp_columns, key=lambda col: float(col[:-3]))


def slice_temp_columns(all_sim_t_df, start_col=TEMP_START_COL, end_col=TEMP_END_COL):
    """截取 all_sim_t_df 的非搭接点和搭接点温度列。

    该函数供 predict.py 调用。
    """
    # 被调用: predict.py
    non_overlap_cols = _sort_temp_columns([c for c in all_sim_t_df.columns if c.endswith("(0)")])
    overlap_cols = _sort_temp_columns([c for c in all_sim_t_df.columns if c.endswith("(1)")])

    if not non_overlap_cols or not overlap_cols:
        raise ValueError("all_sim_t_df 中未找到 (0) 或 (1) 温度列。")

    max_len = min(len(non_overlap_cols), len(overlap_cols))
    if isinstance(end_col, str) and end_col.upper() == "ALL":
        end_col = max_len - 1
    if start_col < 0 or end_col < start_col:
        raise ValueError("截取范围不合法。")
    if start_col >= max_len:
        raise ValueError(f"start_col={start_col} 超出温度列范围，总长度={max_len}。")

    end_col = min(end_col, max_len - 1)
    selected_non_overlap = non_overlap_cols[start_col : end_col + 1]
    selected_overlap = overlap_cols[start_col : end_col + 1]

    return all_sim_t_df[selected_non_overlap].copy(), all_sim_t_df[selected_overlap].copy()


def _find_min_valid_temp_count(all_sim_t_df):
    """找出所有行中最短的有效温度序列长度。"""
    non_overlap_cols = _sort_temp_columns(
        [c for c in all_sim_t_df.columns if c.endswith("(0)")]
    )
    if not non_overlap_cols:
        return 0

    min_count = None
    for _, row in all_sim_t_df[non_overlap_cols].iterrows():
        valid_count = int(row.notna().sum())
        if min_count is None or valid_count < min_count:
            min_count = valid_count

    return min_count or 0


def merge_process_and_temp(process_data, resampled_non_overlap, resampled_overlap,
                           keep_columns=DEFAULT_KEEP_COLUMNS):
    """将工艺列与重采样温度列拼接。

    该函数供 predict.py 调用。
    """
    # 被调用: predict.py
    exist_cols = [c for c in keep_columns if c in process_data.columns]
    missing_cols = [c for c in keep_columns if c not in process_data.columns]
    if missing_cols:
        print(f"警告: 以下工艺列不存在，已忽略: {missing_cols}")

    process_keep = process_data[exist_cols].reset_index(drop=True)
    temp_df = pd.concat(
        [
            resampled_non_overlap.reset_index(drop=True),
            resampled_overlap.reset_index(drop=True),
        ],
        axis=1,
    )
    return pd.concat([process_keep, temp_df], axis=1)


# ═══════════════════════════════════════════════════════════════
# 工艺参数映射
# ═══════════════════════════════════════════════════════════════


def _safe_float(value):
    """将值安全转换为 float，异常或空值时返回 None。"""
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_percent(value):
    """将百分比归一化为 0~1；若已是 0~1 则保持不变。"""
    if value is None:
        return None
    if value > 1.0:
        return value / 100.0
    return value


def _get_initial_temperature_from_row(row, default_tem1=850, default_tem0=830):
    """从 ORT 列读取该条数据的吐丝温度，作为初始温度。

    该函数供 predict.py 调用。
    """
    # 被调用: predict.py
    ort = _safe_float(row.get("ORT"))
    if ort is None:
        return default_tem1, default_tem0, False
    return ort, ort, True


def _apply_process_row_to_sim(row, rolls):
    """将一条生产数据映射到仿真参数（basic_info 元素含量和辊道速度/风机开度）。"""
    # 元素含量映射（转换为质量分数 /100）
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

    # 辊道速度映射 (SPEED1~SPEED10)
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

    # 入口段速度
    entry_speed = _safe_float(row.get("SPEED1"))
    if entry_speed is not None and entry_speed > 0:
        old_v = rolls[0].roll_v
        rolls[0].roll_v = entry_speed
        rolls[0].t = rolls[0].t * (old_v / entry_speed)
        rolls[0].step = int(rolls[0].t / sim._default_dt)

    # 风机开度映射 (FAN1~FAN10)
    fan_map = {f"FAN{i}": i for i in range(1, 11)}
    for col, idx in fan_map.items():
        val = _safe_float(row.get(col))
        val = _normalize_percent(val)
        if val is not None:
            rolls[idx].fan_status = val
            rolls[idx].fan_speed = rolls[idx].fan_air_volume * val / rolls[idx].fan_area


# ═══════════════════════════════════════════════════════════════
# 单条仿真
# ═══════════════════════════════════════════════════════════════


def _build_param_from_dict(param_dict, num=0):
    """从字典构建 parameter_change 对象。"""
    if param_dict is None:
        return None
    param_obj = sim.parameter_change(num)
    for key, value in param_dict.items():
        if hasattr(param_obj, key):
            setattr(param_obj, key, float(value))
    return param_obj


def _format_time_col(t, suffix):
    """将时间戳格式化为温度列名。"""
    return f"{float(t):.2f}{suffix}"


def _resample_to_integer_seconds(history_time, history_t0, history_t1):
    """将仿真结果重采样到整数秒（0, 1, 2, ...）。"""
    time_arr = np.asarray(history_time, dtype=float)
    t0_arr = np.asarray(history_t0, dtype=float)
    t1_arr = np.asarray(history_t1, dtype=float)

    max_t = int(np.floor(time_arr[-1]))
    int_times = np.arange(0, max_t + 1, dtype=float)

    t0_int = np.interp(int_times, time_arr, t0_arr)
    t1_int = np.interp(int_times, time_arr, t1_arr)

    return list(int_times), list(t0_int), list(t1_int)


def run_single_simulation(row, tem1=850, tem0=830, dt_override=None, param_dict=None):
    """对单条生产数据执行温度仿真，返回全量时间、表面温度序列及仿真状态。

    被调用: predict.py

    返回: (history_time, history_t0, history_t1, state, roll_start_time)
        state 和 roll_start_time 供 predict.py 调用 sim.plot_T_results 绘图。
    """
    # 被调用: predict.py
    dt = float(dt_override) if dt_override is not None else DEFAULT_DT
    params = _build_param_from_dict(param_dict)

    rolls, num_rolls = sim.data_loader.load_roll_data()
    _apply_process_row_to_sim(row, rolls)

    state, roll_start_time = sim.run_full_simulation(rolls, tem1=tem1, tem0=tem0, params=params, dt=dt)

    history_time = list(state.history_time)
    history_t0 = list(state.history_T_0[-1])
    history_t1 = list(state.history_T_1[-1])
    min_len = min(len(history_time), len(history_t0), len(history_t1))

    resampled = _resample_to_integer_seconds(
        history_time[:min_len], history_t0[:min_len], history_t1[:min_len]
    )
    return resampled[0], resampled[1], resampled[2], state, roll_start_time


# ═══════════════════════════════════════════════════════════════
# 批量并行仿真
# ═══════════════════════════════════════════════════════════════


def _worker_simulate(item):
    """多进程 worker: item = (idx, row_dict, param_dict, dt_override, return_state)。

    返回: (idx, sim_cols, state_data | None, error)
        state_data 为 dict 或 None，包含相变数据供约束评价使用。
    """
    idx, row_dict, param_dict, dt_override, return_state = item
    try:
        row = pd.Series(row_dict)
        tem1, tem0, _ = _get_initial_temperature_from_row(row)
        history_time, history_t0, history_t1, state, _roll_rt = run_single_simulation(
            row, tem1=tem1, tem0=tem0,
            dt_override=dt_override, param_dict=param_dict,
        )

        sim_cols = {}
        for t, t0 in zip(history_time, history_t0):
            sim_cols[_format_time_col(t, "(0)")] = float(t0)
        for t, t1 in zip(history_time, history_t1):
            sim_cols[_format_time_col(t, "(1)")] = float(t1)

        # 提取相变数据供约束评价（避免跨进程传输完整 SimulationState）
        state_data = None
        if return_state:
            # 温度已重采样到整数秒，但相变数据仍为仿真步长（dt=0.01s），
            # 需统一重采样以对齐数组维度。
            int_time = np.array(history_time, dtype=np.float64)
            full_time = np.array(state.history_time, dtype=np.float64)
            pearl_surf = np.array(state.pearlite_0[-1], dtype=np.float64)
            ferrite_surf = np.array(state.ferrite_0[-1], dtype=np.float64)
            state_data = {
                "time": int_time,
                "T0": np.array(history_t0, dtype=np.float64),
                "T1": np.array(history_t1, dtype=np.float64),
                "pearlite_0_surface": np.interp(int_time, full_time, pearl_surf),
                "ferrite_0_surface": np.interp(int_time, full_time, ferrite_surf),
                "ferrite_final_0": np.array(state.ferrite_final_0, dtype=np.float64),
                "f_total_0": np.array(state.f_total_0, dtype=np.float64),
            }

        return idx, sim_cols, state_data, None
    except Exception as e:
        return idx, None, None, str(e)


def run_all_simulations(process_data, n_workers=0, dt_override=None, params_list=None,
                        return_states=False):
    """逐条仿真并输出全量温度，列顺序为全部 (0) 后全部 (1)。

    被调用: change_parameter_82A.py, predict.py

    参数:
        process_data: pandas.DataFrame
        n_workers: 并行进程数。0 或 None 则使用 CPU 核心数。
        dt_override: 覆盖仿真时间步长
        params_list: list[dict] | dict | None，每条数据的参数字典
        return_states: bool, 若为 True 则额外返回相变状态数据列表

    返回:
        return_states=False: pandas.DataFrame（温度时间列）
        return_states=True:  (pandas.DataFrame, list[dict])
            列表中每个 dict 包含 time, T0, T1, pearlite_0_surface,
            ferrite_0_surface, ferrite_final_0, f_total_0
    """
    if params_list is not None and not isinstance(params_list, list):
        params_list = [params_list] * len(process_data)

    if n_workers is None or n_workers == 0:
        n_workers = os.cpu_count() or 1

    total_rows = len(process_data)
    items = []
    for idx, (_, row) in enumerate(process_data.iterrows(), start=1):
        param_dict = None
        if params_list is not None:
            param_dict = params_list[idx - 1]
        items.append((idx, row.to_dict(), param_dict, dt_override, return_states))

    sim_rows = [None] * (total_rows + 1)  # 1-based index
    all_columns = set()
    all_state_data = [None] * (total_rows + 1) if return_states else None

    if total_rows == 0:
        return (pd.DataFrame(), []) if return_states else pd.DataFrame()

    if n_workers == 1 or total_rows <= 1:
        for item in items:
            idx = item[0]
            print(f"仿真进度: {idx}/{total_rows}")
            idx_res, sim_cols, state_data, err = _worker_simulate(item)
            if err:
                raise RuntimeError(f"仿真失败 idx={idx}: {err}")
            sim_rows[idx] = sim_cols
            all_columns.update(sim_cols.keys())
            if return_states:
                all_state_data[idx] = state_data
    else:
        pool_size = min(n_workers, total_rows)
        print(f"使用并行进程数: {pool_size}，总任务数: {total_rows}")
        with Pool(processes=pool_size) as pool:
            for idx, sim_cols, state_data, err in pool.imap_unordered(_worker_simulate, items):
                if err:
                    raise RuntimeError(f"仿真失败 idx={idx}: {err}")
                sim_rows[idx] = sim_cols
                all_columns.update(sim_cols.keys())
                if return_states:
                    all_state_data[idx] = state_data

    sim_rows_list = [sim_rows[i] for i in range(1, total_rows + 1)]
    sim_df = pd.DataFrame(sim_rows_list)

    non_overlap_cols = sorted(
        [c for c in all_columns if c.endswith("(0)")],
        key=lambda col: float(col.split("(")[0]),
    )
    overlap_cols = sorted(
        [c for c in all_columns if c.endswith("(1)")],
        key=lambda col: float(col.split("(")[0]),
    )
    ordered_cols = non_overlap_cols + overlap_cols
    for col in ordered_cols:
        if col not in sim_df.columns:
            sim_df[col] = np.nan

    result_df = sim_df[ordered_cols]
    if return_states:
        state_list = [all_state_data[i] for i in range(1, total_rows + 1)]
        return result_df, state_list
    return result_df


# ═══════════════════════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    all_sim_t_path = Path(__file__).resolve().parent / "output_data" / "all_sim_T.csv"
    process_data_new_path = Path(__file__).resolve().parent / "output_data" / "process_data_new.csv"
    excel_path = project_root / "工艺数据全.xlsx"

    process_data = load_process_data(excel_path)
    print(f"原始 process_data shape: {process_data.shape}")
    process_data = delete_wrong_rows(process_data)
    print(f"删除异常行后 process_data shape: {process_data.shape}")

    all_sim_t = run_all_simulations(process_data)
    all_sim_t.to_csv(all_sim_t_path, index=False, encoding="utf-8-sig")

    min_temp_count = _find_min_valid_temp_count(all_sim_t)
    print(f"最短温度序列长度: {min_temp_count}")
    non_overlap_df, overlap_df = slice_temp_columns(
        all_sim_t,
        start_col=TEMP_START_COL,
        end_col=min_temp_count - 1 if min_temp_count > 0 else 0,
    )
    process_data_new = merge_process_and_temp(
        process_data, non_overlap_df, overlap_df,
        keep_columns=DEFAULT_KEEP_COLUMNS,
    )
    process_data_new.to_csv(process_data_new_path, index=False, encoding="utf-8-sig")

    print(f"process_data shape: {process_data.shape}")
    print(f"all_sim_t shape: {all_sim_t.shape}")
    print(f"non_overlap_dim: {non_overlap_df.shape[1]}")
    print(f"overlap_dim: {overlap_df.shape[1]}")
    print(f"process_data_new shape: {process_data_new.shape}")
    print(f"saved all_sim_t: {all_sim_t_path}")
    print(f"saved process_data_new: {process_data_new_path}")
