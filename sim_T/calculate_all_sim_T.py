'''
该程序仅用于逐条生产数据进行温度仿真，并输出 all_sim_T.csv。
输出仅包含温度列：先非搭接点(0)，后搭接点(1)，列名为仿真时间加后缀。
'''

from pathlib import Path

import numpy as np
import pandas as pd
import os
from multiprocessing import Pool

import sim_T as sim


# 仿真时间步长默认值（可在 run_all_simulations/run_single_simulation 中覆盖）
DEFAULT_DT = 0.01
dt = DEFAULT_DT

TEMP_START_COL = 0
TEMP_END_COL = "ALL"  # TEMP_END_COL = "ALL" 时，表示截取到最后一个温度列。
RE_SAMPLE_N = 100

# 默认保留的工艺列，可按需增删。
DEFAULT_KEEP_COLUMNS = [
	"C_ELE",
	"SI_ELE",
	"MN_ELE",
	"P_ELE",
	"S_ELE",
	"CR_ELE",
	"NI_ELE",
	"CU_ELE",
	"TS",
]


def load_process_data(excel_path):
	"""读取工艺数据到 DataFrame。"""
	return pd.read_excel(excel_path)


def _sort_temp_columns(temp_columns):
	"""按时间数值排序温度列。"""
	return sorted(temp_columns, key=lambda col: float(col[:-3]))


def slice_temp_columns(all_sim_t_df, start_col=TEMP_START_COL, end_col=TEMP_END_COL):
	"""截取 all_sim_t_df 的非搭接点和搭接点温度列。"""
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
	selected_non_overlap = non_overlap_cols[start_col:end_col + 1]
	selected_overlap = overlap_cols[start_col:end_col + 1]

	return all_sim_t_df[selected_non_overlap].copy(), all_sim_t_df[selected_overlap].copy()


# def resample_temp_data(non_overlap_df, overlap_df, n=RE_SAMPLE_N):
# 	"""对截取后的温度数据重采样。"""
# 	if n < 1:
# 		raise ValueError("n 必须大于等于 1。")

# 	non_overlap_cols = list(non_overlap_df.columns[::n])
# 	overlap_cols = list(overlap_df.columns[::n])

# 	resampled_non_overlap = non_overlap_df[non_overlap_cols].copy()
# 	resampled_overlap = overlap_df[overlap_cols].copy()

# 	non_overlap_dim = resampled_non_overlap.shape[1]
# 	overlap_dim = resampled_overlap.shape[1]
# 	return resampled_non_overlap, resampled_overlap, non_overlap_dim, overlap_dim


def _find_min_valid_temp_count(all_sim_t_df):
    """找出所有行中最短的有效温度序列长度。

    不同工艺数据的总仿真时间不同，导致温度列数不一致。
    返回所有行中非搭接点温度有效值个数的最小值，作为统一截取长度。
    """
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


def merge_process_and_temp(
	process_data,
	resampled_non_overlap,
	resampled_overlap,
	keep_columns=DEFAULT_KEEP_COLUMNS,
):
	"""将工艺列与重采样温度列拼接。"""
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
	"""从 ORT 列读取该条数据的吐丝温度，作为 tem1/tem0 初始温度。"""
	ort = _safe_float(row.get("ORT"))
	if ort is None:
		return default_tem1, default_tem0, False
	return ort, ort, True


def _build_param_from_dict(param_dict, num=0):
	"""从字典构建 parameter_change 对象。"""
	if param_dict is None:
		return None
	param_obj = sim.parameter_change(num)
	for key, value in param_dict.items():
		if hasattr(param_obj, key):
			setattr(param_obj, key, float(value))
	return param_obj


def _init_current_params(param_dict=None, dt_override=None):
	"""初始化与 sim_v1_82A.py 主程序一致的修正参数。"""
	if dt_override is None:
		dt_override = dt
	sim.simulation_model.dt = float(dt_override)

	if param_dict is None:
		sim.simulation_model.current_params = sim.build_parameter_from_file(0)
	else:
		sim.simulation_model.current_params = _build_param_from_dict(param_dict)


def _reset_simulation_state():
	"""重置仿真状态，保证每条生产数据独立仿真。"""
	sm = sim.simulation_model
	n = sm.N

	sm.f_total_0 = np.zeros(n)
	sm.f_total_1 = np.zeros(n)
	sm.suma_f0 = np.zeros(n)
	sm.mark_sf_0 = np.zeros(n, dtype=int)
	sm.mark_ef_0 = np.zeros(n, dtype=int)
	sm.suma_f1 = np.zeros(n)
	sm.mark_sf_1 = np.zeros(n, dtype=int)
	sm.mark_ef_1 = np.zeros(n, dtype=int)
	sm.suma_p0 = np.zeros(n)
	sm.mark_sp_0 = np.zeros(n, dtype=int)
	sm.mark_ep_0 = np.zeros(n, dtype=int)
	sm.suma_p1 = np.zeros(n)
	sm.mark_sp_1 = np.zeros(n, dtype=int)
	sm.mark_ep_1 = np.zeros(n, dtype=int)

	sm.history_time = []
	sm.history_T_0 = [[] for _ in range(n)]
	sm.history_T_1 = [[] for _ in range(n)]
	sm.history_Q_0 = [[] for _ in range(n)]
	sm.history_Q_1 = [[] for _ in range(n)]
	sm.history_h_0 = [[] for _ in range(3)]
	sm.history_h_1 = [[] for _ in range(3)]
	sm.pearlite_0 = [[] for _ in range(n)]
	sm.pearlite_1 = [[] for _ in range(n)]


def _apply_process_row_to_sim(row, rolls):
	"""将一条生产数据映射到仿真参数。"""
	# basic_info 映射：先读取元素含量，再转换为质量分数（/100）
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

	# data_loader 映射：辊道速度（SPEED1~SPEED10）
	sc_map = {
		"SPEED1": 1,
		"SPEED2": 2,
		"SPEED3": 3,
		"SPEED4": 4,
		"SPEED5": 5,
		"SPEED6": 6,
		"SPEED7": 7,
		"SPEED8": 8,
		"SPEED9": 9,
		"SPEED10": 10,
	}
	for col, idx in sc_map.items():
		val = _safe_float(row.get(col))
		if val is not None and val > 0:
			old_v = rolls[idx].roll_v
			rolls[idx].roll_v = val
			rolls[idx].t = rolls[idx].t * (old_v / val)
			rolls[idx].step = int(rolls[idx].t / sim.simulation_model.dt)

	# 新数据无独立入口速度列，使用 SPEED1 作为入口段速度。
	entry_speed = _safe_float(row.get("SPEED1"))
	if entry_speed is not None and entry_speed > 0:
		old_v = rolls[0].roll_v
		rolls[0].roll_v = entry_speed
		rolls[0].t = rolls[0].t * (old_v / entry_speed)
		rolls[0].step = int(rolls[0].t / sim.simulation_model.dt)

	# data_loader 映射：风机开度（FAN1~FAN6）
	puf_map = {
		"FAN1": 1,
		"FAN2": 2,
		"FAN3": 3,
		"FAN4": 4,
		"FAN5": 5,
		"FAN6": 6,
		"FAN7": 7,
	}
	for col, idx in puf_map.items():
		val = _safe_float(row.get(col))
		val = _normalize_percent(val)
		if val is not None:
			rolls[idx].fan_status = val
			rolls[idx].fan_speed = rolls[idx].fan_air_volume * val / rolls[idx].fan_area


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
	"""对单条生产数据执行温度仿真，返回全量时间与表面温度序列。"""
	_init_current_params(param_dict=param_dict, dt_override=dt_override)
	_reset_simulation_state()

	rolls, num_rolls = sim.data_loader.load_roll_data()
	_apply_process_row_to_sim(row, rolls)
	sim.roll_start_time = [0.0]
	for r in rolls:
		sim.roll_start_time.append(sim.roll_start_time[-1] + float(r.t))

	temp_trans_1 = np.full(sim.simulation_model.N, tem1)
	temp_trans_0 = np.full(sim.simulation_model.N, tem0)

	sim.each_roll_time = 0
	for i in range(num_rolls):
		current_roll = rolls[i]
		current_roll.pre_temp_0 = temp_trans_0
		current_roll.pre_temp_1 = temp_trans_1
		sim.simulation_model.Cooling_calculation(current_roll)
		temp_trans_0 = current_roll.post_temp_0
		temp_trans_1 = current_roll.post_temp_1

	history_time_raw = list(sim.simulation_model.history_time)
	history_t0_raw = list(sim.simulation_model.history_T_0[-1])
	history_t1_raw = list(sim.simulation_model.history_T_1[-1])
	min_len = min(len(history_time_raw), len(history_t0_raw), len(history_t1_raw))
	history_time = history_time_raw[:min_len]
	history_t0 = history_t0_raw[:min_len]
	history_t1 = history_t1_raw[:min_len]

	return _resample_to_integer_seconds(history_time, history_t0, history_t1)


def _worker_simulate(item):
	"""Worker for multiprocessing: item is (idx, row_dict, param_dict, dt_override). Returns (idx, sim_cols, error).

	Each worker reconstructs a pandas.Series from the dict and calls run_single_simulation.
	"""
	idx, row_dict, param_dict, dt_override = item
	try:
		row = pd.Series(row_dict)
		tem1, tem0, _ = _get_initial_temperature_from_row(row)
		history_time, history_t0, history_t1 = run_single_simulation(
			row,
			tem1=tem1,
			tem0=tem0,
			dt_override=dt_override,
			param_dict=param_dict,
		)

		sim_cols = {}
		for t, t0 in zip(history_time, history_t0):
			col = _format_time_col(t, "(0)")
			sim_cols[col] = float(t0)
		for t, t1 in zip(history_time, history_t1):
			col = _format_time_col(t, "(1)")
			sim_cols[col] = float(t1)

		return idx, sim_cols, None
	except Exception as e:
		return idx, None, str(e)


def run_all_simulations(process_data, n_workers=0, dt_override=None, params_list=None):
	"""逐条仿真并输出全量温度，列顺序为全部(0)后全部(1)。

	参数:
	- process_data: pandas.DataFrame
	- n_workers: int, 并行工作进程数。默认为 1（串行）。传入 None 或 0 则使用可用 CPU 核心数。
	- dt_override: float, 可选。覆盖仿真时间步长。
	- params_list: list[dict] | dict | None, 可选。为每条数据提供参数字典。
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
		items.append((idx, row.to_dict(), param_dict, dt_override))

	sim_rows = [None] * (total_rows + 1)  # 1-based index
	all_columns = set()

	if n_workers == 1 or total_rows <= 1:
		# 保持原有串行逻辑
		for idx, row_dict, param_dict, dt_override in items:
			print(f"仿真进度: {idx}/{total_rows}")
			idx_res, sim_cols, err = _worker_simulate((idx, row_dict, param_dict, dt_override))
			if err:
				raise RuntimeError(f"仿真失败 idx={idx}: {err}")
			sim_rows[idx] = sim_cols
			all_columns.update(sim_cols.keys())
	else:
		# 并行执行，使用进程池（multiprocessing Pool）以占满可用 CPU
		pool_size = min(n_workers, total_rows)
		print(f"使用并行进程数: {pool_size}，总任务数: {total_rows}")
		with Pool(processes=pool_size) as pool:
			for idx, sim_cols, err in pool.imap_unordered(_worker_simulate, items):
				if err:
					raise RuntimeError(f"仿真失败 idx={idx}: {err}")
				sim_rows[idx] = sim_cols
				all_columns.update(sim_cols.keys())

	# 按原始顺序收集结果
	sim_rows_list = [sim_rows[i] for i in range(1, total_rows + 1)]

	sim_df = pd.DataFrame(sim_rows_list)
	non_overlap_cols = sorted([c for c in all_columns if c.endswith("(0)")], key=lambda col: float(col.split("(")[0]))
	overlap_cols = sorted([c for c in all_columns if c.endswith("(1)")], key=lambda col: float(col.split("(")[0]))
	ordered_cols = non_overlap_cols + overlap_cols
	for col in ordered_cols:
		if col not in sim_df.columns:
			sim_df[col] = np.nan

	return sim_df[ordered_cols]

def delete_wrong_rows(df):
	"""删除数据缺失的异常行。"""
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
	df_new = df.loc[mask].copy()
	return df_new


if __name__ == "__main__":
	project_root = Path(__file__).resolve().parents[1]
	all_sim_t_path = Path(__file__).resolve().parent / "output_data" / "all_sim_T.csv"
	process_data_new_path = Path(__file__).resolve().parent / "output_data" / "process_data_new.csv"
	excel_path = Path(__file__).parents[1] / "工艺数据全.xlsx"

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
	non_overlap_dim = non_overlap_df.shape[1]
	overlap_dim = overlap_df.shape[1]
	process_data_new = merge_process_and_temp(
		process_data,
		non_overlap_df,
		overlap_df,
		keep_columns=DEFAULT_KEEP_COLUMNS,
	)
	process_data_new.to_csv(process_data_new_path, index=False, encoding="utf-8-sig")

	print(f"process_data shape: {process_data.shape}")
	print(f"all_sim_t shape: {all_sim_t.shape}")
	print(f"non_overlap_dim: {non_overlap_dim}")
	print(f"overlap_dim: {overlap_dim}")
	print(f"process_data_new shape: {process_data_new.shape}")
	print(f"saved all_sim_t: {all_sim_t_path}")
	print(f"saved process_data_new: {process_data_new_path}")
