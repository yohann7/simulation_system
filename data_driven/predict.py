import pandas as pd
from pathlib import Path
import sys
import numpy as np
import torch
from functools import lru_cache
import data_driven_model as ddm

_ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT_DIR / "sim_T"))
import calculate_all_sim_T as calc

TEMP_START_COL = 0
TEMP_END_COL = "ALL"


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


def merge_process_and_temp(process_data, resampled_non_overlap, resampled_overlap, keep_columns):
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


def _get_norm_array(npz_data, candidate_keys):
    for key in candidate_keys:
        if key in npz_data:
            return np.asarray(npz_data[key], dtype=np.float32)
    for key in npz_data.files:
        low_key = key.lower()
        if any(token in low_key for token in candidate_keys):
            return np.asarray(npz_data[key], dtype=np.float32)
    raise KeyError(f"norm_params 缺少键: {candidate_keys}")


def _build_model_for_state(state_dict, input_dim):
    import inspect

    # 优先尝试最常见模型名，再尝试 ddm 中全部 nn.Module 子类。
    preferred_names = ["MultiScaleConvRegressor", "Regressor", "Model", "Net"]
    module_classes = []
    for _, cls in inspect.getmembers(ddm, inspect.isclass):
        if cls.__module__ != ddm.__name__:
            continue
        if not issubclass(cls, torch.nn.Module):
            continue
        if cls.__name__ in {"LpLoss", "PositionalEncoding"}:
            continue
        module_classes.append(cls)

    def _priority(cls):
        for i, token in enumerate(preferred_names):
            if token.lower() in cls.__name__.lower():
                return i
        return len(preferred_names)

    module_classes = sorted(module_classes, key=_priority)

    last_error = None
    for cls in module_classes:
        for kwargs in ({"input_dim": input_dim}, {}):
            try:
                model_obj = cls(**kwargs)
            except Exception:
                continue
            try:
                model_obj.load_state_dict(state_dict, strict=True)
                return model_obj
            except Exception as exc:
                last_error = exc
                continue

    raise RuntimeError(
        f"无法在 data_driven_model 中找到可匹配当前权重的模型类。最后错误: {last_error}"
    )


def _resolve_device(device):
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


@lru_cache(maxsize=1)
def _load_predict_assets():
    model_dir = Path(__file__).resolve().parent/ "param" / "predict"
    model_candidates = sorted(model_dir.glob("fine_tuned_head_only*.pth"))
    if not model_candidates:
        raise FileNotFoundError(f"未找到模型参数文件: {model_dir}"
                                " (匹配 fine_tuned_head_only*.pth)")
    model_path = model_candidates[-1]

    norm_path = model_dir / "norm_params.npz"
    if not norm_path.exists():
        raise FileNotFoundError(f"未找到归一化参数文件: {norm_path}")

    norm_npz = np.load(norm_path)
    static_params = {
        "mean": _get_norm_array(norm_npz, ["static_mean"]),
        "std": _get_norm_array(norm_npz, ["static_std"]),
    }
    temp_params = {
        "mean": _get_norm_array(norm_npz, ["temp_mean"]),
        "std": _get_norm_array(norm_npz, ["temp_std"]),
    }
    target_params = {
        "mean": _get_norm_array(norm_npz, ["target_mean"]),
        "std": _get_norm_array(norm_npz, ["target_std"]),
    }

    raw_state = torch.load(model_path, map_location="cpu")
    state = raw_state.get("state_dict") if isinstance(raw_state, dict) and "state_dict" in raw_state else raw_state
    if not isinstance(state, dict):
        raise TypeError("模型参数文件格式不支持，需为 state_dict 或包含 state_dict 的字典。")

    return state, static_params, temp_params, target_params


@lru_cache(maxsize=8)
def _get_model_runtime(input_dim, seq_len, device_str):
    state, static_params, temp_params, target_params = _load_predict_assets()
    ddm.SEQ_LEN = seq_len
    model = _build_model_for_state(state, input_dim=input_dim)
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()
    return model, static_params, temp_params, target_params, device


def predict_temperatures(process_data):
    """基于单条工艺数据计算仿真温度，返回 time, tem0, tem1。"""
    if isinstance(process_data, pd.DataFrame):
        if process_data.empty:
            raise ValueError("process_data 不能为空 DataFrame")
        row = process_data.iloc[0]
    elif isinstance(process_data, pd.Series):
        row = process_data
    else:
        raise TypeError("process_data 仅支持 pandas.DataFrame 或 pandas.Series")

    tem1, tem0, _ = calc._get_initial_temperature_from_row(row)
    time, tem0, tem1 = calc.run_single_simulation(row, tem1=tem1, tem0=tem0)
    return time, tem0, tem1

def data_splicing(time, tem0, tem1, process_data=None):
    """将仿真产生的全部温度与工艺参数拼接成 process_data_new（不做重采样）。

    保持原签名以便兼容调用，但本函数只负责拼接全部 (0) 列后接 (1) 列。
    """
    if not (len(time) == len(tem0) == len(tem1)):
        raise ValueError("time、tem0、tem1 长度必须一致。")

    # 优先使用显式传入的工艺数据，未传入时再回退到历史全局变量 demo。
    if process_data is None:
        process_data = globals().get("demo", None)
    if process_data is None:
        raise ValueError("未提供 process_data，且未找到全局 demo，无法拼接 process_data_new。")
    if isinstance(process_data, pd.Series):
        process_data = process_data.to_frame().T
    elif isinstance(process_data, pd.DataFrame):
        if process_data.empty:
            raise ValueError("demo 不能为空。")
        process_data = process_data.iloc[[0]].copy()
    else:
        raise TypeError("demo 仅支持 pandas.DataFrame 或 pandas.Series。")

    sim_cols = {}
    for t, t0, t1 in zip(time, tem0, tem1):
        t_key = f"{float(t):.2f}"
        sim_cols[f"{t_key}(0)"] = float(t0)
        sim_cols[f"{t_key}(1)"] = float(t1)

    all_sim_t_df = pd.DataFrame([sim_cols])
    # 直接取全部非搭接点(0)列和搭接点(1)列，不做降采样
    non_overlap_df, overlap_df = slice_temp_columns(
        all_sim_t_df,
        start_col=TEMP_START_COL,
        end_col=TEMP_END_COL,
    )

    process_data_new = merge_process_and_temp(
        process_data,
        non_overlap_df,
        overlap_df,
        keep_columns=list(process_data.columns),
    )
    return process_data_new

# 兼容旧名称
resample_sim_data = data_splicing

def predict_Ts(process_data, device="auto"):
    """基于 process_data_new 预测力学性能 TS。"""

    if isinstance(process_data, pd.Series):
        process_data = process_data.to_frame().T
    elif isinstance(process_data, pd.DataFrame):
        if process_data.empty:
            raise ValueError("process_data 不能为空。")
    else:
        raise TypeError("process_data 仅支持 pandas.DataFrame 或 pandas.Series。")

    element_cols = list(ddm.ELEMENT_COLS)
    missing_cols = [c for c in element_cols if c not in process_data.columns]
    if missing_cols:
        raise ValueError(f"process_data 缺少必要元素列: {missing_cols}")

    temp_t0_cols, temp_t1_cols = ddm.infer_temp_columns(process_data)
    static_x = process_data[element_cols].to_numpy(dtype=np.float32)
    t0_x = process_data[temp_t0_cols].to_numpy(dtype=np.float32)
    t1_x = process_data[temp_t1_cols].to_numpy(dtype=np.float32)

    seq_len = len(temp_t0_cols)
    ddm.SEQ_LEN = seq_len

    _, static_params, temp_params, _ = _load_predict_assets()

    static_scaled = ddm.z_score_transform(static_x, static_params)
    temp_x = np.stack([t0_x, t1_x], axis=2).astype(np.float32)
    temp_scaled = ddm.z_score_transform(temp_x.reshape(-1, 2), temp_params).reshape(temp_x.shape)
    static_seq = np.repeat(static_scaled[:, None, :], seq_len, axis=1)
    x = np.concatenate([static_seq, temp_scaled], axis=2).astype(np.float32)

    resolved_device = _resolve_device(device)
    model, _, _, target_params, runtime_device = _get_model_runtime(
        input_dim=x.shape[-1],
        seq_len=seq_len,
        device_str=str(resolved_device),
    )

    with torch.inference_mode():
        x_tensor = torch.tensor(x, dtype=torch.float32, device=runtime_device)
        pred_scaled = model(x_tensor).detach().cpu().numpy().reshape(-1)
    pred = ddm.z_score_inverse_transform(pred_scaled, target_params)
    Ts = float(pred[0]) if pred.shape[0] == 1 else pred
    return Ts

def plot_T_results():
    """绘制仿真温度曲线，并在每段辊道时间区间绘制虚线和交替填充背景色。"""
    import matplotlib.pyplot as plt

    # 取全局计算得到的 time, tem0, tem1（predict_temperatures 调用后应已生成）
    if "time" not in globals() or "tem0" not in globals() or "tem1" not in globals():
        raise RuntimeError("需要先运行 predict_temperatures() 以生成 time, tem0, tem1。")

    t = globals()["time"]
    t0 = globals()["tem0"]
    t1 = globals()["tem1"]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, t0, label="Non-overlap Surface (0)")
    ax.plot(t, t1, label="Overlap Surface (1)")

    # 如果 calculate_all_sim_T 中的 sim 模块保存了 roll_start_time，优先使用它进行分段显示
    roll_start_time = None
    try:
        roll_start_time = calc.sim.roll_start_time
    except Exception:
        roll_start_time = None

    if roll_start_time and len(roll_start_time) >= 2:
        # 构造大段和小段的起始时间索引（与 sim_T.py 保持一致）
        big_idx = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26]
        small_idx = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

        big_start_time = [float(roll_start_time[i]) for i in big_idx if i < len(roll_start_time)]
        small_start_time = [float(roll_start_time[i]) for i in small_idx if i < len(roll_start_time)]

        # 先画交替背景色（按大段区间填充）
        bg_colors = ['#f6f8fb', '#fbf8f3']
        for i in range(len(big_start_time) - 1):
            ax.axvspan(big_start_time[i], big_start_time[i + 1], facecolor=bg_colors[i % 2], alpha=1, zorder=0)

        # 使用当前 y 轴范围作为竖线的上下界
        ymin, ymax = plt.ylim()

        # 小段分割线：浅灰虚线
        if small_start_time:
            ax.vlines(small_start_time, ymin=ymin, ymax=ymax, linestyles='--', colors='gray', alpha=0.7)

        # 大段分割线：深黑虚线
        if big_start_time:
            ax.vlines(big_start_time, ymin=ymin, ymax=ymax, linestyles='--', colors='black', alpha=0.7)
    else:
        # 回退：如果没有 roll_start_time，则在每个 time 点绘制浅灰虚线
        ymin, ymax = plt.ylim()
        ax.vlines(t, ymin=ymin, ymax=ymax, linestyles='--', colors='gray', alpha=0.3)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Simulation Temperature Profile")
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    process_data_path = Path(__file__).resolve().parents[1] / "工艺数据全.xlsx"
    process_df = pd.read_excel(process_data_path)
    # demo = pd.DataFrame({
    #     "C_ELE":0.885, 
    #     "SI_ELE":0.225,
    #     "MN_ELE":0.478,
    #     "P_ELE":0.0077,
    #     "S_ELE":0.0026,
    #     "CR_ELE":0.012,
    #     "NI_ELE":0.0166,
    #     "CU_ELE":0.010,
    #     "ORT":840,
    #     "SPEED1": 0.91,
    #     "SPEED2": 0.88,
    #     "SPEED3": 0.96,
    #     "SPEED4": 1.07,
    #     "SPEED5": 0.43,
    #     "SPEED6": 0.43,
    #     "SPEED7": 1.18,
    #     "SPEED8": 0.43,
    #     "SPEED9": 0.64,
    #     "SPEED10": 1.49,
    #     "FAN1": 35,
    #     "FAN2": 4.0,
    #     "FAN3": 3.0,
    #     "FAN4": 0.0,
    #     "FAN5": 37,
    #     "FAN6": 11,
    #     "TS": 1047
    #     },index=[0])

    n = 15
    demo = process_df.iloc[[n-2]]
    
    time, tem0, tem1 = predict_temperatures(demo)
    process_data_new = resample_sim_data(time, tem0, tem1)
    print(process_data_new)
    Ts = predict_Ts(process_data_new)
    
    #调用sim文件的画图函数，画出温度仿真图
    plot_T_results()

    print(f"预测 TS: {Ts:.2f}; ",f"真实 TS: {demo['TS'].iloc[0]:.2f}; ",f"差值：{(Ts - demo['TS'].iloc[0]):.2f}")


