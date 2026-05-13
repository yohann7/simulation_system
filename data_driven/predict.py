"""
力学性能预测 —— 从工艺参数到温度仿真到 TS 预测的全流程。

调用 sim_T 和 calculate_all_sim_T 进行温度仿真，
调用 data_driven_model 加载模型并预测力学性能 TS。
"""

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import data_driven_model as ddm

# 添加 sim_T 目录到 sys.path，以便导入 calculate_all_sim_T 和 sim_T
_ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(_ROOT_DIR / "sim_T"))
import calculate_all_sim_T as calc
import sim_T as sim

TEMP_START_COL = 0
TEMP_END_COL = "ALL"

# 全局变量：存储最近一次仿真结果，供 plot_T_results 使用
_last_sim_result = {}


# ═══════════════════════════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════════════════════════


def _resolve_device(device):
    """解析设备参数。"""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


@lru_cache(maxsize=1)
def _load_predict_assets():
    """加载预测所需的模型参数和归一化参数。"""
    model_dir = Path(__file__).resolve().parent / "param" / "predict"
    model_path = model_dir / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    try:
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise TypeError("模型文件格式不正确，需要包含 model_state。")

    model_config = checkpoint.get("model_config")
    if not model_config:
        raise KeyError("模型文件缺少 model_config。")

    norm_params = ddm.unpack_norm_params(checkpoint.get("norm"))
    model_seq_len = checkpoint.get("meta", {}).get("seq_len", None)
    return checkpoint["model_state"], model_config, norm_params, model_seq_len


@lru_cache(maxsize=8)
def _get_model_runtime(input_dim, seq_len, device_str):
    """获取模型推理实例。"""
    model_state, model_config, norm_params, model_seq_len = _load_predict_assets()
    expected_dim = model_config.get("input_dim")
    if expected_dim is not None and expected_dim != input_dim:
        raise ValueError(f"模型输入维度不匹配: checkpoint={expected_dim}, 当前={input_dim}")

    ddm.SEQ_LEN = seq_len
    model = ddm.Transformer_Decoder(**model_config)
    model.load_state_dict(model_state, strict=True)
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()
    return model, norm_params, device


# ═══════════════════════════════════════════════════════════════
# 温度仿真
# ═══════════════════════════════════════════════════════════════


def predict_temperatures(process_data):
    """基于单条工艺数据计算仿真温度，返回 time, tem0, tem1。

    同时将 state 和 roll_start_time 存入 _last_sim_result，
    供 plot_T_results() 调用 sim_T.plot_T_results() 使用。

    被调用: predict.py __main__
    """
    if isinstance(process_data, pd.DataFrame):
        if process_data.empty:
            raise ValueError("process_data 不能为空 DataFrame")
        row = process_data.iloc[0]
    elif isinstance(process_data, pd.Series):
        row = process_data
    else:
        raise TypeError("process_data 仅支持 pandas.DataFrame 或 pandas.Series")

    tem1, tem0, _ = calc._get_initial_temperature_from_row(row)
    time, tem0, tem1, state, roll_start_time = calc.run_single_simulation(
        row, tem1=tem1, tem0=tem0,
    )

    # 存储供 sim_T.plot_T_results 调用
    _last_sim_result["time"] = time
    _last_sim_result["tem0"] = tem0
    _last_sim_result["tem1"] = tem1
    _last_sim_result["state"] = state
    _last_sim_result["roll_start_time"] = roll_start_time

    return time, tem0, tem1


def data_splicing(time, tem0, tem1, process_data=None):
    """将仿真产生的全部温度与工艺参数拼接成 process_data_new。

    被调用: predict.py __main__, 外部
    """
    if not (len(time) == len(tem0) == len(tem1)):
        raise ValueError("time、tem0、tem1 长度必须一致。")

    if process_data is None:
        process_data = globals().get("demo", None)
    if process_data is None:
        raise ValueError("未提供 process_data，且未找到全局 demo。")
    if isinstance(process_data, pd.Series):
        process_data = process_data.to_frame().T
    elif isinstance(process_data, pd.DataFrame):
        if process_data.empty:
            raise ValueError("process_data 不能为空。")
        process_data = process_data.iloc[[0]].copy()
    else:
        raise TypeError("process_data 仅支持 pandas.DataFrame 或 pandas.Series。")

    sim_cols = {}
    for t, t0, t1 in zip(time, tem0, tem1):
        t_key = f"{float(t):.2f}"
        sim_cols[f"{t_key}(0)"] = float(t0)
        sim_cols[f"{t_key}(1)"] = float(t1)

    all_sim_t_df = pd.DataFrame([sim_cols])

    # 调用 calculate_all_sim_T 中的截取和拼接函数
    non_overlap_df, overlap_df = calc.slice_temp_columns(
        all_sim_t_df, start_col=TEMP_START_COL, end_col=TEMP_END_COL,
    )
    process_data_new = calc.merge_process_and_temp(
        process_data, non_overlap_df, overlap_df,
        keep_columns=list(process_data.columns),
    )
    return process_data_new


# 兼容旧名称
resample_sim_data = data_splicing


# ═══════════════════════════════════════════════════════════════
# TS 预测
# ═══════════════════════════════════════════════════════════════


def predict_Ts(process_data, device="auto"):
    """基于 process_data_new 预测力学性能 TS。

    被调用: predict.py __main__, 外部
    """
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

    _, _, norm_params, model_seq_len = _load_predict_assets()
    if model_seq_len is not None and model_seq_len < len(temp_t0_cols):
        temp_t0_cols = temp_t0_cols[:model_seq_len]
        temp_t1_cols = temp_t1_cols[:model_seq_len]

    static_x = torch.as_tensor(process_data[element_cols].to_numpy(dtype="float32"))
    t0_x = torch.as_tensor(process_data[temp_t0_cols].to_numpy(dtype="float32"))
    t1_x = torch.as_tensor(process_data[temp_t1_cols].to_numpy(dtype="float32"))

    seq_len = len(temp_t0_cols)
    ddm.SEQ_LEN = seq_len
    static_params = norm_params["static_z_score"]
    temp_params = norm_params["temp_z_score"]
    target_params = norm_params["target_z_score"]

    static_scaled = ddm.z_score_transform(static_x, static_params)
    temp_x = torch.stack([t0_x, t1_x], dim=2)
    temp_scaled = ddm.z_score_transform(temp_x.reshape(-1, 2), temp_params).reshape(temp_x.shape)
    static_seq = static_scaled.unsqueeze(1).repeat(1, seq_len, 1)
    x = torch.cat([static_seq, temp_scaled], dim=2).to(torch.float32)

    resolved_device = _resolve_device(device)
    model, norm_params, runtime_device = _get_model_runtime(
        input_dim=x.shape[-1], seq_len=seq_len, device_str=str(resolved_device),
    )

    with torch.inference_mode():
        x_tensor = x.to(runtime_device)
        pred_scaled = model(x_tensor).detach().cpu().view(-1)
    pred = ddm.z_score_inverse_transform(pred_scaled, target_params)
    Ts = float(pred[0]) if pred.numel() == 1 else pred.cpu().numpy()
    return Ts


# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════


def plot_T_results():
    """绘制仿真温度曲线 —— 直接调用 sim_T.plot_T_results()。"""
    # 被调用: sim_T.plot_T_results (sim_T.py)
    if "state" not in _last_sim_result or "roll_start_time" not in _last_sim_result:
        raise RuntimeError("需要先运行 predict_temperatures() 以生成仿真结果。")
    sim.plot_T_results(_last_sim_result["state"], _last_sim_result["roll_start_time"])
    plt.show()


# ═══════════════════════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    process_data_path = Path(__file__).resolve().parents[1] / "工艺数据全.xlsx"
    process_df = pd.read_excel(process_data_path)

    n = 51
    demo = process_df.iloc[[n - 2]]

    time, tem0, tem1 = predict_temperatures(demo)
    process_data_new = resample_sim_data(time, tem0, tem1)
    print(process_data_new)
    Ts = predict_Ts(process_data_new)

    # 绘制温度仿真图
    plot_T_results()

    print(f"预测 TS: {Ts:.2f}; 真实 TS: {demo['TS'].iloc[0]:.2f}; "
          f"差值：{(Ts - demo['TS'].iloc[0]):.2f}")
