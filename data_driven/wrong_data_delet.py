"""对全量数据进行模型预测并分析预测误差分布。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

path = r"data_model_parameter(4)"
path_model = r"best(7.2216,12.9054,98.3887).pth"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[3]
MODEL_MODULE_PATH = SCRIPT_DIR / "data_driven_model.py"
DATA_PATH = SCRIPT_DIR.parent / "sim_T" / "output_data" / "process_data_new.csv"
OLD_DATA_PATH = SCRIPT_DIR.parent / "sim_T" / "output_data" / "process_data_old.csv"
MODEL_PATH = Path(__file__).resolve().parent / "param" / "train" / path / path_model
NORM_PARAMS_PATH = Path(__file__).resolve().parent / "param" / "train" / path / "norm_params.npz"
ANOMALY_DIFF_THRESHOLD = 30 # 预测误差绝对值超过该阈值的样本将被视为异常值并删除


def load_model_module():
	"""从同目录的 data_driven_model.py 动态加载模型与工具函数。"""
	spec = spec_from_file_location("data_driven_model", MODEL_MODULE_PATH)
	if spec is None or spec.loader is None:
		raise ImportError(f"无法加载模型模块: {MODEL_MODULE_PATH}")
	module = module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


MODEL_MODULE = load_model_module()


def load_source_data(csv_path=DATA_PATH):
	"""读取原始数据并完成与训练脚本一致的列检查逻辑。"""
	if not csv_path.exists():
		raise FileNotFoundError(f"数据文件不存在: {csv_path}")
	df = pd.read_csv(csv_path)
	MODEL_MODULE.TEMP_T0_COLS, MODEL_MODULE.TEMP_T1_COLS = MODEL_MODULE.infer_temp_columns(df)
	MODEL_MODULE.SEQ_LEN = len(MODEL_MODULE.TEMP_T0_COLS)
	MODEL_MODULE.validate_required_columns(df, target_col=MODEL_MODULE.TARGET_COL)
	return df


def build_full_inputs(df, norm_params):
	"""使用训练阶段保存的参数，对全量数据构造模型输入。"""
	static_raw = df[MODEL_MODULE.ELEMENT_COLS].to_numpy(dtype=np.float32)
	static_scaled = MODEL_MODULE.z_score_transform(static_raw, norm_params["static_z_score"])

	t0 = df[MODEL_MODULE.TEMP_T0_COLS].to_numpy(dtype=np.float32)
	t1 = df[MODEL_MODULE.TEMP_T1_COLS].to_numpy(dtype=np.float32)
	temp = np.stack([t0, t1], axis=2)
	temp_scaled = MODEL_MODULE.z_score_transform(temp.reshape(-1, 2), norm_params["temp_z_score"]).reshape(temp.shape)

	seq_len = len(MODEL_MODULE.TEMP_T0_COLS)
	static_seq = np.repeat(static_scaled[:, None, :], seq_len, axis=1)
	x_all = np.concatenate([static_seq, temp_scaled], axis=2).astype(np.float32)
	y_all = df[MODEL_MODULE.TARGET_COL].to_numpy(dtype=np.float32)
	return x_all, y_all


class FullDataset(Dataset):
	"""用于全量推理的数据集封装。"""

	def __init__(self, x, y):
		self.x = torch.tensor(x, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


def load_model(device="cpu"):
	"""按保存参数构建并加载完整预测模型。"""
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"模型参数文件不存在: {MODEL_PATH}")
	if not NORM_PARAMS_PATH.exists():
		raise FileNotFoundError(f"归一化参数文件不存在: {NORM_PARAMS_PATH}")

	norm_params = load_norm_params(NORM_PARAMS_PATH)
	input_dim = len(MODEL_MODULE.ELEMENT_COLS) + 2
	model = MODEL_MODULE.Transformer_Decoder(input_dim=input_dim).to(device)
	MODEL_MODULE.load_model_state(model, MODEL_PATH, device=device)
	model.eval()
	return model, norm_params


def load_norm_params(norm_path):
	"""优先复用模型模块接口；若不存在则按 npz 文件结构兼容加载。"""
	if hasattr(MODEL_MODULE, "load_norm_params"):
		return MODEL_MODULE.load_norm_params(norm_path)

	with np.load(norm_path) as data:
		required = [
			"static_mean",
			"static_std",
			"temp_mean",
			"temp_std",
			"target_mean",
			"target_std",
		]
		missing = [key for key in required if key not in data]
		if missing:
			raise KeyError(f"归一化参数文件缺少字段: {missing}")

		return {
			"static_z_score": {"mean": data["static_mean"], "std": data["static_std"]},
			"temp_z_score": {"mean": data["temp_mean"], "std": data["temp_std"]},
			"target_z_score": {"mean": data["target_mean"], "std": data["target_std"]},
		}


def predict_all(model, x_all, y_all, target_params, device="cpu", batch_size=256):
	"""对全量数据做一次预测，并返回原始量纲下的真实值、预测值与差值。"""
	dataset = FullDataset(x_all, y_all)
	loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

	preds = []
	trues = []

	with torch.no_grad():
		for batch_x, batch_y in loader:
			batch_pred = model(batch_x.to(device)).cpu().numpy().ravel()
			batch_true = batch_y.numpy().ravel()
			preds.append(batch_pred)
			trues.append(batch_true)

	y_pred_norm = np.concatenate(preds)
	y_true_norm = np.concatenate(trues)
	y_pred = MODEL_MODULE.z_score_inverse_transform(y_pred_norm, target_params)
	y_true = y_true_norm
	diff = y_pred - y_true
	return y_true, y_pred, diff


def append_prediction_columns(df, y_true, y_pred, diff):
	"""把预测结果拼接回原始数据变量。"""
	df["pred_ts"] = y_pred
	df["diff"] = diff
	df["abs_diff"] = np.abs(diff)
	return df


def archive_and_save_clean_data(source_df, diff, threshold=ANOMALY_DIFF_THRESHOLD, source_path=DATA_PATH, old_path=OLD_DATA_PATH):
	"""将原始数据改名为 old 文件，并把剔除异常值后的数据保存回 new 文件。"""
	if threshold < 0:
		raise ValueError("threshold 必须大于等于 0。")
	if not source_path.exists():
		raise FileNotFoundError(f"数据文件不存在: {source_path}")

	if old_path.exists():
		old_path.unlink()
	source_path.replace(old_path)

	keep_mask = np.abs(diff) <= threshold
	clean_df = source_df.loc[keep_mask].copy()
	removed_count = int((~keep_mask).sum())
	clean_df.to_csv(source_path, index=False, encoding="utf-8-sig")
	print(f"异常值删除数量: {removed_count}")
	print(f"原始数据已改名为: {old_path}")
	print(f"处理后数据已保存为: {source_path}")
	return clean_df, removed_count


def analyze_diff(diff):
	"""计算差值统计量。"""
	mean_diff = float(np.mean(diff))
	max_abs_diff = float(np.max(np.abs(diff)))
	var_diff = float(np.var(diff))
	return mean_diff, max_abs_diff, var_diff


def plot_diff_distribution(diff):
	"""绘制差值分布直方图。"""
	fig, ax = plt.subplots(figsize=(10, 6))
	bin_width = 5
	diff_min = np.floor(np.min(diff) / bin_width) * bin_width
	diff_max = np.ceil(np.max(diff) / bin_width) * bin_width
	bins = np.arange(diff_min, diff_max + bin_width, bin_width)
	ax.hist(diff, bins=bins, edgecolor="black", alpha=0.85, rwidth=0.95)
	ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5)
	ax.set_title("Prediction Error Distribution")
	ax.set_xlabel("Difference (pred - true)")
	ax.set_ylabel("Count")
	ax.set_xticks(np.arange(diff_min, diff_max + bin_width, bin_width))
	ax.grid(True, linestyle=":", alpha=0.4)
	fig.tight_layout()
	plt.show()
	plt.close(fig)


def main():
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"使用设备: {device}")

	df = load_source_data(DATA_PATH)
	original_df = df.copy()
	model, norm_params = load_model(device=device)
	x_all, y_all = build_full_inputs(df, norm_params)
	y_true, y_pred, diff = predict_all(
		model=model,
		x_all=x_all,
		y_all=y_all,
		target_params=norm_params["target_z_score"],
		device=device,
	)

	df = append_prediction_columns(df, y_true, y_pred, diff)
	mean_diff, max_abs_diff, var_diff = analyze_diff(diff)

	print("-" * 60)
	print(f"样本数: {len(df)}")
	print(f"平均差值: {mean_diff:.6f}")
	print(f"最大差值: {max_abs_diff:.6f}")
	print(f"差值方差: {var_diff:.6f}")

	archive_and_save_clean_data(original_df, diff, threshold=ANOMALY_DIFF_THRESHOLD)

	plot_diff_distribution(diff)


if __name__ == "__main__":
	main()
