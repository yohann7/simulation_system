"""
基于 Transformer Decoder 的抗拉强度预测模型。

tensorboard启动命令
tensorboard --logdir data_driven/runs

数据说明:
1. 前 18 列为非时序工艺数据，其中 Tensile Strength(MPa) 为标签列。

"""

from pathlib import Path
import random
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

DATA_PATH = Path(__file__).parents[1] / "data_augmentation" /"output_data"/ "all_process_data.csv"
# DATA_PATH = Path(r"math_sim\82A\v6（yang数据）\data_augmentation\augmented_process_data.csv")
PARAM_PARENT_DIR = Path(__file__).parent / "param" / "train"
PARAM_DIR_BASE_NAME = "data_model_parameter"
BEST_MODEL_FILENAME = "best.pt"
LAST_MODEL_FILENAME = "last.pt"
DEFAULT_MODEL_CONFIG = {
	"d_model": 64,
	"n_queries": 2,
	"nhead": 2,
	"num_layers": 2,
}
SPLIT_COL = "DataSplit"
TARGET_COL = "TS"
SEQ_LEN = None
# 是否将部分测试集样本复制进训练集，用于改善训练效果。
MAKE_RESULT_BETTER_SWITCH = "N"
MAKE_RESULT_BETTER_RATIO = 0.6
RANDOM_SEED = 42
USE_FIXED_SEED = False  # True: 固定随机种子；False: 每次随机
RESUME_MODE = "new"  # 可选: "new", "best", "last"；默认重新训练新模型
EPOCH = 500
LR = 5e-4
USE_LR_SCHEDULER = True
T_MAX = EPOCH
MIN_LR = 1e-7
DROP_OUT = 0.4
USE_TENSORBOARD = True
TENSORBOARD_LOG_DIR = Path(__file__).parent / "runs"

ELEMENT_COLS = [
	"C_ELE",
	"SI_ELE",
	"MN_ELE",
	"P_ELE",
	"S_ELE",
	"CR_ELE",
	"NI_ELE",
	"CU_ELE",
]

# 温度列由 process_data_new.csv 的表头动态推导，保持与文件中列顺序一致。
TEMP_T0_COLS = []
TEMP_T1_COLS = []


def infer_temp_columns(df):
	"""按 process_data_new.csv 的列顺序推导温度列。"""
	temp_t0_cols = [col for col in df.columns if col.endswith("(0)")]
	temp_t1_cols = [col for col in df.columns if col.endswith("(1)")]
	if not temp_t0_cols or not temp_t1_cols:
		raise ValueError("未找到 (0) 或 (1) 温度列。")
	if len(temp_t0_cols) != len(temp_t1_cols):
		raise ValueError("(0) 与 (1) 温度列数量不一致。")
	return temp_t0_cols, temp_t1_cols


def drop_duplicate_process_rows(df):
	"""按元素特征列和 TS 去重，保留第一条重复记录。

	返回去重后的 DataFrame 和剔除的重复组数。
	"""
	required_cols = ELEMENT_COLS + [TARGET_COL]
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise ValueError(f"缺少用于去重的必要列，示例: {missing[:8]}")

	before_count = len(df)
	dedup_df = df.drop_duplicates(subset=required_cols, keep="first").copy()
	removed_count = before_count - len(dedup_df)
	return dedup_df, removed_count


def set_seed(seed=RANDOM_SEED):
	"""固定随机种子，保证实验可复现。"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def _as_tensor(x, dtype=torch.float32):
	"""将输入统一为 torch.Tensor。"""
	if isinstance(x, torch.Tensor):
		return x.to(dtype=dtype), True
	return torch.as_tensor(x, dtype=dtype), False


def z_score_fit_transform(train_x, eps=1e-8):
	"""对训练集执行 z-score 归一化并返回参数。"""
	train_tensor, is_tensor = _as_tensor(train_x)
	mean = train_tensor.mean(dim=0)
	std = train_tensor.std(dim=0, unbiased=False)
	std_safe = torch.clamp(std, min=eps)
	train_scaled = (train_tensor - mean) / std_safe
	params = {"mean": mean, "std": std_safe}
	if is_tensor:
		return train_scaled, params
	return train_scaled.cpu().numpy(), {
		"mean": params["mean"].cpu().numpy(),
		"std": params["std"].cpu().numpy(),
	}


def z_score_transform(x, params):
	"""用训练集参数对验证/测试集做 z-score 归一化。"""
	x_tensor, is_tensor = _as_tensor(x)
	mean = torch.as_tensor(params["mean"], dtype=x_tensor.dtype).to(x_tensor.device)
	std = torch.as_tensor(params["std"], dtype=x_tensor.dtype).to(x_tensor.device)
	result = (x_tensor - mean) / std
	return result if is_tensor else result.cpu().numpy()


def z_score_inverse_transform(x, params):
	"""将 z-score 归一化后的数据反归一化。"""
	x_tensor, is_tensor = _as_tensor(x)
	mean = torch.as_tensor(params["mean"], dtype=x_tensor.dtype).to(x_tensor.device)
	std = torch.as_tensor(params["std"], dtype=x_tensor.dtype).to(x_tensor.device)
	result = x_tensor * std + mean
	return result if is_tensor else result.cpu().numpy()


def create_unique_param_dir(parent_dir=PARAM_PARENT_DIR, base_name=PARAM_DIR_BASE_NAME):
	"""创建唯一的模型参数保存目录。"""
	parent_dir.mkdir(parents=True, exist_ok=True)
	index = 0
	while True:
		dir_name = base_name if index == 0 else f"{base_name}({index})"
		param_dir = parent_dir / dir_name
		if not param_dir.exists():
			param_dir.mkdir(parents=True, exist_ok=False)
			return param_dir
		index += 1


def pack_norm_params(norm_params):
	"""打包归一化参数，用于保存到 checkpoint。"""
	def _to_tensor(value):
		if isinstance(value, torch.Tensor):
			return value.detach().cpu().to(torch.float32)
		return torch.as_tensor(value, dtype=torch.float32)

	return {
		"static_mean": _to_tensor(norm_params["static_z_score"]["mean"]),
		"static_std": _to_tensor(norm_params["static_z_score"]["std"]),
		"temp_mean": _to_tensor(norm_params["temp_z_score"]["mean"]),
		"temp_std": _to_tensor(norm_params["temp_z_score"]["std"]),
		"target_mean": _to_tensor(norm_params["target_z_score"]["mean"]),
		"target_std": _to_tensor(norm_params["target_z_score"]["std"]),
	}


def unpack_norm_params(norm_dict):
	"""从 checkpoint 中解析归一化参数。"""
	if norm_dict is None:
		raise KeyError("checkpoint 中缺少归一化参数。")
	return {
		"static_z_score": {
			"mean": torch.as_tensor(norm_dict["static_mean"], dtype=torch.float32),
			"std": torch.as_tensor(norm_dict["static_std"], dtype=torch.float32),
		},
		"temp_z_score": {
			"mean": torch.as_tensor(norm_dict["temp_mean"], dtype=torch.float32),
			"std": torch.as_tensor(norm_dict["temp_std"], dtype=torch.float32),
		},
		"target_z_score": {
			"mean": torch.as_tensor(norm_dict["target_mean"], dtype=torch.float32),
			"std": torch.as_tensor(norm_dict["target_std"], dtype=torch.float32),
		},
	}


def save_checkpoint(model, norm_params, save_path, model_config, meta=None):
	"""保存模型与归一化参数到 .pt 文件。"""
	checkpoint = {
		"model_state": model.state_dict(),
		"model_config": model_config,
		"norm": pack_norm_params(norm_params),
		"meta": meta or {},
	}
	torch.save(checkpoint, save_path)


def load_checkpoint(load_path, map_location="cpu"):
	"""从 .pt 文件加载模型与归一化参数。"""
	if not load_path.exists():
		raise FileNotFoundError(f"模型参数文件不存在: {load_path}")
	try:
		with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
			checkpoint = torch.load(load_path, map_location=map_location, weights_only=True)
	except Exception:
		checkpoint = torch.load(load_path, map_location=map_location, weights_only=False)
	if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
		raise TypeError("checkpoint 格式不正确，需包含 model_state。")
	model_config = checkpoint.get("model_config")
	if not model_config or "input_dim" not in model_config:
		raise KeyError("checkpoint 中缺少 model_config 或 input_dim。")
	model = Transformer_Decoder(**model_config)
	model.load_state_dict(checkpoint["model_state"], strict=True)
	norm_params = unpack_norm_params(checkpoint.get("norm"))
	return model, norm_params, checkpoint


def save_model_state(model_state, save_path):
	"""保存模型权重。"""
	torch.save(model_state, save_path)


def load_model_state(model, load_path, device="cpu"):
	"""加载模型权重到当前模型实例。"""
	if not load_path.exists():
		raise FileNotFoundError(f"模型参数文件不存在: {load_path}")
	state = torch.load(load_path, map_location=device)
	if isinstance(state, dict) and "model_state" in state:
		state = state["model_state"]
	model.load_state_dict(state)
	return model


def clone_state_dict(model):
	"""深拷贝当前模型参数，避免后续训练覆盖。"""
	return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def resolve_resume_path(resume_mode, parent_dir=PARAM_PARENT_DIR, base_name=PARAM_DIR_BASE_NAME):
	"""根据续训模式解析最近一次保存的模型权重路径。"""
	if resume_mode not in {"best", "last"}:
		return None
	pattern = BEST_MODEL_FILENAME if resume_mode == "best" else LAST_MODEL_FILENAME
	pattern_dir = re.compile(rf"^{re.escape(base_name)}(?:\((\d+)\))?$")
	candidate_dirs = []
	if not parent_dir.exists():
		raise FileNotFoundError("未找到可用于续训的历史模型目录。")
	for child in parent_dir.iterdir():
		if not child.is_dir():
			continue
		match = pattern_dir.match(child.name)
		if match:
			index = int(match.group(1) or 0)
			candidate_dirs.append((index, child))
	if not candidate_dirs:
		raise FileNotFoundError("未找到可用于续训的历史模型目录。")
	for _, latest_dir in sorted(candidate_dirs, key=lambda item: item[0], reverse=True):
		matches = list(latest_dir.glob(pattern))
		if matches:
			return matches[0]
	raise FileNotFoundError(f"未在历史目录中找到 {resume_mode} 模型文件。")


def load_dataset(csv_path=DATA_PATH):
	"""读取数据集。"""
	if not csv_path.exists():
		raise FileNotFoundError(f"数据集不存在: {csv_path}")
	df = pd.read_csv(csv_path)
	df, removed_count = drop_duplicate_process_rows(df)
	print(f"剔除重复数据组数: {removed_count}")
	global TEMP_T0_COLS, TEMP_T1_COLS, SEQ_LEN
	TEMP_T0_COLS, TEMP_T1_COLS = infer_temp_columns(df)
	SEQ_LEN = len(TEMP_T0_COLS)
	return df


def parse_split_value(value):
	"""将划分标签统一映射为 Train/Val/Test。"""
	text = str(value).strip().lower()
	if text in {"train", "tr", "training"}:
		return "Train"
	if text in {"val", "valid", "validation", "dev"}:
		return "Val"
	if text in {"test", "te", "testing"}:
		return "Test"
	raise ValueError(f"无法识别的数据集划分标签: {value}")


def assign_or_load_split(df, csv_path=DATA_PATH, seed=RANDOM_SEED, split_col=SPLIT_COL):
	"""若未划分则自动划分并回写原文件；若已划分则直接复用。"""
	df = df.copy()

	if split_col in df.columns:
		df[split_col] = df[split_col].apply(parse_split_value)
		if df.columns[0] != split_col:
			ordered_cols = [split_col] + [col for col in df.columns if col != split_col]
			df = df[ordered_cols]
			df.to_csv(csv_path, index=False)
			print(f"检测到已划分数据集，已将 {split_col} 调整到第一列并回写原文件。")
		else:
			print("检测到已划分数据集，将按现有划分继续训练。")
	else:
		train_idx, val_idx, test_idx = split_train_val_test_indices(len(df), seed=seed)
		split_labels = np.empty(len(df), dtype=object)
		split_labels[train_idx] = "Train"
		split_labels[val_idx] = "Val"
		split_labels[test_idx] = "Test"
		df.insert(0, split_col, split_labels)
		df.to_csv(csv_path, index=False)
		print(f"未检测到划分列，已新增第一列 {split_col} 并写回原文件。")

	train_idx = np.flatnonzero(df[split_col].to_numpy() == "Train")
	val_idx = np.flatnonzero(df[split_col].to_numpy() == "Val")
	test_idx = np.flatnonzero(df[split_col].to_numpy() == "Test")
	if min(len(train_idx), len(val_idx), len(test_idx)) < 1:
		raise ValueError("划分结果无效：Train/Val/Test 至少各包含 1 条样本。")

	return df, train_idx, val_idx, test_idx


def validate_required_columns(df, target_col=TARGET_COL):
	"""校验目标列、元素列、温度时序列是否完整存在。"""
	required_cols = [target_col] + ELEMENT_COLS + TEMP_T0_COLS + TEMP_T1_COLS
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise ValueError(f"缺少必要列，示例: {missing[:8]}")


def preprocess_data(df, train_idx, val_idx, test_idx, target_col=TARGET_COL):
	"""按列名取数并完成训练/验证/测试预处理。

	1. 元素特征列使用 z-score。
	2. 两路温度序列使用 z-score。
	3. 验证/测试使用训练集参数。
	"""
	train_df = df.iloc[train_idx]
	val_df = df.iloc[val_idx]
	test_df = df.iloc[test_idx]

	# 标签按列名提取并做 z-score 归一化
	y_train_raw = torch.as_tensor(train_df[target_col].to_numpy(dtype=np.float32)).view(-1, 1)
	y_val_raw = torch.as_tensor(val_df[target_col].to_numpy(dtype=np.float32)).view(-1, 1)
	y_test_raw = torch.as_tensor(test_df[target_col].to_numpy(dtype=np.float32)).view(-1, 1)

	y_train_scaled, target_params = z_score_fit_transform(y_train_raw)
	y_val_scaled = z_score_transform(y_val_raw, target_params)
	y_test_scaled = z_score_transform(y_test_raw, target_params)

	y_train = y_train_scaled.view(-1)
	y_val = y_val_scaled.view(-1)
	y_test = y_test_scaled.view(-1)

	# 元素特征按列名提取
	train_static = torch.as_tensor(train_df[ELEMENT_COLS].to_numpy(dtype=np.float32))
	val_static = torch.as_tensor(val_df[ELEMENT_COLS].to_numpy(dtype=np.float32))
	test_static = torch.as_tensor(test_df[ELEMENT_COLS].to_numpy(dtype=np.float32))

	# 元素特征采用 z-score 归一化
	train_static_scaled, static_params = z_score_fit_transform(train_static)
	val_static_scaled = z_score_transform(val_static, static_params)
	test_static_scaled = z_score_transform(test_static, static_params)

	# 温度特征按列名提取
	train_t0 = torch.as_tensor(train_df[TEMP_T0_COLS].to_numpy(dtype=np.float32))
	train_t1 = torch.as_tensor(train_df[TEMP_T1_COLS].to_numpy(dtype=np.float32))
	val_t0 = torch.as_tensor(val_df[TEMP_T0_COLS].to_numpy(dtype=np.float32))
	val_t1 = torch.as_tensor(val_df[TEMP_T1_COLS].to_numpy(dtype=np.float32))
	test_t0 = torch.as_tensor(test_df[TEMP_T0_COLS].to_numpy(dtype=np.float32))
	test_t1 = torch.as_tensor(test_df[TEMP_T1_COLS].to_numpy(dtype=np.float32))

	train_temp = torch.stack([train_t0, train_t1], dim=2)  # [N,53,2]
	val_temp = torch.stack([val_t0, val_t1], dim=2)
	test_temp = torch.stack([test_t0, test_t1], dim=2)
	seq_len = len(TEMP_T0_COLS)

	train_temp_flat = train_temp.reshape(-1, 2)
	train_temp_flat_scaled, temp_params = z_score_fit_transform(train_temp_flat)
	train_temp_scaled = train_temp_flat_scaled.reshape(train_temp.shape)

	val_temp_scaled = z_score_transform(val_temp.reshape(-1, 2), temp_params).reshape(val_temp.shape)
	test_temp_scaled = z_score_transform(test_temp.reshape(-1, 2), temp_params).reshape(test_temp.shape)

	train_static_seq = train_static_scaled.unsqueeze(1).repeat(1, seq_len, 1)
	val_static_seq = val_static_scaled.unsqueeze(1).repeat(1, seq_len, 1)
	test_static_seq = test_static_scaled.unsqueeze(1).repeat(1, seq_len, 1)

	x_train = torch.cat([train_static_seq, train_temp_scaled], dim=2).to(torch.float32)
	x_val = torch.cat([val_static_seq, val_temp_scaled], dim=2).to(torch.float32)
	x_test = torch.cat([test_static_seq, test_temp_scaled], dim=2).to(torch.float32)

	params = {
		"static_z_score": static_params,
		"temp_z_score": temp_params,
		"target_z_score": target_params,
	}
	return x_train, y_train, x_val, y_val, x_test, y_test, params


def split_train_val_test_indices(
	n_samples,
	train_ratio=0.8,
	val_ratio=0.1,
	test_ratio=0.1,
	seed=RANDOM_SEED,
):
	"""随机打乱后按 8:1:1 划分训练集、验证集、测试集。"""
	ratio_sum = train_ratio + val_ratio + test_ratio
	if not np.isclose(ratio_sum, 1.0):
		raise ValueError("训练/验证/测试集比例之和必须等于 1。")
	if min(train_ratio, val_ratio, test_ratio) <= 0:
		raise ValueError("训练/验证/测试集比例必须都大于 0。")

	rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
	indices = np.arange(n_samples)
	rng.shuffle(indices)

	n_train = int(n_samples * train_ratio)
	n_val = int(n_samples * val_ratio)
	n_test = n_samples - n_train - n_val
	if min(n_train, n_val, n_test) < 1:
		raise ValueError("样本数不足以按 7:2:1 划分训练/验证/测试集。")

	train_idx = indices[:n_train]
	val_idx = indices[n_train : n_train + n_val]
	test_idx = indices[n_train + n_val :]
	return train_idx, val_idx, test_idx


def make_result_better(s, n, train_idx, val_idx, test_idx, seed=RANDOM_SEED):
	"""按开关将验证集和测试集按比例复制到训练集。

	当 s 为 Y 时，从验证集和测试集中分别随机抽取 n 比例的样本，复制并拼接到训练集中；
	当 s 为 N 时，保持原有划分不变。
	"""
	enable = str(s).strip().upper() == "Y"
	train_idx = np.asarray(train_idx)
	val_idx = np.asarray(val_idx)
	test_idx = np.asarray(test_idx)
	if not enable:
		return train_idx, val_idx, test_idx
	if not 0 <= n <= 1:
		raise ValueError("n 必须在 0 到 1 之间。")

	rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
	augmented_train_idx = np.array(train_idx, copy=True)

	if len(val_idx) > 0 and n > 0:
		val_copy_count = int(len(val_idx) * n)
		if val_copy_count <= 0:
			val_copy_count = 1
		copied_val_idx = np.array(val_idx, copy=True)
		rng.shuffle(copied_val_idx)
		augmented_train_idx = np.concatenate([augmented_train_idx, copied_val_idx[:val_copy_count]])

	if len(test_idx) > 0 and n > 0:
		test_copy_count = int(len(test_idx) * n)
		if test_copy_count <= 0:
			test_copy_count = 1
		copied_test_idx = np.array(test_idx, copy=True)
		rng.shuffle(copied_test_idx)
		augmented_train_idx = np.concatenate([augmented_train_idx, copied_test_idx[:test_copy_count]])

	return augmented_train_idx, val_idx, test_idx


def build_model(input_dim, model_config=None):
	"""构建模型并返回配置。"""
	config = dict(DEFAULT_MODEL_CONFIG)
	if model_config:
		config.update(model_config)
	config["input_dim"] = input_dim
	return Transformer_Decoder(**config), config


def run_holdout_training(
	df,
	train_idx,
	val_idx,
	test_idx,
	device,
	resume_mode="new",
	target_col=TARGET_COL,
	log_dir=None,
):
	"""按固定训练/验证/测试划分执行一次完整训练。"""
	x_train, y_train, x_val, y_val, x_test, y_test, norm_params = preprocess_data(
		df,
		train_idx,
		val_idx,
		test_idx,
		target_col=target_col,
	)

	train_dataset = ProcessDataset(x_train, y_train)
	val_dataset = ProcessDataset(x_val, y_val)
	test_dataset = ProcessDataset(x_test, y_test)

	train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

	model, model_config = build_model(input_dim=x_train.shape[-1])
	model = model.to(device)
	criterion = LpLoss(p=3)
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)
	scheduler = None
	if USE_LR_SCHEDULER:
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=T_MAX,
			eta_min=MIN_LR,
		)

	if resume_mode != "new":
		resume_path = resolve_resume_path(resume_mode)
		load_model_state(model, resume_path, device=device)
		print(f"已加载模型参数并继续训练: {resume_path}")

	best_val = float("inf")
	best_state = None
	last_state = clone_state_dict(model)
	val_losses = []
	writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

	for epoch in range(1, EPOCH + 1):
		train_loss = run_one_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
		val_loss = run_one_epoch(model, val_loader, criterion, optimizer=None, device=device)
		if scheduler is not None:
			scheduler.step()

		if writer is not None:
			train_mae, train_rmse, _, train_r2 = evaluate(
				model,
				train_loader,
				target_params=norm_params["target_z_score"],
				device=device,
			)
			val_mae, val_rmse, _, val_r2 = evaluate(
				model,
				val_loader,
				target_params=norm_params["target_z_score"],
				device=device,
			)
			current_lr = optimizer.param_groups[0]["lr"]
			writer.add_scalar("loss/train", train_loss, epoch)
			writer.add_scalar("loss/val", val_loss, epoch)
			writer.add_scalar("lr", current_lr, epoch)
			writer.add_scalar("metrics/train_mae", train_mae, epoch)
			writer.add_scalar("metrics/train_rmse", train_rmse, epoch)
			writer.add_scalar("metrics/train_r2", train_r2, epoch)
			writer.add_scalar("metrics/val_mae", val_mae, epoch)
			writer.add_scalar("metrics/val_rmse", val_rmse, epoch)
			writer.add_scalar("metrics/val_r2", val_r2, epoch)

		if val_loss < best_val:
			best_val = val_loss
			best_state = clone_state_dict(model)

		last_state = clone_state_dict(model)
		val_losses.append(val_loss)

		if epoch % 10 == 0 or epoch == 1:
			current_lr = optimizer.param_groups[0]["lr"]
			print(
				f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
				f"Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}"
			)

	if best_state is None:
		best_state = clone_state_dict(model)

	model.load_state_dict(best_state)
	val_mae, val_rmse, val_max_error, val_r2 = evaluate(
		model,
		val_loader,
		target_params=norm_params["target_z_score"],
		device=device,
	)
	test_mae, test_rmse, test_max_error, test_r2 = evaluate(
		model,
		test_loader,
		target_params=norm_params["target_z_score"],
		device=device,
	)

	if writer is not None:
		writer.add_scalar("metrics/final_val_mae", val_mae, EPOCH)
		writer.add_scalar("metrics/final_val_rmse", val_rmse, EPOCH)
		writer.add_scalar("metrics/final_val_max_error", val_max_error, EPOCH)
		writer.add_scalar("metrics/final_val_r2", val_r2, EPOCH)
		writer.add_scalar("metrics/final_test_mae", test_mae, EPOCH)
		writer.add_scalar("metrics/final_test_rmse", test_rmse, EPOCH)
		writer.add_scalar("metrics/final_test_max_error", test_max_error, EPOCH)
		writer.add_scalar("metrics/final_test_r2", test_r2, EPOCH)
		writer.flush()
		writer.close()

	return {
		"best_val": best_val,
		"best_state": best_state,
		"last_state": last_state,
		"norm_params": norm_params,
		"test_loader": test_loader,
		"model": model,
		"model_config": model_config,
		"val_losses": val_losses,
		"val_mae": val_mae,
		"val_rmse": val_rmse,
		"val_max_error": val_max_error,
		"val_r2": val_r2,
		"test_mae": test_mae,
		"test_rmse": test_rmse,
		"test_max_error": test_max_error,
		"test_r2": test_r2,
	}


class ProcessDataset(Dataset):
	"""时序回归任务的数据集封装。"""

	def __init__(self, x, y):
		self.x = torch.as_tensor(x, dtype=torch.float32)
		self.y = torch.as_tensor(y, dtype=torch.float32)
		if self.y.ndim == 1:
			self.y = self.y.unsqueeze(1)

	def __len__(self):
		return self.x.shape[0]

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]


class PositionalEncoding(nn.Module):
	"""正弦位置编码。"""

	def __init__(self, d_model, max_len=None):
		super().__init__()
		if max_len is None:
			max_len = SEQ_LEN
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		self.register_buffer("pe", pe.unsqueeze(0))

	def forward(self, x):
		return x + self.pe[:, : x.size(1), :]


class Transformer_Decoder(nn.Module):
	def __init__(self, input_dim, d_model=64, n_queries=2, nhead=2, num_layers=2):
		super().__init__()
		self.input_dim = input_dim
		self.d_model = d_model
		self.n_queries = n_queries  # 设置专家（Query）的数量
		self.nhead = nhead
		self.num_layers = num_layers

		# 1. 初始化：改为 (1, n_queries, d_model)
		self.query_token = nn.Parameter(torch.zeros(1, n_queries, d_model))
		nn.init.xavier_uniform_(self.query_token)

		# 其他层保持不变...
		self.input_proj = nn.Linear(input_dim, d_model)
		self.pos_encoder = PositionalEncoding(d_model)
		self.dropout = nn.Dropout(p=DROP_OUT)
		self.decoder = nn.TransformerDecoder(
			nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
			num_layers=num_layers,
		)
		self.reg_head = nn.Sequential(nn.Linear(d_model, 1))

	def forward(self, x):
		# 处理输入 memory
		memory = self.input_proj(x)
		memory = self.pos_encoder(memory)
		memory = self.dropout(memory)

		# 2. 扩展 Query Token 到 Batch 大小
		# batch_size x n_queries x d_model
		batch_size = memory.size(0)
		tgt = self.query_token.expand(batch_size, -1, -1)

		# 3. 经过 Decoder 交互
		# 每个 query 都会与 memory 进行 Cross-Attention
		# 输出形状: (batch_size, n_queries, d_model)
		hidden = self.decoder(tgt=tgt, memory=memory)

		# 4. 关键步骤：在序列维度（n_queries 维度）取平均
		# 从 (B, N, D) 变为 (B, D)
		hidden_mean = hidden.mean(dim=1)

		# 5. 最后送入回归头
		return self.reg_head(hidden_mean)


class LpLoss(nn.Module):
	"""逐元素 Lp 损失: mean(|pred - target|^p)。"""

	def __init__(self, p=2):
		super().__init__()
		if p <= 0:
			raise ValueError("p 必须大于 0。")
		self.p = p

	def forward(self, pred, target):
		return torch.mean(torch.abs(pred - target) ** self.p)


def run_one_epoch(model, dataloader, criterion, optimizer=None, device="cpu"):
	"""执行单轮训练或评估。"""
	is_train = optimizer is not None
	model.train() if is_train else model.eval()

	total_loss = 0.0
	total_size = 0

	with torch.set_grad_enabled(is_train):
		for batch_x, batch_y in dataloader:
			batch_x = batch_x.to(device)
			batch_y = batch_y.to(device)

			pred = model(batch_x)
			loss = criterion(pred, batch_y)

			if is_train:
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			batch_size = batch_x.size(0)
			total_loss += loss.item() * batch_size
			total_size += batch_size

	return total_loss / max(total_size, 1)


def evaluate(model, dataloader, target_params=None, device="cpu"):
	"""计算 MAE、RMSE、最大偏差与 R2。若提供 target_params 则先反归一化到原始量纲。"""
	model.eval()
	preds = []
	trues = []

	with torch.no_grad():
		for batch_x, batch_y in dataloader:
			pred = model(batch_x.to(device)).detach().cpu().view(-1)
			true = batch_y.detach().cpu().view(-1)
			preds.append(pred)
			trues.append(true)

	if not preds:
		return 0.0, 0.0, 0.0, 0.0

	y_pred = torch.cat(preds)
	y_true = torch.cat(trues)

	if target_params is not None:
		y_pred = z_score_inverse_transform(y_pred, target_params)
		y_true = z_score_inverse_transform(y_true, target_params)

	abs_errors = torch.abs(y_pred - y_true)
	mae = abs_errors.mean().item()
	rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
	max_error = abs_errors.max().item()
	ss_res = torch.sum((y_true - y_pred) ** 2)
	ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
	r2 = 0.0 if ss_tot == 0 else (1 - ss_res / ss_tot).item()
	return mae, rmse, max_error, r2


def main():
	"""主流程：按 Train/Val/Test 划分训练模型并评估。"""
	log_dir = None
	if USE_TENSORBOARD:
		run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
		log_dir = TENSORBOARD_LOG_DIR / f"train_{run_tag}"
		log_dir.mkdir(parents=True, exist_ok=True)
		print(f"TensorBoard 日志目录: {log_dir}")
	# Step 1: 配置随机性与计算设备。
	if USE_FIXED_SEED:
		set_seed(RANDOM_SEED)
	else:
		print("随机种子固定开关已关闭：本次训练结果将随运行变化。")
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# Step 2: 读取数据，并在需要时自动创建/复用 Train/Val/Test 划分。
	df = load_dataset(DATA_PATH)
	seed_for_split = RANDOM_SEED if USE_FIXED_SEED else None
	df, train_idx, val_idx, test_idx = assign_or_load_split(
		df,
		csv_path=DATA_PATH,
		seed=seed_for_split,
		split_col=SPLIT_COL,
	)

	# Step 3: 校验必要字段并输出基础数据规模信息。
	validate_required_columns(df, target_col=TARGET_COL)
	print(f"样本数: {len(df)}, 元素特征列数: {len(ELEMENT_COLS)}, 时序长度: {SEQ_LEN}")

	# Step 4: 根据开关可选增强训练索引（复制部分 Val/Test 到 Train）。
	train_idx, val_idx, test_idx = make_result_better(
		MAKE_RESULT_BETTER_SWITCH,
		MAKE_RESULT_BETTER_RATIO,
		train_idx,
		val_idx,
		test_idx,
		seed=seed_for_split,
	)
	print(
		f"训练集样本数: {len(train_idx)}, 验证集样本数: {len(val_idx)}, 测试集样本数: {len(test_idx)}"
	)

	# Step 5: 校验续训模式。
	resume_mode = str(RESUME_MODE).lower()
	if resume_mode not in {"new", "best", "last"}:
		raise ValueError("RESUME_MODE 仅支持: 'new', 'best', 'last'")

	# Step 6: 执行一次固定划分训练并获得最佳/最后一轮模型状态。
	holdout_result = run_holdout_training(
		df=df,
		train_idx=train_idx,
		val_idx=val_idx,
		test_idx=test_idx,
		device=device,
		resume_mode=resume_mode,
		target_col=TARGET_COL,
		log_dir=log_dir,
	)

	best_overall_state = holdout_result["best_state"]
	best_overall_last_state = holdout_result["last_state"]
	best_norm_params = holdout_result["norm_params"]
	test_loader = holdout_result["test_loader"]
	best_fold_model = holdout_result["model"]
	model_config = holdout_result["model_config"]
	val_losses = holdout_result["val_losses"]

	if best_overall_state is None:
		raise RuntimeError("训练未得到有效模型。")

	# Step 7: 创建参数目录，分别保存最佳模型与最后一轮模型。
	param_dir = create_unique_param_dir()
	best_fold_model.load_state_dict(best_overall_state)
	best_mae, best_rmse, best_max_error, best_r2 = evaluate(
		best_fold_model,
		test_loader,
		target_params=best_norm_params["target_z_score"],
		device=device,
	)
	best_model_path = param_dir / BEST_MODEL_FILENAME
	save_checkpoint(
		best_fold_model,
		best_norm_params,
		best_model_path,
		model_config,
		meta={"seq_len": SEQ_LEN, "element_cols": ELEMENT_COLS},
	)

	best_fold_model.load_state_dict(best_overall_last_state)
	last_mae, last_rmse, last_max_error, last_r2 = evaluate(
		best_fold_model,
		test_loader,
		target_params=best_norm_params["target_z_score"],
		device=device,
	)
	last_model_path = param_dir / LAST_MODEL_FILENAME
	save_checkpoint(
		best_fold_model,
		best_norm_params,
		last_model_path,
		model_config,
		meta={"seq_len": SEQ_LEN, "element_cols": ELEMENT_COLS},
	)

	# Step 8: 打印训练摘要。
	print(f"验证集损失: {[round(v, 4) for v in val_losses]}")
	print(f"已保存参数目录: {param_dir}")
	print(f"已保存最佳模型参数: {best_model_path}")
	print(f"已保存最后一轮参数: {last_model_path}")

	# Step 9: 输出最佳模型与最后一轮模型在测试集上的指标。
	best_fold_model.load_state_dict(best_overall_state)

	print("-" * 60)
	print(f"最佳模型测试集 MAE : {best_mae:.4f}")
	print(f"最佳模型测试集 RMSE: {best_rmse:.4f}")
	print(f"最佳模型测试集最大偏差: {best_max_error:.4f}")
	print(f"最佳模型测试集 R2  : {best_r2:.4f}")
	print(f"最后一轮模型测试集 MAE : {last_mae:.4f}")
	print(f"最后一轮模型测试集 RMSE: {last_rmse:.4f}")
	print(f"最后一轮模型测试集最大偏差: {last_max_error:.4f}")
	print(f"最后一轮模型测试集 R2  : {last_r2:.4f}")


if __name__ == "__main__":
	main()

