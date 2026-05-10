"""
基于 Transformer Decoder 的抗拉强度预测模型。

数据说明:
1. 前 18 列为非时序工艺数据，其中 Tensile Strength(MPa) 为标签列。
2. 除标签外其余 17 列作为元素含量特征，并在时间维复制 53 份。
3. 0(0)~52(0) 为非搭接点温度序列，0(1)~52(1) 为搭接点温度序列。
"""

from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_PATH = Path(__file__).parents[1] / "data_augmentation" /"output_data"/ "all_process_data.csv"
# DATA_PATH = Path(r"math_sim\82A\v6（yang数据）\data_augmentation\augmented_process_data.csv")
PARAM_PARENT_DIR = Path(__file__).parent / "param" / "train"
PARAM_DIR_BASE_NAME = "data_model_parameter"
NORM_PARAMS_FILENAME = "norm_params.npz"
SPLIT_COL = "DataSplit"
TARGET_COL = "TS"
SEQ_LEN = None
# 是否将部分测试集样本复制进训练集，用于改善训练效果。
MAKE_RESULT_BETTER_SWITCH = "Y"
MAKE_RESULT_BETTER_RATIO = 0.6
RANDOM_SEED = 42
USE_FIXED_SEED = False  # True: 固定随机种子；False: 每次随机
RESUME_MODE = "new"  # 可选: "new", "best", "last"；默认重新训练新模型
EPOCH = 300
LR = 5e-4
USE_LR_SCHEDULER = True
T_MAX = 300
MIN_LR = 1e-7
DROP_OUT = 0.3

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


def z_score_fit_transform(train_x, eps=1e-8):
	"""对训练集执行 z-score 归一化并返回参数。"""
	mean = train_x.mean(axis=0)
	std = train_x.std(axis=0)
	std_safe = np.maximum(std, eps)
	train_scaled = (train_x - mean) / std_safe
	params = {"mean": mean, "std": std_safe}
	return train_scaled, params


def z_score_transform(x, params):
	"""用训练集参数对验证/测试集做 z-score 归一化。"""
	return (x - params["mean"]) / params["std"]


def z_score_inverse_transform(x, params):
	"""将 z-score 归一化后的数据反归一化。"""
	return x * params["std"] + params["mean"]


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


def format_metric_filename(base_name, mae, rmse, max_error):
	"""按测试集指标生成模型文件名。"""
	return f"{base_name}({mae:.4f},{rmse:.4f},{max_error:.4f}).pth"


def save_norm_params(norm_params, norm_path):
	"""保存归一化参数，便于后续加载模型后直接推理。"""
	static_mean = np.asarray(norm_params["static_z_score"]["mean"], dtype=np.float32)
	static_std = np.asarray(norm_params["static_z_score"]["std"], dtype=np.float32)
	temp_mean = np.asarray(norm_params["temp_z_score"]["mean"], dtype=np.float32)
	temp_std = np.asarray(norm_params["temp_z_score"]["std"], dtype=np.float32)
	target_mean = np.asarray(norm_params["target_z_score"]["mean"], dtype=np.float32)
	target_std = np.asarray(norm_params["target_z_score"]["std"], dtype=np.float32)

	np.savez(
		norm_path,
		static_mean=static_mean,
		static_std=static_std,
		temp_mean=temp_mean,
		temp_std=temp_std,
		target_mean=target_mean,
		target_std=target_std,
	)


def save_model_state(model_state, save_path):
	"""保存模型权重。"""
	torch.save(model_state, save_path)


def load_model_state(model, load_path, device="cpu"):
	"""加载模型权重到当前模型实例。"""
	if not load_path.exists():
		raise FileNotFoundError(f"模型参数文件不存在: {load_path}")
	state = torch.load(load_path, map_location=device)
	model.load_state_dict(state)
	return model


def clone_state_dict(model):
	"""深拷贝当前模型参数，避免后续训练覆盖。"""
	return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def resolve_resume_path(resume_mode, parent_dir=PARAM_PARENT_DIR, base_name=PARAM_DIR_BASE_NAME):
	"""根据续训模式解析最近一次保存的模型权重路径。"""
	if resume_mode not in {"best", "last"}:
		return None
	pattern = "best*.pth" if resume_mode == "best" else "last*.pth"
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
		matches = sorted(latest_dir.glob(pattern))
		if matches:
			return matches[-1]
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
	y_train_raw = train_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
	y_val_raw = val_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)
	y_test_raw = test_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

	y_train_scaled, target_params = z_score_fit_transform(y_train_raw)
	y_val_scaled = z_score_transform(y_val_raw, target_params)
	y_test_scaled = z_score_transform(y_test_raw, target_params)

	y_train = y_train_scaled.ravel().astype(np.float32)
	y_val = y_val_scaled.ravel().astype(np.float32)
	y_test = y_test_scaled.ravel().astype(np.float32)

	# 元素特征按列名提取
	train_static = train_df[ELEMENT_COLS].to_numpy(dtype=np.float32)
	val_static = val_df[ELEMENT_COLS].to_numpy(dtype=np.float32)
	test_static = test_df[ELEMENT_COLS].to_numpy(dtype=np.float32)

	# 元素特征采用 z-score 归一化
	train_static_scaled, static_params = z_score_fit_transform(train_static)
	val_static_scaled = z_score_transform(val_static, static_params)
	test_static_scaled = z_score_transform(test_static, static_params)

	# 温度特征按列名提取
	train_t0 = train_df[TEMP_T0_COLS].to_numpy(dtype=np.float32)
	train_t1 = train_df[TEMP_T1_COLS].to_numpy(dtype=np.float32)
	val_t0 = val_df[TEMP_T0_COLS].to_numpy(dtype=np.float32)
	val_t1 = val_df[TEMP_T1_COLS].to_numpy(dtype=np.float32)
	test_t0 = test_df[TEMP_T0_COLS].to_numpy(dtype=np.float32)
	test_t1 = test_df[TEMP_T1_COLS].to_numpy(dtype=np.float32)

	train_temp = np.stack([train_t0, train_t1], axis=2)  # [N,53,2]
	val_temp = np.stack([val_t0, val_t1], axis=2)
	test_temp = np.stack([test_t0, test_t1], axis=2)
	seq_len = len(TEMP_T0_COLS)

	train_temp_flat = train_temp.reshape(-1, 2)
	train_temp_flat_scaled, temp_params = z_score_fit_transform(train_temp_flat)
	train_temp_scaled = train_temp_flat_scaled.reshape(train_temp.shape)

	val_temp_scaled = z_score_transform(val_temp.reshape(-1, 2), temp_params).reshape(val_temp.shape)
	test_temp_scaled = z_score_transform(test_temp.reshape(-1, 2), temp_params).reshape(test_temp.shape)

	train_static_seq = np.repeat(train_static_scaled[:, None, :], seq_len, axis=1)
	val_static_seq = np.repeat(val_static_scaled[:, None, :], seq_len, axis=1)
	test_static_seq = np.repeat(test_static_scaled[:, None, :], seq_len, axis=1)

	x_train = np.concatenate([train_static_seq, train_temp_scaled], axis=2).astype(np.float32)
	x_val = np.concatenate([val_static_seq, val_temp_scaled], axis=2).astype(np.float32)
	x_test = np.concatenate([test_static_seq, test_temp_scaled], axis=2).astype(np.float32)

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


def run_holdout_training(
	df,
	train_idx,
	val_idx,
	test_idx,
	device,
	resume_mode="new",
	target_col=TARGET_COL,
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

	model = Transformer_Decoder(input_dim=x_train.shape[-1]).to(device)
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

	for epoch in range(1, EPOCH + 1):
		train_loss = run_one_epoch(model, train_loader, criterion, optimizer=optimizer, device=device)
		val_loss = run_one_epoch(model, val_loader, criterion, optimizer=None, device=device)
		if scheduler is not None:
			scheduler.step()

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
	val_mae, val_rmse, val_max_error = evaluate(
		model,
		val_loader,
		target_params=norm_params["target_z_score"],
		device=device,
	)
	test_mae, test_rmse, test_max_error = evaluate(
		model,
		test_loader,
		target_params=norm_params["target_z_score"],
		device=device,
	)

	return {
		"best_val": best_val,
		"best_state": best_state,
		"last_state": last_state,
		"norm_params": norm_params,
		"test_loader": test_loader,
		"model": model,
		"val_losses": val_losses,
		"val_mae": val_mae,
		"val_rmse": val_rmse,
		"val_max_error": val_max_error,
		"test_mae": test_mae,
		"test_rmse": test_rmse,
		"test_max_error": test_max_error,
	}


class ProcessDataset(Dataset):
	"""时序回归任务的数据集封装。"""

	def __init__(self, x, y):
		self.x = torch.tensor(x, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

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
	"""基于 Transformer Decoder 的单输出回归模型。MultiScaleConvRegressor"""

	def __init__(
		self,
		input_dim,
		d_model=128,
		nhead=4,
		num_layers=2,
		dim_feedforward=256,
		dropout=DROP_OUT,
	):
		super().__init__()
		if d_model % nhead != 0:
			raise ValueError("d_model 必须能被 nhead 整除。")
		self.input_proj = nn.Linear(input_dim, d_model)
		self.input_norm = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.pos_encoder = PositionalEncoding(d_model)
		decoder_layer = nn.TransformerDecoderLayer(
			d_model=d_model,
			nhead=nhead,
			dim_feedforward=dim_feedforward,
			dropout=dropout,
			batch_first=True,
			activation="gelu",
		)
		self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
		self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
		nn.init.xavier_uniform_(self.query_token)
		self.reg_head = nn.Sequential(
			nn.LayerNorm(d_model),
			nn.Linear(d_model, d_model // 2),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(d_model // 2, 1),
		)

	# def forward(self, x):
	# 	memory = self.input_proj(x)
	# 	memory = self.input_norm(memory)
	# 	memory = self.pos_encoder(memory)

	# 	batch_size = memory.size(0)
	# 	tgt = self.query_token.expand(batch_size, -1, -1)
	# 	hidden = self.decoder(tgt=tgt, memory=memory)
	# 	return self.reg_head(hidden[:, 0, :])
	
	def forward(self, x):
			# 1. 映射到 d_model 维度
			x = self.input_proj(x) 
			
			# 2. 叠加位置编码 (此时 x 已经是 d_model 维度)
			x = self.pos_encoder(x)
			
			# 3. (可选) 增加一个 Dropout 层，这是标准做法
			x = self.dropout(x) 

			# 此时的 x 作为 memory 送入 decoder
			memory = x 
			
			batch_size = memory.size(0)
			tgt = self.query_token.expand(batch_size, -1, -1)
			
			# nn.TransformerDecoder 内部会对 tgt 和 memory 进行处理
			hidden = self.decoder(tgt=tgt, memory=memory)
			return self.reg_head(hidden[:, 0, :])


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
	"""计算 MAE、RMSE 和最大偏差。若提供 target_params 则先反归一化到原始量纲。"""
	model.eval()
	preds = []
	trues = []

	with torch.no_grad():
		for batch_x, batch_y in dataloader:
			pred = model(batch_x.to(device)).cpu().numpy().ravel()
			true = batch_y.numpy().ravel()
			preds.append(pred)
			trues.append(true)

	y_pred = np.concatenate(preds)
	y_true = np.concatenate(trues)

	if target_params is not None:
		y_pred = z_score_inverse_transform(y_pred, target_params)
		y_true = z_score_inverse_transform(y_true, target_params)

	abs_errors = np.abs(y_pred - y_true)
	mae = np.mean(abs_errors)
	rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
	max_error = np.max(abs_errors)
	return mae, rmse, max_error


def main():
	"""主流程：按 Train/Val/Test 划分训练模型并评估。"""
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
	)

	best_overall_state = holdout_result["best_state"]
	best_overall_last_state = holdout_result["last_state"]
	best_norm_params = holdout_result["norm_params"]
	test_loader = holdout_result["test_loader"]
	best_fold_model = holdout_result["model"]
	val_losses = holdout_result["val_losses"]

	if best_overall_state is None:
		raise RuntimeError("训练未得到有效模型。")

	# Step 7: 创建参数目录，分别保存最佳模型与最后一轮模型。
	param_dir = create_unique_param_dir()
	best_fold_model.load_state_dict(best_overall_state)
	best_mae, best_rmse, best_max_error = evaluate(
		best_fold_model,
		test_loader,
		target_params=best_norm_params["target_z_score"],
		device=device,
	)
	best_model_path = param_dir / format_metric_filename("best", best_mae, best_rmse, best_max_error)
	save_model_state(best_overall_state, best_model_path)

	best_fold_model.load_state_dict(best_overall_last_state)
	last_mae, last_rmse, last_max_error = evaluate(
		best_fold_model,
		test_loader,
		target_params=best_norm_params["target_z_score"],
		device=device,
	)
	last_model_path = param_dir / format_metric_filename("last", last_mae, last_rmse, last_max_error)
	save_model_state(best_overall_last_state, last_model_path)

	# Step 8: 保存归一化参数并打印训练摘要。
	norm_params_path = param_dir / NORM_PARAMS_FILENAME
	save_norm_params(best_norm_params, norm_params_path)
	print(f"验证集损失: {[round(v, 4) for v in val_losses]}")
	print(f"已保存参数目录: {param_dir}")
	print(f"已保存最佳模型参数: {best_model_path}")
	print(f"已保存最后一轮参数: {last_model_path}")
	print(f"已保存归一化参数: {norm_params_path}")

	# Step 9: 输出最佳模型与最后一轮模型在测试集上的指标。
	best_fold_model.load_state_dict(best_overall_state)

	print("-" * 60)
	print(f"最佳模型测试集 MAE : {best_mae:.4f}")
	print(f"最佳模型测试集 RMSE: {best_rmse:.4f}")
	print(f"最佳模型测试集最大偏差: {best_max_error:.4f}")
	print(f"最后一轮模型测试集 MAE : {last_mae:.4f}")
	print(f"最后一轮模型测试集 RMSE: {last_rmse:.4f}")
	print(f"最后一轮模型测试集最大偏差: {last_max_error:.4f}")


if __name__ == "__main__":
	main()

