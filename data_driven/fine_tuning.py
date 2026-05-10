"""仅微调预测头的脚本。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re

import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_MODULE_PATH = SCRIPT_DIR / "data_driven_model.py"

DATA_PATH = SCRIPT_DIR.parent / "sim_T" / "output_data" / "process_data_new.csv"
PARAM_PARENT_DIR = SCRIPT_DIR / "param" / "train"
PARAM_DIR_BASE_NAME = "data_model_parameter"
OUTPUT_DIR = SCRIPT_DIR / "param" / "fine_tuning"

RANDOM_SEED = 42
USE_FIXED_SEED = True
EPOCHS = 100
LR = 1e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 64


def load_model_module():
    """动态加载同目录下的 data_driven_model.py，复用其模型与数据处理逻辑。"""
    spec = spec_from_file_location("data_driven_model", MODEL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模型模块: {MODEL_MODULE_PATH}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_data_path():
    """解析并校验微调数据路径。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"数据集文件不存在: {DATA_PATH}")
    return DATA_PATH


def resolve_latest_param_dir(parent_dir, base_name):
    """按 data_model_parameter、data_model_parameter(n) 规则定位最新参数目录。"""
    pattern = re.compile(rf"^{re.escape(base_name)}(?:\((\d+)\))?$")
    candidates = []

    if not parent_dir.exists():
        raise FileNotFoundError(f"模型参数父目录不存在: {parent_dir}")

    for child in parent_dir.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            index = int(match.group(1) or 0)
            candidates.append((index, child))

    if not candidates:
        raise FileNotFoundError("未找到历史模型参数目录 data_model_parameter*。")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def resolve_pretrained_paths():
    """从最新参数目录中解析预训练模型(best*.pth)与归一化参数文件。"""
    param_dir = resolve_latest_param_dir(PARAM_PARENT_DIR, PARAM_DIR_BASE_NAME)

    best_files = sorted(param_dir.glob("best*.pth"))
    if not best_files:
        raise FileNotFoundError(f"目录中未找到 best*.pth: {param_dir}")

    model_path = best_files[-1]
    norm_params_path = param_dir / "norm_params.npz"
    if not norm_params_path.exists():
        raise FileNotFoundError(f"归一化参数文件不存在: {norm_params_path}")

    return param_dir, model_path, norm_params_path


def freeze_except_reg_head(model):
    """冻结除 reg_head 外的全部参数，仅允许预测头参与微调。"""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("reg_head")

    # 显式冻结 decoder，避免误改。
    for param in model.decoder.parameters():
        param.requires_grad = False


def save_artifacts(model_module, model, norm_params, model_state, test_mae, test_rmse, test_max_error):
    """将微调后的模型、归一化参数和网络结构保存到 fine_tuning 目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_filename = (
        f"fine_tuned_head_only({test_mae:.4f},{test_rmse:.4f},{test_max_error:.4f}).pth"
    )
    model_path = OUTPUT_DIR / model_filename
    norm_path = OUTPUT_DIR / "norm_params.npz"
    arch_path = OUTPUT_DIR / "model_structure.txt"

    model_module.save_model_state(model_state, model_path)
    model_module.save_norm_params(norm_params, norm_path)
    arch_path.write_text(str(model), encoding="utf-8")

    return model_path, norm_path, arch_path


def main():
    """主流程：加载预训练参数 -> 划分数据 -> 仅微调预测头 -> 测试与保存。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_module = load_model_module()

    if USE_FIXED_SEED:
        model_module.set_seed(RANDOM_SEED)

    data_path = resolve_data_path()
    param_dir, pretrained_model_path, _ = resolve_pretrained_paths()

    df = model_module.load_dataset(csv_path=data_path)
    # 复用原脚本划分逻辑：若存在 DataSplit 列则复用，否则自动划分并回写。
    df, train_idx, val_idx, test_idx = model_module.assign_or_load_split(
        df,
        csv_path=data_path,
        seed=RANDOM_SEED if USE_FIXED_SEED else None,
        split_col=model_module.SPLIT_COL,
    )
    model_module.validate_required_columns(df, target_col=model_module.TARGET_COL)

    x_train, y_train, x_val, y_val, x_test, y_test, norm_params = model_module.preprocess_data(
        df,
        train_idx,
        val_idx,
        test_idx,
        target_col=model_module.TARGET_COL,
    )

    # 构建 Train/Val/Test 数据加载器。
    train_loader = DataLoader(model_module.ProcessDataset(x_train, y_train), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(model_module.ProcessDataset(x_val, y_val), batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(model_module.ProcessDataset(x_test, y_test), batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # 初始化模型并加载历史最佳参数作为微调起点。
    model = model_module.Transformer_Decoder(input_dim=x_train.shape[-1]).to(device)
    model_module.load_model_state(model, pretrained_model_path, device=device)

    freeze_except_reg_head(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("未找到可训练参数，请检查预测头命名是否为 reg_head。")

    criterion = model_module.LpLoss(p=3)
    optimizer = torch.optim.Adam(trainable_params, lr=LR)

    best_val_loss = float("inf")
    best_state = model_module.clone_state_dict(model)

    for epoch in range(1, EPOCHS + 1):
        # 每轮先训练后验证，并保留验证集最优权重。
        train_loss = model_module.run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss = model_module.run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model_module.clone_state_dict(model)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

    model.load_state_dict(best_state)

    test_mae, test_rmse, test_max_error = model_module.evaluate(
        model,
        test_loader,
        target_params=norm_params["target_z_score"],
        device=device,
    )

    saved_model_path, saved_norm_path, saved_arch_path = save_artifacts(
        model_module=model_module,
        model=model,
        norm_params=norm_params,
        model_state=best_state,
        test_mae=test_mae,
        test_rmse=test_rmse,
        test_max_error=test_max_error,
    )

    print("-" * 60)
    print(f"预训练参数目录: {param_dir}")
    print(f"已加载预训练模型: {pretrained_model_path}")
    print(f"微调轮数: {EPOCHS}")
    print(f"学习率: {LR}")
    print(f"测试集 MAE: {test_mae:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"测试集最大偏差: {test_max_error:.4f}")
    print(f"已保存微调模型: {saved_model_path}")
    print(f"已保存归一化参数: {saved_norm_path}")
    print(f"已保存网络结构: {saved_arch_path}")


if __name__ == "__main__":
    main()
