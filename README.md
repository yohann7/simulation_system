# 82A 帘线钢数字化仿真系统

## 项目简介

本项目实现 SWRH82A 帘线钢轧后斯太尔摩冷却全流程的数字化仿真，涵盖三个核心模块：

| 模块 | 功能 | 核心文件 |
|------|------|---------|
| **sim_T** | 温度场与组织相变机理仿真 | `sim_T/sim_T.py` |
| **data_augmentation** | 仿真数据增强 | `data_augmentation/make_data_big.py` |
| **data_driven** | 数据驱动的力学性能（抗拉强度 TS）预测 | `data_driven/data_driven_model.py` |

## 系统架构

```
工艺数据（元素成分、辊道速度、风机开度）
        │
        ▼
┌─────────────────────────┐
│   sim_T 机理仿真        │  ← 有限差分 + JMAK 相变动力学
│   温度曲线 + 组织转变量  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   data_augmentation     │  ← Jittering 数据增强
│   扩充训练数据集         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   data_driven 模型      │  ← Transformer Decoder
│   抗拉强度 TS 预测       │     回归模型
└─────────────────────────┘
```

## sim_T —— 温度与组织相变仿真

基于有限差分法求解线材径向 1D 瞬态热传导方程，耦合 JMAK 相变动力学模型。

**物理模型：**
- 对流换热：Žukauskas 强制对流 + Churchill & Chu 自然对流
- 辐射换热：Stefan-Boltzmann 线性化
- 相变潜热：铁素体 + 珠光体 JMAK 动力学
- 佳灵装置 (Optiflex)：搭接/非搭接点风速分配（Deng et al., 2025）
- 保温罩、搭接点换热折减

**核心函数（`sim_T/sim_T.py`）：**

| 函数 | 功能 |
|------|------|
| `run_full_simulation()` | 运行完整 26 段斯太尔摩冷却线仿真 |
| `run_simulation_with_params()` | 便捷接口，返回测温点温度 |
| `calculate_heat_transfer()` | 计算对流+辐射换热系数 |
| `calculate_phase_change_heat()` | 计算相变热焓（JMAK 动力学） |
| `cooling_calculation()` | 单段辊道有限差分求解 |
| `plot_T_results()` / `plot_H_results()` / `plot_Q_results()` | 可视化 |

**辅助文件：**
- `calculate_all_sim_T.py` — 批量并行仿真，输出 CSV
- `change_parameter_82A.py` — PSO 参数优化（优化换热/相变修正系数）
- `calculate_the_effect.py` — 优化效果评估（Parity Plot）

## data_augmentation —— 数据增强

基于 Jittering 方法对工艺-温度数据集进行增强：
- 对元素含量、温度序列、TS 标签添加受控高斯噪声
- 保证增强数据不与原始数据重复
- 输出 `all_process_data.csv` 供模型训练使用

## data_driven —— 力学性能预测

基于 Transformer Decoder 架构的回归模型，预测帘线钢抗拉强度。

**模型结构：**
- 输入：8 维元素成分（静态特征）+ 2 路温度时序列（非搭接点/搭接点）
- 可学习 Query Token 与温度序列做 Cross-Attention
- 回归头输出标量 TS 预测值

**核心函数（`data_driven/data_driven_model.py`）：**

| 函数 | 功能 |
|------|------|
| `load_dataset()` | 加载并自动划分 Train/Val/Test |
| `preprocess_data()` | z-score 归一化 + 时序特征拼接 |
| `run_holdout_training()` | 完整训练流程 |
| `save_checkpoint()` / `load_checkpoint()` | 模型持久化 |
| `evaluate()` | MAE / RMSE / R² 评估 |

**辅助文件：**
- `predict.py` — 全流程预测：工艺参数 → 温度仿真 → TS 预测
- `fine_tuning.py` — 微调预测头（未启用）
- `wrong_data_delet.py` — 离群数据剔除（未启用）

**TensorBoard 可视化：**
```bash
tensorboard --logdir data_driven/runs
```

## 运行方式

### 单次温度仿真（含绘图）
```bash
cd sim_T
python sim_T.py
```

### 批量温度仿真（输出 CSV）
```bash
cd sim_T
python calculate_all_sim_T.py
```

### PSO 参数优化
```bash
cd sim_T
python change_parameter_82A.py
```

### 数据增强
```bash
cd data_augmentation
python make_data_big.py
```

### 模型训练
```bash
cd data_driven
python data_driven_model.py
```

### 全流程预测（工艺参数 → TS）
```bash
cd data_driven
python predict.py
```

## 依赖

- Python 3.x
- NumPy, Pandas, Matplotlib
- PyTorch (data_driven)
- TensorBoard (可选，训练可视化)

## 参考文献

- Deng T.W. et al. (2025). Finite Element Simulation and Parameter Optimization of SWRH82B Wire Rod in Stelmor Cooling Process. *J. Mater. Eng. Perform.*, 34, 11212-11225.
- Žukauskas, A. (1972). Heat Transfer from Tubes in Crossflow. *Advances in Heat Transfer*, 8, 93-160.
- Churchill, S.W. & Chu, H.H.S. (1975). Correlating equations for laminar and turbulent free convection from a horizontal cylinder. *Int. J. Heat Mass Transfer*, 18, 1049-1053.
- Yafei S. et al. (2009). Effect of temperature and composition on thermal properties of carbon steel. *CCDC*.
