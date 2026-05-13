"""
粒子群优化 (PSO) 算法 —— 优化 sim_T 仿真的修正参数，使温度仿真逼近实测值。

核心仿真与评价函数调用 sim_T.py，本模块仅负责 PSO 优化逻辑。
"""

import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import sim_T as sim
import calculate_all_sim_T as calc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_BOUNDS = {
    "xs_hc0": (0.8, 1.2),
    "xs_hc1": (0.8, 1.2),
    "view_factor": (0.8, 1.0),
    "xs_tauf": (0.8, 1.2),
    "xs_taup": (0.8, 1.2),
    "xs_dqp": (0.8, 1.2),
    "xs_dqf": (0.8, 1.2),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
LOWER_BOUNDS = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES], dtype=float)
UPPER_BOUNDS = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES], dtype=float)
LOWER_BOUNDS_T = torch.tensor(LOWER_BOUNDS, dtype=torch.float32, device=DEVICE)
UPPER_BOUNDS_T = torch.tensor(UPPER_BOUNDS, dtype=torch.float32, device=DEVICE)

PARAMETER_FILE = Path(__file__).with_name("parameter.txt")
SIM_DT = 0.1


# ═══════════════════════════════════════════════════════════════
# PSO 工具函数
# ═══════════════════════════════════════════════════════════════


def vector_to_param_obj(vector, num):
    """将 PSO 参数向量映射为 parameter_change 对象。"""
    if isinstance(vector, torch.Tensor):
        vector_np = vector.detach().cpu().numpy()
    else:
        vector_np = np.asarray(vector, dtype=float)

    param_obj = sim.parameter_change(num)
    for idx, name in enumerate(PARAM_NAMES):
        setattr(param_obj, name, float(vector_np[idx]))
    return param_obj


def write_parameter_file(param_obj, file_path=PARAMETER_FILE):
    """将优化后的参数写入 parameter.txt。"""
    lines = []
    for name in PARAM_NAMES:
        value = float(getattr(param_obj, name))
        lines.append(f"{name}:{value:.6f}")
    file_path.write_text("\n".join(lines), encoding="utf-8")


def initialize_swarm(swarm_size):
    """初始化粒子群位置与速度。"""
    dim = len(PARAM_NAMES)
    rand_pos = torch.rand((swarm_size, dim), dtype=torch.float32, device=DEVICE)
    positions = LOWER_BOUNDS_T + rand_pos * (UPPER_BOUNDS_T - LOWER_BOUNDS_T)

    v_max = 0.2 * (UPPER_BOUNDS_T - LOWER_BOUNDS_T)
    rand_vel = torch.rand((swarm_size, dim), dtype=torch.float32, device=DEVICE)
    velocities = -v_max + rand_vel * (2.0 * v_max)
    return positions, velocities


def _vector_to_param_dict(vector):
    """将参数向量转为 dict，便于批量仿真调用。"""
    if isinstance(vector, torch.Tensor):
        vector_np = vector.detach().cpu().numpy()
    else:
        vector_np = np.asarray(vector, dtype=float)
    return {name: float(vector_np[idx]) for idx, name in enumerate(PARAM_NAMES)}


def _build_dummy_process_data(batch_size):
    """构造仅用于并行仿真的占位工艺数据。"""
    return pd.DataFrame([{} for _ in range(batch_size)])


# ═══════════════════════════════════════════════════════════════
# 测温点提取
# ═══════════════════════════════════════════════════════════════


def _get_measure_point_times():
    """计算测温点时间序列 —— 委托 sim_T.get_measure_point_times()。"""
    # 被调用: sim_T.get_measure_point_times (sim_T.py)
    rolls, _ = sim.data_loader.load_roll_data()
    roll_start_time = [0.0]
    for r in rolls:
        roll_start_time.append(roll_start_time[-1] + float(r.t))
    return sim.get_measure_point_times(roll_start_time)


def _extract_measure_points_from_row(sim_row, measure_times):
    """从仿真列中提取测温点温度，匹配最接近的时间列。"""
    time_map_0 = {}
    time_map_1 = {}
    for col in sim_row.index:
        if col.endswith("(0)"):
            try:
                time_map_0[float(col[:-3])] = col
            except ValueError:
                continue
        elif col.endswith("(1)"):
            try:
                time_map_1[float(col[:-3])] = col
            except ValueError:
                continue

    if not time_map_0 or not time_map_1:
        raise ValueError("仿真结果缺少 (0)/(1) 温度列，无法提取测温点。")

    time_values = sorted(time_map_1.keys())

    def _closest_time(target):
        return min(time_values, key=lambda t: abs(t - target))

    sim1, sim0 = [], []
    for t in measure_times:
        closest_t = _closest_time(float(t))
        sim1.append(float(sim_row[time_map_1[closest_t]]))
        sim0.append(float(sim_row[time_map_0[closest_t]]))

    return np.asarray(sim1, dtype=np.float64), np.asarray(sim0, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
# 误差评价函数
# ═══════════════════════════════════════════════════════════════


def calculate_mae_and_similarity(real_data1, sim_data1, real_data0, sim_data0):
    """计算 MAE 与相似度。similarity = 1 / (1 + total_mae)。"""
    real_arr1 = np.array(real_data1)
    sim_arr1 = np.array(sim_data1)
    real_arr0 = np.array(real_data0)
    sim_arr0 = np.array(sim_data0)

    if real_arr1.shape != sim_arr1.shape:
        raise ValueError(f"sim1 维度不匹配: {real_arr1.shape} vs {sim_arr1.shape}")
    if real_arr0.shape != sim_arr0.shape:
        raise ValueError(f"sim0 维度不匹配: {real_arr0.shape} vs {sim_arr0.shape}")

    mae1 = np.mean(np.abs(real_arr1 - sim_arr1))
    mae0 = np.mean(np.abs(real_arr0 - sim_arr0))
    total_mae = (mae1 + mae0) / 2.0
    similarity = 1.0 / (1.0 + total_mae)

    return total_mae, similarity


def calculate_wmae_and_similarity(real_data1, sim_data1, real_data0, sim_data0):
    """计算加权 WMAE 与相似度。使用 sim_T.calculate_importance_for_real_point 计算权重。"""
    def _to_1d_array(data, name):
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{name} 不能为空")
        return arr

    def _calculate_group_wmae(x_real, y_real, y_sim, group_name):
        x_arr = _to_1d_array(x_real, f"{group_name} X")
        y_real_arr = _to_1d_array(y_real, f"{group_name} 真实温度")
        y_sim_arr = _to_1d_array(y_sim, f"{group_name} 仿真温度")

        if x_arr.shape != y_real_arr.shape:
            raise ValueError(f"{group_name} X 与真实温度维度不匹配: {x_arr.shape} vs {y_real_arr.shape}")
        if y_real_arr.shape != y_sim_arr.shape:
            raise ValueError(f"{group_name} 数据维度不匹配: {y_real_arr.shape} vs {y_sim_arr.shape}")

        # 调用 sim_T 中的权重计算函数
        weights = np.asarray(
            sim.calculate_importance_for_real_point(x_arr, y_real_arr),
            dtype=np.float64,
        ).reshape(-1)
        return float(np.sum(weights * np.abs(y_real_arr - y_sim_arr)))

    x_real1 = np.arange(1, len(real_data1) + 1, dtype=np.float64)
    x_real0 = np.arange(1, len(real_data0) + 1, dtype=np.float64)

    wmae1 = _calculate_group_wmae(x_real1, real_data1, sim_data1, "sim1")
    wmae0 = _calculate_group_wmae(x_real0, real_data0, sim_data0, "sim0")
    total_wmae = (wmae1 + wmae0) / 2.0
    similarity = 1.0 / (1.0 + total_wmae)

    return total_wmae, similarity


# ═══════════════════════════════════════════════════════════════
# 评估函数（单粒子 / 批量）
# ═══════════════════════════════════════════════════════════════


def evaluate_position(position, measure_point_T_real1, measure_point_T_real0,
                      eval_id, metric_mode="MAE"):
    """评估一个粒子位置，调用 sim_T.run_simulation_with_params 执行仿真。

    返回: (param_obj, total_error, similarity, sim1, sim0)
    """
    param_obj = vector_to_param_obj(position, eval_id)
    measure_point_T_sim1, measure_point_T_sim0 = sim.run_simulation_with_params(
        param_obj, dt=SIM_DT,
    )

    mode = str(metric_mode).upper()
    if mode == "MAE":
        total_mae, similarity = calculate_mae_and_similarity(
            measure_point_T_real1, measure_point_T_sim1,
            measure_point_T_real0, measure_point_T_sim0,
        )
    elif mode == "WMAE":
        total_mae, similarity = calculate_wmae_and_similarity(
            measure_point_T_real1, measure_point_T_sim1,
            measure_point_T_real0, measure_point_T_sim0,
        )
    else:
        raise ValueError(f"不支持的误差模式: {metric_mode}，仅支持 'MAE' 或 'WMAE'")

    param_obj.result = similarity
    return param_obj, total_mae, similarity, measure_point_T_sim1, measure_point_T_sim0


def evaluate_positions_batch(positions, measure_point_T_real1, measure_point_T_real0,
                             eval_start_id, metric_mode="MAE", dt=SIM_DT):
    """批量评估粒子位置，使用 calculate_all_sim_T.run_all_simulations 并行计算。"""
    import pandas as pd

    if isinstance(positions, torch.Tensor):
        positions_np = positions.detach().cpu().numpy()
    else:
        positions_np = np.asarray(positions, dtype=float)

    batch_size = positions_np.shape[0]
    param_objs = []
    param_dicts = []
    for i in range(batch_size):
        eval_id = eval_start_id + i
        param_obj = vector_to_param_obj(positions_np[i], eval_id)
        param_objs.append(param_obj)
        param_dicts.append(_vector_to_param_dict(positions_np[i]))

    process_batch_df = _build_dummy_process_data(batch_size)
    sim_batch = calc.run_all_simulations(
        process_batch_df, n_workers=0, dt_override=dt, params_list=param_dicts,
    )

    measure_times = _get_measure_point_times()
    results = []
    for i in range(batch_size):
        sim1, sim0 = _extract_measure_points_from_row(sim_batch.iloc[i], measure_times)
        mode = str(metric_mode).upper()
        if mode == "MAE":
            total_error, similarity = calculate_mae_and_similarity(
                measure_point_T_real1, sim1, measure_point_T_real0, sim0,
            )
        elif mode == "WMAE":
            total_error, similarity = calculate_wmae_and_similarity(
                measure_point_T_real1, sim1, measure_point_T_real0, sim0,
            )
        else:
            raise ValueError(f"不支持的误差模式: {metric_mode}，仅支持 'MAE' 或 'WMAE'")

        param_obj = param_objs[i]
        param_obj.result = similarity
        results.append((param_obj, total_error, similarity, sim1, sim0))

    return results


# ═══════════════════════════════════════════════════════════════
# PSO 主流程
# ═══════════════════════════════════════════════════════════════


def main():
    global SIM_DT
    SIM_DT = 0.1
    metric_mode = "WMAE"  # 可选: "MAE" 或 "WMAE"

    swarm_size = 100
    max_iter = 200
    early_stop_patience = 10
    inertia_w = 0.7
    cognitive_c1 = 1.7
    social_c2 = 1.7

    print(
        f"开始 PSO 优化：粒子数={swarm_size}, 迭代={max_iter}, "
        f"评价={metric_mode}, 设备={DEVICE}"
    )

    # 现场实测测温点温度（数据定义在 sim_T.py）
    measure_point_T_real1 = np.array(sim.MEASURE_POINT_T_REAL_1)
    measure_point_T_real0 = np.array(sim.MEASURE_POINT_T_REAL_0)

    positions, velocities = initialize_swarm(swarm_size)

    pbest_positions = positions.clone()
    pbest_scores = torch.full((swarm_size,), -float("inf"), dtype=torch.float32, device=DEVICE)

    gbest_position = None
    gbest_score = -float("inf")
    best_param = None
    eval_counter = 0
    no_improve_rounds = 0

    v_max = 0.2 * (UPPER_BOUNDS_T - LOWER_BOUNDS_T)

    start_time = time.time()
    interrupted = False

    try:
        for iteration in range(max_iter):
            print(f"\n--- 第 {iteration + 1}/{max_iter} 轮 ---")
            gbest_updated_this_round = False

            batch_results = evaluate_positions_batch(
                positions, measure_point_T_real1, measure_point_T_real0,
                eval_counter + 1, metric_mode, dt=SIM_DT,
            )

            for i, (param_obj, total_error, similarity, _, _) in enumerate(batch_results):
                eval_counter += 1
                print(
                    f"[Eval {eval_counter:04d}] 粒子 {i + 1:02d}/{swarm_size}, "
                    f"{metric_mode}: {total_error:.4f}, 相似度: {similarity:.6f}"
                )

                if similarity > float(pbest_scores[i].item()):
                    pbest_scores[i] = similarity
                    pbest_positions[i] = positions[i].clone()

                if similarity > gbest_score:
                    gbest_score = similarity
                    gbest_position = positions[i].clone()
                    best_param = copy.deepcopy(param_obj)
                    gbest_updated_this_round = True

            if gbest_updated_this_round:
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            print(f"本轮结束，全局最优相似度: {gbest_score:.6f}，连续未更新: {no_improve_rounds}")

            if no_improve_rounds >= early_stop_patience:
                print(f"连续 {early_stop_patience} 轮未更新最优解，触发提前停止。")
                break

            r1 = torch.rand((swarm_size, len(PARAM_NAMES)), dtype=torch.float32, device=DEVICE)
            r2 = torch.rand((swarm_size, len(PARAM_NAMES)), dtype=torch.float32, device=DEVICE)

            cognitive_term = cognitive_c1 * r1 * (pbest_positions - positions)
            social_term = social_c2 * r2 * (gbest_position.unsqueeze(0) - positions)

            velocities = inertia_w * velocities + cognitive_term + social_term
            velocities = torch.max(torch.min(velocities, v_max), -v_max)
            positions = positions + velocities
            positions = torch.max(torch.min(positions, UPPER_BOUNDS_T), LOWER_BOUNDS_T)

    except KeyboardInterrupt:
        interrupted = True
        print("\n检测到手动中断，正在输出当前最优参数...")
    except Exception as e:
        interrupted = True
        print(f"\n运行异常: {type(e).__name__}: {e}")
        print("正在输出当前最优参数...")

    end_time = time.time()

    print("\n" + "=" * 40)
    if gbest_position is None:
        print("未获得可用最优解。")
        print("=" * 40)
        return

    best_param = vector_to_param_obj(
        gbest_position, best_param.num if best_param is not None else 0,
    )
    best_param.result = gbest_score

    print(f"PSO 优化完成！总耗时: {end_time - start_time:.2f} 秒")
    print(f"最佳匹配度: {best_param.result:.6f}")
    print(f"最佳参数组合 [评估序号 {best_param.num}]:")
    print(f"  xs_hc0 (非搭接对流): {best_param.xs_hc0:.4f}")
    print(f"  xs_hc1 (搭接对流):   {best_param.xs_hc1:.4f}")
    print(f"  view_factor (辐射):  {best_param.view_factor:.4f}")
    print(f"  xs_tauf (铁素体孕育): {best_param.xs_tauf:.4f}")
    print(f"  xs_taup (珠光体孕育): {best_param.xs_taup:.4f}")
    print(f"  xs_dqp (珠光体热焓):  {best_param.xs_dqp:.4f}")
    print(f"  xs_dqf (铁素体热焓):  {best_param.xs_dqf:.4f}")
    print("=" * 40)

    write_parameter_file(best_param)
    print(f"优化参数已保存到: {PARAMETER_FILE}")

    if interrupted:
        print("检测到异常退出，已输出当前最优参数，跳过绘图。")
        return

    # 绘制最佳结果对比图
    best_sim1, best_sim0 = sim.run_simulation_with_params(best_param, dt=SIM_DT)
    x_points = np.arange(1, len(measure_point_T_real1) + 1)
    mode = str(metric_mode).upper()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    axes[0].plot(x_points, best_sim1, linestyle='-', linewidth=2, color='tab:blue',
                 label='Best Simulated Temperature 1')

    if mode == "WMAE":
        weights1 = np.asarray(
            sim.calculate_importance_for_real_point(x_points, measure_point_T_real1),
            dtype=np.float64,
        )
        weights0 = np.asarray(
            sim.calculate_importance_for_real_point(x_points, measure_point_T_real0),
            dtype=np.float64,
        )
        norm = plt.Normalize(
            vmin=min(np.min(weights1), np.min(weights0)),
            vmax=max(np.max(weights1), np.max(weights0)),
        )
        cmap = plt.get_cmap("RdYlGn_r")

        sc1 = axes[0].scatter(x_points, measure_point_T_real1, marker='o', s=90,
                              c=weights1, cmap=cmap, norm=norm,
                              edgecolors='black', linewidths=0.6,
                              label='Measured Temperature 1 (Weighted)')
        for x, y, w in zip(x_points, measure_point_T_real1, weights1):
            axes[0].text(x + 0.05, y + 2.0, f"{w:.3f}", fontsize=8, color='black')
    else:
        axes[0].scatter(x_points, measure_point_T_real1, marker='o', s=90,
                        color='tab:orange', edgecolors='black', linewidths=0.6,
                        label='Measured Temperature 1')

    axes[0].set_xlabel('Measurement Point Index')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Temperature Group 1')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_points, best_sim0, linestyle='-', linewidth=2, color='tab:green',
                 label='Best Simulated Temperature 0')

    if mode == "WMAE":
        sc0 = axes[1].scatter(x_points, measure_point_T_real0, marker='o', s=90,
                              c=weights0, cmap=cmap, norm=norm,
                              edgecolors='black', linewidths=0.6,
                              label='Measured Temperature 0 (Weighted)')
        for x, y, w in zip(x_points, measure_point_T_real0, weights0):
            axes[1].text(x + 0.05, y + 2.0, f"{w:.3f}", fontsize=8, color='black')
    else:
        axes[1].scatter(x_points, measure_point_T_real0, marker='o', s=90,
                        color='tab:red', edgecolors='black', linewidths=0.6,
                        label='Measured Temperature 0')

    axes[1].set_xlabel('Measurement Point Index')
    axes[1].set_title('Temperature Group 0')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if mode == "WMAE":
        cbar = fig.colorbar(sc0, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
        cbar.set_label('Real-point Weight (low=green, high=red)')

    fig.suptitle(f'Best Simulation vs Measured Temperature ({mode})', fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
