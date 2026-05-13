import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import torch
import pandas as pd
from pathlib import Path

# 导入重命名后的仿真主文件（原文件需重命名为 sim_T.py）
import sim_T as sim
import calculate_all_sim_T as calc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_BOUNDS = {
    'xs_hc0': (0.8, 1.2),
    'xs_hc1': (0.8, 1.2),
    'view_factor': (0.8, 1.0),
    'xs_tauf': (0.8, 1.2),
    'xs_taup': (0.8, 1.2),
    'xs_dqp': (0.8, 1.2),
    'xs_dqf': (0.8, 1.2),
}

PARAM_NAMES = list(PARAM_BOUNDS.keys())
LOWER_BOUNDS = np.array([PARAM_BOUNDS[name][0] for name in PARAM_NAMES], dtype=float)
UPPER_BOUNDS = np.array([PARAM_BOUNDS[name][1] for name in PARAM_NAMES], dtype=float)
LOWER_BOUNDS_T = torch.tensor(LOWER_BOUNDS, dtype=torch.float32, device=DEVICE)
UPPER_BOUNDS_T = torch.tensor(UPPER_BOUNDS, dtype=torch.float32, device=DEVICE)

PARAMETER_FILE = Path(__file__).with_name("parameter.txt")
SIM_DT = 0.1


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


def _get_measure_point_times():
    """按 sim_T.py 的规则计算测温点时间序列。"""
    rolls, _ = sim.data_loader.load_roll_data()
    roll_start_time = [0.0]
    for r in rolls:
        roll_start_time.append(roll_start_time[-1] + float(r.t))

    big_idx = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22]
    small_idx = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    big_start_time = [roll_start_time[i] for i in big_idx if i < len(roll_start_time)]
    small_start_time = [roll_start_time[i] for i in small_idx if i < len(roll_start_time)]

    measure_point_time = [
        big_start_time[0],
        big_start_time[1],
        small_start_time[0],
        big_start_time[2],
        small_start_time[1],
        big_start_time[3],
        small_start_time[2],
        big_start_time[4],
        small_start_time[3],
        big_start_time[5],
        small_start_time[4],
        big_start_time[6],
        small_start_time[5],
        big_start_time[7],
        small_start_time[6],
    ]

    return measure_point_time[:8]


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

    sim1 = []
    sim0 = []
    for t in measure_times:
        closest_t = _closest_time(float(t))
        sim1.append(float(sim_row[time_map_1[closest_t]]))
        sim0.append(float(sim_row[time_map_0[closest_t]]))

    return np.asarray(sim1, dtype=np.float64), np.asarray(sim0, dtype=np.float64)

def calculate_mae_and_similarity(real_data1, sim_data1, real_data0, sim_data0):
    """
    功能 3: 仿真结果评价方法。分别计算两组测点的 MAE，并综合得到总 MAE 与相似度。
    采用公式: similarity = 1 / (1 + total_mae)。
    """
    real_arr1 = np.array(real_data1)
    sim_arr1 = np.array(sim_data1)
    real_arr0 = np.array(real_data0)
    sim_arr0 = np.array(sim_data0)

    # 确保每组真实值与仿真值维度一致
    if real_arr1.shape != sim_arr1.shape:
        raise ValueError(f"sim1 数据维度不匹配! 真实值维度: {real_arr1.shape}, 仿真值维度: {sim_arr1.shape}")
    if real_arr0.shape != sim_arr0.shape:
        raise ValueError(f"sim0 数据维度不匹配! 真实值维度: {real_arr0.shape}, 仿真值维度: {sim_arr0.shape}")

    # 分别计算两组 MAE，再取均值作为综合 MAE
    mae1 = np.mean(np.abs(real_arr1 - sim_arr1))
    mae0 = np.mean(np.abs(real_arr0 - sim_arr0))
    total_mae = (mae1 + mae0) / 2.0

    # 将误差映射到 0~1 之间。完全一致时 MAE=0, 相似度=1
    similarity = 1.0 / (1.0 + total_mae)
    return total_mae, similarity

def calculate_importance_for_real_point(X, Y, peak_weight_ratio=0.5, lambda_smooth=0.9):
    """
    高度可调的权重计算算法。
    
    参数:
    X, Y: 原始数据的横纵坐标向量。
    peak_weight_ratio (float): 0~1 之间。值越大，越侧重于捕捉梯度峰值（回温）；
                            值越小，越侧重于捕捉差分不一致（转折）。默认 0.7。
    lambda_smooth (float): 0~1 之间。值越大，权重方差越小，分配越均匀。默认 0.4。
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    n = len(X)
    
    if n < 3:
        return np.ones(n) / n

    # 1. 计算前向和后向差分
    m_b = np.zeros(n)
    m_f = np.zeros(n)
    m_b[1:] = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])
    m_b[0] = m_b[1]
    m_f[:-1] = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])
    m_f[-1] = m_f[-2]

    # 2. 提取两个核心特征得分
    # 特征 A: 差分不一致性 (转折点特征)
    s_discord = np.abs(m_f - m_b)
    
    # 特征 B: 梯度峰值特征 (回温/潜热特征)
    # 增加平方运算，显著拉开峰值与普通点的得分差距
    s_peak = np.zeros(n)
    diff_left = m_b[1:-1] - m_b[:-2]
    diff_right = m_b[1:-1] - m_b[2:]
    s_peak[1:-1] = (np.maximum(0, diff_left) * np.maximum(0, diff_right))**2

    # 3. 标准化处理 (Min-Max)
    def safe_norm(vec):
        v_min, v_max = np.min(vec), np.max(vec)
        return (vec - v_min) / (v_max - v_min + 1e-9)

    norm_discord = safe_norm(s_discord)
    norm_peak = safe_norm(s_peak)

    # 4. 线性加权融合得分
    # 使用 peak_weight_ratio 调节两者的权重
    combined_score = (1 - peak_weight_ratio) * norm_discord + peak_weight_ratio * norm_peak

    # 5. 非线性缩放处理与混合权重模型
    # 取平方根是为了在保持特征导向的同时，平抑极值，控制方差
    feature_impact = np.sqrt(combined_score + 1e-9)
    feature_weight = feature_impact / np.sum(feature_impact)

    # 最终混合：保底权重 + 特征权重
    uniform_weight = np.ones(n) / n
    final_weights = lambda_smooth * uniform_weight + (1 - lambda_smooth) * feature_weight
    
    return final_weights

def calculate_wmae_and_similarity(real_data1, sim_data1, real_data0, sim_data0):
    """
    功能 3(加权版): 仿真结果评价方法。分别计算两组测点的 WMAE，并综合得到总 WMAE 与相似度。
    采用公式: similarity = 1 / (1 + total_wmae)。
    """
    def _to_1d_array(data, name):
        arr = np.asarray(data, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{name} 不能为空")
        return arr

    def _calculate_group_wmae(x_real, y_real, y_sim, group_name):
        x_arr = _to_1d_array(x_real, f"{group_name} 的 X")
        y_real_arr = _to_1d_array(y_real, f"{group_name} 的真实温度")
        y_sim_arr = _to_1d_array(y_sim, f"{group_name} 的仿真温度")

        if x_arr.shape != y_real_arr.shape:
            raise ValueError(
                f"{group_name} 的 X 与真实温度维度不匹配! X: {x_arr.shape}, 真实值: {y_real_arr.shape}"
            )
        if y_real_arr.shape != y_sim_arr.shape:
            raise ValueError(
                f"{group_name} 数据维度不匹配! 真实值维度: {y_real_arr.shape}, 仿真值维度: {y_sim_arr.shape}"
            )

        weights = np.asarray(
            calculate_importance_for_real_point(x_arr, y_real_arr),
            dtype=np.float64,
        ).reshape(-1)

        if weights.shape != y_real_arr.shape:
            raise ValueError(
                f"{group_name} 权重维度异常! 权重维度: {weights.shape}, 数据维度: {y_real_arr.shape}"
            )

        # 重要性加权平均绝对误差
        return float(np.sum(weights * np.abs(y_real_arr - y_sim_arr)))

    x_real1 = np.arange(1, len(real_data1) + 1, dtype=np.float64)
    x_real0 = np.arange(1, len(real_data0) + 1, dtype=np.float64)

    wmae1 = _calculate_group_wmae(x_real1, real_data1, sim_data1, "sim1")
    wmae0 = _calculate_group_wmae(x_real0, real_data0, sim_data0, "sim0")
    total_wmae = (wmae1 + wmae0) / 2.0

    similarity = 1.0 / (1.0 + total_wmae)
    return total_wmae, similarity


def run_simulation_with_params(param_obj, dt=SIM_DT):
    """
    功能 2: 将一组参数带入仿真模型，执行计算并返回该次仿真的测点温度数组。
    这里复用了原文件 __main__ 中的核心逻辑。
    """
    # 1. 将生成的参数对象应用到仿真模型中
    # 注意：这里假设你的 simulation_model 类中有一个全局的参数变量接收修正系数
    # 如果你的代码逻辑不同，请确保将 param_obj 传递给 Cooling_calculation 计算函数中
    sim.simulation_model.current_params = param_obj 
    
    # 2. 初始化环境（这部分与原程序的 __main__ 完全一致）
    sim.simulation_model.dt = float(dt)
    rolls, num_rolls = sim.data_loader.load_roll_data()
    tem1 = 850  # 入口温度
    tem0 = 830
    temp_trans_1 = np.full(sim.simulation_model.N, tem1)
    temp_trans_0 = np.full(sim.simulation_model.N, tem0)

    # 每组参数都必须从同一初始状态开始，避免历史数据跨轮污染
    N = sim.simulation_model.N
    sim.simulation_model.history_time = []
    sim.simulation_model.history_T_0 = [[] for _ in range(N)]
    sim.simulation_model.history_T_1 = [[] for _ in range(N)]
    sim.simulation_model.history_Q_0 = [[] for _ in range(N)]
    sim.simulation_model.history_Q_1 = [[] for _ in range(N)]
    sim.simulation_model.history_h_0 = [[] for _ in range(3)]
    sim.simulation_model.history_h_1 = [[] for _ in range(3)]
    sim.simulation_model.pearlite_0 = [[] for _ in range(N)]
    sim.simulation_model.pearlite_1 = [[] for _ in range(N)]

    sim.simulation_model.f_total_0 = np.zeros(N)
    sim.simulation_model.f_total_1 = np.zeros(N)
    sim.simulation_model.suma_f0 = np.zeros(N)
    sim.simulation_model.suma_f1 = np.zeros(N)
    sim.simulation_model.suma_p0 = np.zeros(N)
    sim.simulation_model.suma_p1 = np.zeros(N)
    sim.simulation_model.mark_sf_0 = np.zeros(N, dtype=int)
    sim.simulation_model.mark_ef_0 = np.zeros(N, dtype=int)
    sim.simulation_model.mark_sf_1 = np.zeros(N, dtype=int)
    sim.simulation_model.mark_ef_1 = np.zeros(N, dtype=int)
    sim.simulation_model.mark_sp_0 = np.zeros(N, dtype=int)
    sim.simulation_model.mark_ep_0 = np.zeros(N, dtype=int)
    sim.simulation_model.mark_sp_1 = np.zeros(N, dtype=int)
    sim.simulation_model.mark_ep_1 = np.zeros(N, dtype=int)

    sim.each_roll_time = 0
    #记录每个辊道的开始时间
    sim.roll_start_time = []
    sim.roll_start_time.append(sim.each_roll_time)

    # 3. 循环计算每个辊道（模拟斯太尔摩冷却线的物理过程）
    for i in range(num_rolls):
        current_roll = rolls[i]
        current_roll.pre_temp_0 = temp_trans_0
        current_roll.pre_temp_1 = temp_trans_1
        
        # 执行冷却计算。模型内部会调用当前传入的 param_obj 中的各项修正系数
        sim.simulation_model.Cooling_calculation(current_roll)
        
        temp_trans_0 = current_roll.post_temp_0
        temp_trans_1 = current_roll.post_temp_1

        #记录该辊道开始和结束时间
        sim.roll_start_time.append(sim.each_roll_time)

    # 4. 获取仿真运行完毕后的测点温度数据
    # 注意：这里假设 simulation_model 提供了一个获取测点温度数组的方法
    # 你可能需要根据实际情况替换为如：sim.simulation_model.measure_point_T_sim 等变量名
    measure_point_T_sim1, measure_point_T_sim0 = sim.simulation_model.get_measure_point_T_results()
    
    return measure_point_T_sim1, measure_point_T_sim0


def evaluate_position(position, measure_point_T_real1, measure_point_T_real0, eval_id, metric_mode="MAE"):
    """评估一个粒子位置，返回参数对象、综合误差、相似度与仿真结果。"""
    param_obj = vector_to_param_obj(position, eval_id)
    measure_point_T_sim1, measure_point_T_sim0 = run_simulation_with_params(param_obj, dt=SIM_DT)

    mode = str(metric_mode).upper()
    if mode == "MAE":
        total_mae, similarity = calculate_mae_and_similarity(
            measure_point_T_real1,
            measure_point_T_sim1,
            measure_point_T_real0,
            measure_point_T_sim0,
        )
    elif mode == "WMAE":
        total_mae, similarity = calculate_wmae_and_similarity(
            measure_point_T_real1,
            measure_point_T_sim1,
            measure_point_T_real0,
            measure_point_T_sim0,
        )
    else:
        raise ValueError(f"不支持的误差模式: {metric_mode}，仅支持 'MAE' 或 'WMAE'")
    param_obj.result = similarity

    return param_obj, total_mae, similarity, measure_point_T_sim1, measure_point_T_sim0


def evaluate_positions_batch(positions, measure_point_T_real1, measure_point_T_real0, eval_start_id, metric_mode="MAE", dt=SIM_DT):
    """批量评估粒子位置，使用 calculate_all_sim_T.run_all_simulations 并行计算仿真温度。"""
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
        process_batch_df,
        n_workers=0,
        dt_override=dt,
        params_list=param_dicts,
    )

    measure_times = _get_measure_point_times()
    results = []
    for i in range(batch_size):
        sim1, sim0 = _extract_measure_points_from_row(sim_batch.iloc[i], measure_times)
        mode = str(metric_mode).upper()
        if mode == "MAE":
            total_error, similarity = calculate_mae_and_similarity(
                measure_point_T_real1,
                sim1,
                measure_point_T_real0,
                sim0,
            )
        elif mode == "WMAE":
            total_error, similarity = calculate_wmae_and_similarity(
                measure_point_T_real1,
                sim1,
                measure_point_T_real0,
                sim0,
            )
        else:
            raise ValueError(f"不支持的误差模式: {metric_mode}，仅支持 'MAE' 或 'WMAE'")

        param_obj = param_objs[i]
        param_obj.result = similarity
        results.append((param_obj, total_error, similarity, sim1, sim0))

    return results

def main():
    global SIM_DT
    SIM_DT = 0.1
    metric_mode = "WMAE"  # 可选: "MAE" 或 "WMAE"

    # PSO 超参数
    swarm_size = 100  #粒子数量
    max_iter = 200   #迭代轮数
    early_stop_patience = 10  # 连续多少轮未更新全局最优则提前停止
    inertia_w = 0.7     #惯性权重
    cognitive_c1 = 1.7  #自身权重
    social_c2 = 1.7     #社会权重

    print(
        f"开始执行粒子群优化（PSO）：粒子数={swarm_size}, 迭代轮数={max_iter}, "
        f"评价指标={metric_mode.upper()}, 计算设备={DEVICE}"
    )
    
    # 假设这是从现场采集到的真实测点温度数据（用于对比基准）
    # 实际应用中，你需要从你的工艺信息库中加载这组真实数据
    measure_point_T_real1 = np.array([845, 822, 718, 656, 650, 595, 575, 558]) 
    measure_point_T_real0 = np.array([830, 815, 705, 645, 630, 595, 568, 558]) 
    
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
                positions,
                measure_point_T_real1,
                measure_point_T_real0,
                eval_counter + 1,
                metric_mode,
                dt=SIM_DT,
            )

            for i, (param_obj, total_error, similarity, _, _) in enumerate(batch_results):
                eval_counter += 1
                print(
                    f"[Eval {eval_counter:04d}] 粒子 {i + 1:02d}/{swarm_size}, "
                    f"{metric_mode.upper()}: {total_error:.4f}, 相似度: {similarity:.6f}"
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

            print(f"本轮结束，全局最优相似度: {gbest_score:.6f}，连续未更新轮数: {no_improve_rounds}")

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
        print(f"\n运行过程中发生异常: {type(e).__name__}: {e}")
        print("正在输出当前最优参数...")

    end_time = time.time()
    
    # 输出最终寻优结果
    print("\n" + "="*40)
    if gbest_position is None:
        print("未获得可用最优解（可能在首次评估前即退出）。")
        print("="*40)
        return

    best_param = vector_to_param_obj(gbest_position, best_param.num if best_param is not None else 0)
    best_param.result = gbest_score

    print(f"PSO 优化完成！总耗时: {end_time - start_time:.2f} 秒")
    print(f"最佳匹配度 (Result): {best_param.result:.6f}")
    print(f"最佳参数组合 [评估序号 {best_param.num}]:")
    print(f"  xs_hc0 (非搭接点对流): {best_param.xs_hc0:.4f}")
    print(f"  xs_hc1 (搭接点对流):   {best_param.xs_hc1:.4f}")
    print(f"  view_factor (辐射):    {best_param.view_factor:.4f}")
    print(f"  xs_tauf (铁素体孕育):  {best_param.xs_tauf:.4f}")
    print(f"  xs_taup (珠光体孕育):  {best_param.xs_taup:.4f}")
    print(f"  xs_dqp (珠光体热焓):   {best_param.xs_dqp:.4f}")
    print(f"  xs_dqf (铁素体热焓):   {best_param.xs_dqf:.4f}")
    print("="*40)

    write_parameter_file(best_param)
    print(f"优化参数已保存到: {PARAMETER_FILE}")

    if interrupted:
        print("检测到异常退出，已输出当前最优参数，跳过后续绘图。")
        return

    #绘制最佳结果的曲线图
    best_measure_point_T_sim1, best_measure_point_T_sim0 = run_simulation_with_params(best_param, dt=SIM_DT)
    x_points = np.arange(1, len(measure_point_T_real1) + 1)
    mode = str(metric_mode).upper()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    axes[0].plot(
        x_points,
        best_measure_point_T_sim1,
        linestyle='-',
        linewidth=2,
        color='tab:blue',
        label='Best Simulated Temperature 1'
    )

    if mode == "WMAE":
        weights1 = np.asarray(
            calculate_importance_for_real_point(x_points, measure_point_T_real1),
            dtype=np.float64,
        )
        weights0 = np.asarray(
            calculate_importance_for_real_point(x_points, measure_point_T_real0),
            dtype=np.float64,
        )
        norm = plt.Normalize(
            vmin=min(np.min(weights1), np.min(weights0)),
            vmax=max(np.max(weights1), np.max(weights0)),
        )
        cmap = plt.get_cmap("RdYlGn_r")

        sc1 = axes[0].scatter(
            x_points,
            measure_point_T_real1,
            marker='o',
            s=90,
            c=weights1,
            cmap=cmap,
            norm=norm,
            edgecolors='black',
            linewidths=0.6,
            label='Measured Temperature 1 (Weighted)'
        )
        for x, y, w in zip(x_points, measure_point_T_real1, weights1):
            axes[0].text(x + 0.05, y + 2.0, f"{w:.3f}", fontsize=8, color='black')
    else:
        sc1 = axes[0].scatter(
            x_points,
            measure_point_T_real1,
            marker='o',
            s=90,
            color='tab:orange',
            edgecolors='black',
            linewidths=0.6,
            label='Measured Temperature 1'
        )

    axes[0].set_xlabel('Measurement Point Index')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('Temperature Group 1')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        x_points,
        best_measure_point_T_sim0,
        linestyle='-',
        linewidth=2,
        color='tab:green',
        label='Best Simulated Temperature 0'
    )

    if mode == "WMAE":
        sc0 = axes[1].scatter(
            x_points,
            measure_point_T_real0,
            marker='o',
            s=90,
            c=weights0,
            cmap=cmap,
            norm=norm,
            edgecolors='black',
            linewidths=0.6,
            label='Measured Temperature 0 (Weighted)'
        )
        for x, y, w in zip(x_points, measure_point_T_real0, weights0):
            axes[1].text(x + 0.05, y + 2.0, f"{w:.3f}", fontsize=8, color='black')
    else:
        sc0 = axes[1].scatter(
            x_points,
            measure_point_T_real0,
            marker='o',
            s=90,
            color='tab:red',
            edgecolors='black',
            linewidths=0.6,
            label='Measured Temperature 0'
        )

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