import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "data_driven"))
sys.path.append(os.path.join(project_root, "sim_T"))
import predict as pred
import calculate_all_sim_T as calc
import expert_constraints as ec

# ==================== 1. 数据结构 ====================

class Solution:
    """候选解：位置向量 + 代价（适应度值）。"""
    def __init__(self, dim):
        self.X = np.zeros(dim)
        self.Cost = float('inf')


# ==================== 2. 搜索空间定义（21 维，不含成分） ====================

# 固定化学成分 —— 82A 标准中值，非优化变量
FIXED_CHEMISTRY = {
    "C_ELE": 0.82, "SI_ELE": 0.25, "MN_ELE": 0.50,
    "P_ELE": 0.012, "S_ELE": 0.010,
    "CR_ELE": 0.20, "NI_ELE": 0.05, "CU_ELE": 0.05,
}

# 优化变量（21 维）：ORT + SPEED1~10 + FAN1~10
PROCESS_VAR_COLS = [
    "ORT",
    "SPEED1", "SPEED2", "SPEED3", "SPEED4", "SPEED5",
    "SPEED6", "SPEED7", "SPEED8", "SPEED9", "SPEED10",
    "FAN1", "FAN2", "FAN3", "FAN4", "FAN5",
    "FAN6", "FAN7", "FAN8", "FAN9", "FAN10",
]

# 优化变量上下界（工艺可行性约束）
_VAR_LB = np.array([
    890,                                    # ORT (°C)
    0.50, 0.50, 0.50, 0.50, 0.50,          # SPEED1~5 (m/s)
    0.50, 0.50, 0.50, 0.50, 0.50,          # SPEED6~10
    0, 0, 0, 0, 0, 0,                      # FAN1~6 (%)
    0, 0, 0, 0,                            # FAN7~10 (%)
], dtype=np.float64)

_VAR_UB = np.array([
    940,                                    # ORT (°C)
    1.60, 1.60, 1.60, 1.60, 1.60,          # SPEED1~5 (m/s)
    1.60, 1.60, 1.60, 1.60, 1.60,          # SPEED6~10
    100, 100, 100, 100, 100, 100,           # FAN1~6 (%)
    100, 100, 100, 100,                     # FAN7~10 (%)
], dtype=np.float64)


def get_search_bounds():
    """返回 (下界, 上界, 维度)。"""
    return _VAR_LB.copy(), _VAR_UB.copy(), len(_VAR_LB)


# ==================== 3. 候选解 → 工艺参数 DataFrame ====================

def _build_process_df(X_batch):
    """将优化变量矩阵转为工艺参数 DataFrame，注入固定化学成分。"""
    X_arr = np.asarray(X_batch, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    if X_arr.shape[1] != len(PROCESS_VAR_COLS):
        raise ValueError(
            f"候选解维度不匹配，期望 {len(PROCESS_VAR_COLS)}，实际 {X_arr.shape[1]}"
        )

    df = pd.DataFrame(X_arr, columns=PROCESS_VAR_COLS)
    # 注入固定化学成分
    for col, value in FIXED_CHEMISTRY.items():
        df[col] = value
    # 列顺序与 run_all_simulations 期望一致：化学成分 + 工艺参数
    ordered_cols = list(FIXED_CHEMISTRY.keys()) + PROCESS_VAR_COLS
    return df[ordered_cols]


def _evaluate_candidates_batch(X_batch, CostFunction, BatchCostFunction=None, batch_tag=""):
    X_arr = np.asarray(X_batch, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    if BatchCostFunction is not None:
        return np.asarray(BatchCostFunction(X_arr, batch_tag=batch_tag), dtype=np.float64)

    costs = np.empty(X_arr.shape[0], dtype=np.float64)
    for i in range(X_arr.shape[0]):
        costs[i] = CostFunction(X_arr[i])
    return costs


def stelmor_batch_cost_function(X_batch, batch_tag=""):
    """
    批量评估候选解，依次执行：
    1) 预检查 —— 过滤明显越界候选解
    2) 温度仿真 + 相变数据提取
    3) 温度重采样 + ML 力学性能预测
    4) 专家约束评价 + 评分聚合
       Score = k_MP * MP_cost + k_EK * (EK_penalty - w_bonus * EK_bonus)
    """
    X_arr = np.asarray(X_batch, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    n_batch = X_arr.shape[0]
    process_batch_df = _build_process_df(X_arr)
    tag = batch_tag if batch_tag else "Batch"

    # ── 阶段 1: 预检查 ──
    pre_penalties = np.zeros(n_batch, dtype=np.float64)
    feasible_mask = np.ones(n_batch, dtype=bool)
    for i in range(n_batch):
        row = process_batch_df.iloc[i]
        params = {
            "ORT": float(row["ORT"]),
            "SPEED": [float(row[f"SPEED{j}"]) for j in range(1, 11)],
            "FAN": [float(row[f"FAN{j}"]) for j in range(1, 11)],
        }
        ok, p = ec.pre_check_bounds(params)
        feasible_mask[i] = ok
        pre_penalties[i] = p

    n_feasible = feasible_mask.sum()
    print(f"[{tag}] 预检查: {n_feasible}/{n_batch} 通过")

    # ── 阶段 2: 温度仿真（含相变状态数据） ──
    t_stage = time.perf_counter()
    print(f"[{tag}] 温度仿真开始，可行批量大小={n_feasible}")
    all_sim_t_batch, state_data_list = calc.run_all_simulations(
        process_batch_df, n_workers=0, return_states=True
    )
    print(f"[{tag}] 温度仿真完成，用时 {time.perf_counter() - t_stage:.2f}s")

    # ── 阶段 3: 温度重采样 + ML 预测 ──
    t_stage = time.perf_counter()
    print(f"[{tag}] 温度重采样开始")
    resampled_list = []
    for i in range(n_batch):
        demo_i = process_batch_df.iloc[[i]]
        row_sim_t = all_sim_t_batch.iloc[[i]]
        time_values = sorted(
            {float(col[:-3]) for col in row_sim_t.columns if col.endswith("(0)")},
            key=float,
        )
        tem0_i = [float(row_sim_t.iloc[0][f"{t:.2f}(0)"]) for t in time_values]
        tem1_i = [float(row_sim_t.iloc[0][f"{t:.2f}(1)"]) for t in time_values]
        resampled_i = pred.resample_sim_data(time_values, tem0_i, tem1_i, process_data=demo_i)
        resampled_list.append(resampled_i)
    process_data_new_batch = pd.concat(resampled_list, axis=0, ignore_index=True)
    print(f"[{tag}] 温度重采样完成，用时 {time.perf_counter() - t_stage:.2f}s")

    t_stage = time.perf_counter()
    print(f"[{tag}] 力学性能预测开始")
    predict_device = os.getenv("STELMOR_PREDICT_DEVICE", "auto")
    Ts_batch = pred.predict_Ts(process_data_new_batch, device=predict_device)
    Ts_batch = np.asarray(Ts_batch, dtype=np.float64).reshape(-1)
    if Ts_batch.size == 1 and n_batch > 1:
        Ts_batch = np.repeat(Ts_batch, n_batch)
    if Ts_batch.size != n_batch:
        raise ValueError(f"predict_Ts 返回长度异常，期望 {n_batch}，实际 {Ts_batch.size}")
    print(f"[{tag}] 力学性能预测完成，用时 {time.perf_counter() - t_stage:.2f}s")

    # ── 阶段 4: 专家约束评价 + 评分聚合 ──
    # Score = k_MP * MP_cost + k_EK * (EK_penalty - w_bonus * EK_bonus)
    scores = np.full(n_batch, float("inf"), dtype=np.float64)
    for i in range(n_batch):
        if not feasible_mask[i]:
            continue  # 保持 inf

        vals = ec.extract_from_state_data(state_data_list[i])
        feasible, penalty, bonus, details = ec.evaluate_constraints(
            vals, pred_TS=float(Ts_batch[i]), pred_Z=None
        )
        ek_penalty = penalty + pre_penalties[i]  # 专家知识惩罚（含预检查）
        ek_bonus = bonus
        score = ec.compute_total_score(feasible, ek_penalty, ek_bonus, mp_cost=0.0)
        scores[i] = score

        if i == 0 or (i < 3 and batch_tag == "Init"):
            print(f"[{tag}] 样本[{i}] TS={Ts_batch[i]:.0f} cr_pearl={vals.get('cr_pearl','?'):.1f} "
                  f"pearl={vals.get('pearl_frac',0):.3f} dT={vals.get('max_dT_total',0):.1f} "
                  f"feasible={feasible} EK_penalty={ek_penalty:.2f} EK_bonus={ek_bonus:.1f} score={score:.2f}")

    return scores

# ==================== 4. 边界检查 ====================

def _clamp_to_bounds(X, lb, ub):
    """将解向量的每个维度限制在 [lb, ub] 范围内。"""
    return np.clip(X, lb, ub)


def _repair_speeds(X, lb, ub):
    """修复 H7 硬约束：确保相邻段辊速比 ≤ 1.5。

    从 SPEED1 向后级联：每段的辊速被限制在 [prev/1.5, prev*1.5] 内。
    SPEED 索引对应 X[1:11]（ORT 在 X[0]，FAN1~10 在 X[11:21]）。
    修复后再夹紧到全局边界。
    """
    X_r = np.clip(X, lb, ub)
    for i in range(2, 11):  # SPEED2 ~ SPEED10
        prev = X_r[i - 1]
        lo = prev / 1.5
        hi = prev * 1.5
        if X_r[i] < lo or X_r[i] > hi:
            X_r[i] = np.clip(X_r[i], lo, hi)
    return np.clip(X_r, lb, ub)


# ==================== 5. Puma 优化算法 ====================

# --- 探索阶段 ---
def _exploration_phase(Sol, lb, ub, dim, nSol, CostFunction, BatchCostFunction=None, batch_tag=""):
    """探索阶段：全局搜索，通过随机差分变异生成新候选解。"""
    # 按代价排序种群
    Sol = sorted(Sol, key=lambda s: s.Cost)
    pCR = 0.2
    PCR = 1 - pCR
    p = PCR / nSol
    NewSol = []
    candidates = np.zeros((nSol, dim), dtype=np.float64)
    
    for i in range(nSol):
        x = Sol[i].X.copy()
        # 随机选择6个不同的其他个体
        A = np.random.permutation(nSol)
        A = A[A != i][:6]
        a,b,c,d,e,f = A
        
        G = 2*np.random.rand() - 1
        if np.random.rand() < 0.5:
            y = np.random.uniform(lb, ub, dim)
        else:
            y = (Sol[a].X + G*(Sol[a].X-Sol[b].X) + 
                 G*(((Sol[a].X-Sol[b].X)-(Sol[c].X-Sol[d].X)) + 
                    ((Sol[c].X-Sol[d].X)-(Sol[e].X-Sol[f].X))))
        
        y = _clamp_to_bounds(y, lb, ub)
        z = np.zeros_like(x)
        j0 = np.random.randint(dim)

        for j in range(dim):
            if j == j0 or np.random.rand() <= PCR:
                z[j] = y[j]
            else:
                z[j] = x[j]

        candidates[i] = _repair_speeds(z, lb, ub)

    candidate_costs = _evaluate_candidates_batch(
        candidates,
        CostFunction,
        BatchCostFunction=BatchCostFunction,
        batch_tag=batch_tag,
    )

    for i in range(nSol):
        new_sol = Solution(dim)
        new_sol.X = candidates[i]
        new_sol.Cost = candidate_costs[i]

        if new_sol.Cost < Sol[i].Cost:
            NewSol.append(new_sol)
        else:
            NewSol.append(Sol[i])
            pCR += p
    
    return NewSol

# --- 开发阶段 ---
def _exploitation_phase(Sol, lb, ub, dim, nSol, Best, MaxIter, Iter, CostFunction, BatchCostFunction=None, batch_tag=""):
    """开发阶段：局部搜索，利用当前最优解引导搜索方向。"""
    Q = 0.67
    Beta = 2
    NewSol = []
    candidates = np.zeros((nSol, dim), dtype=np.float64)
    X_mat = np.array([s.X for s in Sol])
    mbest = np.mean(X_mat, axis=0)
    
    for i in range(nSol):
        beta1 = 2*np.random.rand()
        beta2 = np.random.randn(dim)
        w = np.random.randn(dim)
        v = np.random.randn(dim)
        F1 = np.random.randn(dim) * np.exp(2 - Iter*(2/MaxIter))
        F2 = w * v**2 * np.cos(2*np.random.rand()*w)
        
        R_1 = 2*np.random.rand() - 1
        S1 = (2*np.random.rand()-1) + np.random.randn(dim)
        S2 = F1*R_1*Sol[i].X + F2*(1-R_1)*Best.X
        VEC = S2 / S1  # 与原MATLAB保持一致
        
        new_sol = Solution(dim)
        if np.random.rand() <= 0.5:
            Xatack = VEC
            if np.random.rand() > Q:
                r_idx = np.random.randint(nSol)
                new_sol.X = Best.X + beta1*np.exp(beta2)*(Sol[r_idx].X - Sol[i].X)
            else:
                new_sol.X = beta1*Xatack - Best.X
        else:
            r1 = np.random.randint(nSol)
            sign = (-1)**np.random.randint(0,2)
            new_sol.X = (mbest*Sol[r1].X - sign*Sol[i].X) / (1 + Beta*np.random.rand())

        new_sol.X = _repair_speeds(_clamp_to_bounds(new_sol.X, lb, ub), lb, ub)
        candidates[i] = new_sol.X

    candidate_costs = _evaluate_candidates_batch(
        candidates,
        CostFunction,
        BatchCostFunction=BatchCostFunction,
        batch_tag=batch_tag,
    )

    for i in range(nSol):
        new_sol = Solution(dim)
        new_sol.X = candidates[i]
        new_sol.Cost = candidate_costs[i]

        if new_sol.Cost < Sol[i].Cost:
            NewSol.append(new_sol)
        else:
            NewSol.append(Sol[i])
    
    return NewSol

# --- 算法主流程 ---
def puma_optimize(nSol, MaxIter, lb, ub, dim, CostFunction, BatchCostFunction=None,
                  patience=30):
    """
    Puma（美洲豹）优化算法。

    参数：
        nSol: 种群规模（≥7）
        MaxIter: 最大迭代次数（≥3）
        lb, ub: 搜索空间下/上界，可为标量或同维向量
        dim: 搜索空间维度
        CostFunction: 单样本代价函数 f(x) → float
        BatchCostFunction: 批量代价函数 f(X) → ndarray，若提供则用于批量评估
        patience: 早停容忍轮数，最优解连续未更新达到此数则提前终止（默认 30）

    返回：
        (best_position, best_cost, convergence_curve)
    """
    # 参数初始化
    UnSelected = np.ones(2)
    F3_Explore = F3_Exploit = 0
    Seq_Time_Explore = Seq_Time_Exploit = np.ones(3)
    Seq_Cost_Explore = Seq_Cost_Exploit = np.ones(3)
    Score_Explore = Score_Exploit = 0
    PF = np.array([0.5, 0.5, 0.3])
    PF_F3 = []
    Mega_Explor = Mega_Exploit = 0.99
    
    # 种群初始化（含硬约束修复：H7 速比 ≤ 1.5）
    X_init = np.random.uniform(lb, ub, (nSol, dim))
    for i in range(nSol):
        X_init[i] = _repair_speeds(X_init[i], lb, ub)
    init_costs = _evaluate_candidates_batch(
        X_init,
        CostFunction,
        BatchCostFunction=BatchCostFunction,
        batch_tag="Init",
    )
    Sol = []
    for i in range(nSol):
        s = Solution(dim)
        s.X = X_init[i]
        s.Cost = init_costs[i]
        Sol.append(s)
    
    # 初始最优解
    Best = min(Sol, key=lambda s: s.Cost)
    Initial_Best = Best
    Flag_Change = 1
    Convergence = []
    no_improve_count = 0  # 早停计数器
    
    # -------------------- 无经验阶段（前3次迭代） --------------------
    for Iter in range(1, 4):
        # 执行探索和开发
        Sol_Explor = _exploration_phase(
            Sol,
            lb,
            ub,
            dim,
            nSol,
            CostFunction,
            BatchCostFunction=BatchCostFunction,
            batch_tag=f"Iter {Iter} Explore",
        )
        cost_explor = min(s.Cost for s in Sol_Explor)
        Seq_Cost_Explore[Iter-1] = cost_explor
        
        Sol_Exploit = _exploitation_phase(
            Sol,
            lb,
            ub,
            dim,
            nSol,
            Best,
            MaxIter,
            Iter,
            CostFunction,
            BatchCostFunction=BatchCostFunction,
            batch_tag=f"Iter {Iter} Exploit",
        )
        cost_exploit = min(s.Cost for s in Sol_Exploit)
        Seq_Cost_Exploit[Iter-1] = cost_exploit
        
        # 合并并选择最优
        Sol = sorted(Sol + Sol_Explor + Sol_Exploit, key=lambda s: s.Cost)[:nSol]
        Best = Sol[0]
        Convergence.append(Best.Cost)
        print(f"Iteration: {Iter} Best Cost = {Best.Cost}")
    
    # -------------------- 超参数初始化 --------------------
    # 计算Seq_Cost差值
    Seq_Cost_Explore[0] = abs(Initial_Best.Cost - Seq_Cost_Explore[0])
    Seq_Cost_Exploit[0] = abs(Initial_Best.Cost - Seq_Cost_Exploit[0])
    for i in range(1, 3):
        Seq_Cost_Explore[i] = abs(Seq_Cost_Explore[i] - Seq_Cost_Explore[i-1])
        Seq_Cost_Exploit[i] = abs(Seq_Cost_Exploit[i] - Seq_Cost_Exploit[i-1])
    
    # 收集非零成本
    for i in range(3):
        if Seq_Cost_Explore[i] != 0: PF_F3.append(Seq_Cost_Explore[i])
        if Seq_Cost_Exploit[i] != 0: PF_F3.append(Seq_Cost_Exploit[i])
    
    # 计算初始分数
    F1_Explor = PF[0] * (Seq_Cost_Explore[0]/Seq_Time_Explore[0])
    F1_Exploit = PF[0] * (Seq_Cost_Exploit[0]/Seq_Time_Exploit[0])
    F2_Explor = PF[1] * (np.sum(Seq_Cost_Explore)/np.sum(Seq_Time_Explore))
    F2_Exploit = PF[1] * (np.sum(Seq_Cost_Exploit)/np.sum(Seq_Time_Exploit))
    Score_Explore = PF[0]*F1_Explor + PF[1]*F2_Explor
    Score_Exploit = PF[0]*F1_Exploit + PF[1]*F2_Exploit
    
    # -------------------- 有经验阶段（第4次迭代起） --------------------
    for Iter in range(4, MaxIter+1):
        if Score_Explore > Score_Exploit:
            # 选择探索
            SelectFlag = 1
            Sol = _exploration_phase(
                Sol,
                lb,
                ub,
                dim,
                nSol,
                CostFunction,
                BatchCostFunction=BatchCostFunction,
                batch_tag=f"Iter {Iter} Explore",
            )
            Count_select = UnSelected.copy()
            UnSelected[1] += 1
            UnSelected[0] = 1
            F3_Explore = PF[2]
            F3_Exploit += PF[2]
        else:
            # 选择开发
            SelectFlag = 2
            Sol = _exploitation_phase(
                Sol,
                lb,
                ub,
                dim,
                nSol,
                Best,
                MaxIter,
                Iter,
                CostFunction,
                BatchCostFunction=BatchCostFunction,
                batch_tag=f"Iter {Iter} Exploit",
            )
            Count_select = UnSelected.copy()
            UnSelected[0] += 1
            UnSelected[1] = 1
            F3_Explore += PF[2]
            F3_Exploit = PF[2]
        
        # 更新当前最优
        TBest = min(Sol, key=lambda s: s.Cost)
        if SelectFlag == 1:
            Seq_Cost_Explore[2] = Seq_Cost_Explore[1]
            Seq_Cost_Explore[1] = Seq_Cost_Explore[0]
            Seq_Cost_Explore[0] = abs(Best.Cost - TBest.Cost)
            if Seq_Cost_Explore[0] != 0: PF_F3.append(Seq_Cost_Explore[0])
        else:
            Seq_Cost_Exploit[2] = Seq_Cost_Exploit[1]
            Seq_Cost_Exploit[1] = Seq_Cost_Exploit[0]
            Seq_Cost_Exploit[0] = abs(Best.Cost - TBest.Cost)
            if Seq_Cost_Exploit[0] != 0: PF_F3.append(Seq_Cost_Exploit[0])
        
        if TBest.Cost < Best.Cost:
            Best = TBest
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 更新时间序列
        if Flag_Change != SelectFlag:
            Flag_Change = SelectFlag
            Seq_Time_Explore[2] = Seq_Time_Explore[1]
            Seq_Time_Explore[1] = Seq_Time_Explore[0]
            Seq_Time_Explore[0] = Count_select[0]
            Seq_Time_Exploit[2] = Seq_Time_Exploit[1]
            Seq_Time_Exploit[1] = Seq_Time_Exploit[0]
            Seq_Time_Exploit[0] = Count_select[1]
        
        # 更新分数
        F1_Explor = PF[0] * (Seq_Cost_Explore[0]/Seq_Time_Explore[0])
        F1_Exploit = PF[0] * (Seq_Cost_Exploit[0]/Seq_Time_Exploit[0])
        F2_Explor = PF[1] * (np.sum(Seq_Cost_Explore)/np.sum(Seq_Time_Explore))
        F2_Exploit = PF[1] * (np.sum(Seq_Cost_Exploit)/np.sum(Seq_Time_Exploit))
        
        if Score_Explore < Score_Exploit:
            Mega_Explor = max(Mega_Explor - 0.01, 0.01)
            Mega_Exploit = 0.99
        elif Score_Explore > Score_Exploit:
            Mega_Explor = 0.99
            Mega_Exploit = max(Mega_Exploit - 0.01, 0.01)
        
        lmn_Explore = 1 - Mega_Explor
        lmn_Exploit = 1 - Mega_Exploit
        min_PF_F3 = min(PF_F3) if PF_F3 else 1e-6
        
        Score_Explore = (Mega_Explor*F1_Explor + Mega_Explor*F2_Explor + 
                         lmn_Explore*min_PF_F3*F3_Explore)
        Score_Exploit = (Mega_Exploit*F1_Exploit + Mega_Exploit*F2_Exploit + 
                         lmn_Exploit*min_PF_F3*F3_Exploit)
        
        Convergence.append(Best.Cost)
        print(f"Iteration: {Iter} Best Cost = {Best.Cost}")

        # 早停机制
        if no_improve_count >= patience:
            print(f"早停: 最优解已连续 {patience} 轮未更新，在第 {Iter} 轮提前终止")
            break

    return Best.X, Best.Cost, Convergence

# ==================== 6. 代价函数接口 ====================

def get_stelmor_sim_details():
    """返回 (下界, 上界, 维度) —— 兼容旧接口，委托给 get_search_bounds。"""
    return get_search_bounds()


def stelmor_sim_CostFunction(x):
    """单样本代价函数包装，供优化器逐样本评估时调用。"""
    return float(stelmor_batch_cost_function(
        np.asarray(x, dtype=np.float64).reshape(1, -1), batch_tag="Single")[0])


# ==================== 7. 主程序入口 ====================

def main():
    """82A 斯太尔摩工艺参数优化主程序。

    Score = k_MP * MP_cost + k_EK * (EK_penalty - w_bonus * EK_bonus)
    MP_cost 暂为 0（力学性能数据不完全），当前仅专家知识项生效。
    """
    MaxIter = 200
    PopSize = 30

    lb, ub, dim = get_search_bounds()

    best_pos, best_score, convergence = puma_optimize(
        PopSize,
        MaxIter,
        lb,
        ub,
        dim,
        stelmor_sim_CostFunction,
        BatchCostFunction=stelmor_batch_cost_function,
    )

    # 输出最优解（21 维：ORT + SPEED1~10 + FAN1~10）
    best_solution_text = (
        f"Best Score:{best_score:.3f} (0=ideal, inf=infeasible)\n"
        f"ORT:{best_pos[0]:.0f}\n"
        f"SPEED1:{best_pos[1]:.3f}\nSPEED2:{best_pos[2]:.3f}\nSPEED3:{best_pos[3]:.3f}\n"
        f"SPEED4:{best_pos[4]:.3f}\nSPEED5:{best_pos[5]:.3f}\nSPEED6:{best_pos[6]:.3f}\n"
        f"SPEED7:{best_pos[7]:.3f}\nSPEED8:{best_pos[8]:.3f}\nSPEED9:{best_pos[9]:.3f}\n"
        f"SPEED10:{best_pos[10]:.3f}\n"
        f"FAN1:{best_pos[11]:.0f}\nFAN2:{best_pos[12]:.0f}\nFAN3:{best_pos[13]:.0f}\n"
        f"FAN4:{best_pos[14]:.0f}\nFAN5:{best_pos[15]:.0f}\nFAN6:{best_pos[16]:.0f}\n"
        f"FAN7:{best_pos[17]:.0f}\nFAN8:{best_pos[18]:.0f}\nFAN9:{best_pos[19]:.0f}\n"
        f"FAN10:{best_pos[20]:.0f}"
    )
    best_solution_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_solution.txt")
    with open(best_solution_path, "w", encoding="utf-8") as f:
        f.write(best_solution_text)

    # 绘制收敛曲线。
    # semilogy 仅适用于正值；当目标值包含非正数时回退到线性坐标，避免对数缩放告警。
    curve = np.asarray(convergence, dtype=np.float64)
    plt.figure(figsize=(8, 5))
    if np.all(curve > 0):
        plt.semilogy(curve, color='r', linewidth=1.25)
        y_label = 'Best Cost So Far'
    elif np.all((-curve) > 0):
        plt.semilogy(-curve, color='r', linewidth=1.25)
        y_label = 'Best Objective Value So Far'
    else:
        plt.plot(curve, color='r', linewidth=1.25)
        y_label = 'Best Cost So Far (Linear Scale)'

    plt.title('Convergence Curve (Constraint Score)')
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    plt.xlim(0, max(len(curve) - 1, 1))
    plt.grid(False)
    plt.box(True)
    plt.show()

    print(best_solution_text)

if __name__ == "__main__":
    main()
