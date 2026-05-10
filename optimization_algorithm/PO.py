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

# ==================== 解的类定义 ====================
class Solution:
    """存储单个解的位置和代价（适应度），对应MATLAB中的结构体Sol(i)"""
    def __init__(self, dim):
        self.X = np.zeros(dim)  # 位置向量
        self.Cost = float('inf')  # 代价（适应度值）


# BASE_CHEMISTRY = {
#     "C_ELE": 0.82,
#     "SI_ELE": 0.21,
#     "MN_ELE": 0.52,
#     "P_ELE": 0.011,
#     "S_ELE": 0.006,
#     "CR_ELE": 0.012,
#     "NI_ELE": 0.007,
#     "CU_ELE": 0.01,
# }

PROCESS_VAR_COLS = [
    "C_ELE","SI_ELE","MN_ELE","P_ELE","S_ELE","CR_ELE","NI_ELE","CU_ELE",
    "ORT",
    "SPEED1", "SPEED2", "SPEED3", "SPEED4", "SPEED5",
    "SPEED6", "SPEED7", "SPEED8", "SPEED9", "SPEED10",
    "FAN1", "FAN2", "FAN3", "FAN4", "FAN5", "FAN6",
]


def _build_process_batch_df(X_batch):
    X_arr = np.asarray(X_batch, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    if X_arr.shape[1] != len(PROCESS_VAR_COLS):
        raise ValueError(
            f"候选解维度不匹配，期望 {len(PROCESS_VAR_COLS)}，实际 {X_arr.shape[1]}"
        )

    df = pd.DataFrame(X_arr, columns=PROCESS_VAR_COLS)
    # for col, value in BASE_CHEMISTRY.items():
    #     df[col] = value

    # ordered_cols = list(BASE_CHEMISTRY.keys()) + PROCESS_VAR_COLS
    ordered_cols = PROCESS_VAR_COLS
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
    按批次评估候选解，阶段顺序为：
    1) 批次温度仿真 run_all_simulations
    2) 批次温度重采样 resample_sim_data
    3) 批次性能预测 predict_Ts
    """
    X_arr = np.asarray(X_batch, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    n_batch = X_arr.shape[0]
    process_batch_df = _build_process_batch_df(X_arr)
    tag = batch_tag if batch_tag else "Batch"

    t_stage = time.perf_counter()
    print(f"[{tag}] 温度仿真开始，批量大小={n_batch}")
    all_sim_t_batch = calc.run_all_simulations(process_batch_df, n_workers=0)
    print(f"[{tag}] 温度仿真完成，用时 {time.perf_counter() - t_stage:.2f}s")

    t_stage = time.perf_counter()
    print(f"[{tag}] 温度重采样开始，批量大小={n_batch}")
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
    print(f"[{tag}] 力学性能预测开始，批量大小={n_batch}")
    predict_device = os.getenv("STELMOR_PREDICT_DEVICE", "auto")
    Ts_batch = pred.predict_Ts(process_data_new_batch, device=predict_device)
    Ts_batch = np.asarray(Ts_batch, dtype=np.float64).reshape(-1)
    if Ts_batch.size == 1 and n_batch > 1:
        Ts_batch = np.repeat(Ts_batch, n_batch)
    if Ts_batch.size != n_batch:
        raise ValueError(f"predict_Ts 返回长度异常，期望 {n_batch}，实际 {Ts_batch.size}")
    print(f"[{tag}] 力学性能预测完成，用时 {time.perf_counter() - t_stage:.2f}s")

    return -Ts_batch

# ==================== 1. 边界检查函数 ====================
def boundary_check(X, lb, ub):
    """
    边界检查：确保解的每个维度都在上下界范围内
    对应MATLAB文件：boundaryCheck.m
    
    参数：
    X : np.ndarray - 待检查的位置向量
    lb : float/np.ndarray - 下界
    ub : float/np.ndarray - 上界
    
    返回：
    X : np.ndarray - 修正后的位置向量
    """
    return np.clip(X, lb, ub)

# ==================== 2. 测试函数定义（F1-F23） ====================
def F1(x):
    """Sphere函数，对应MATLAB的F1"""
    return np.sum(x**2)

def F2(x):
    """对应MATLAB的F2"""
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    """对应MATLAB的F3"""
    dim = len(x)
    return sum(np.sum(x[:i+1])**2 for i in range(dim))

def F4(x):
    """对应MATLAB的F4"""
    return np.max(np.abs(x))

def F5(x):
    """Rosenbrock函数，对应MATLAB的F5"""
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (x[:-1]-1)**2)

def F6(x):
    """对应MATLAB的F6"""
    return np.sum((x + 0.5)**2)

def F7(x):
    """对应MATLAB的F7"""
    dim = len(x)
    return np.sum(np.arange(1, dim+1) * x**4) + np.random.rand()

def F8(x):
    """Schwefel函数，对应MATLAB的F8"""
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F9(x):
    """Rastrigin函数，对应MATLAB的F9"""
    dim = len(x)
    return np.sum(x**2 - 10*np.cos(2*np.pi*x)) + 10*dim

def F10(x):
    """Ackley函数，对应MATLAB的F10"""
    dim = len(x)
    return (-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/dim)) 
            - np.exp(np.sum(np.cos(2*np.pi*x))/dim) + 20 + np.exp(1))

def F11(x):
    """Griewank函数，对应MATLAB的F11"""
    dim = len(x)
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, dim+1)))) + 1

def Ufun(x, a, k, m):
    """辅助函数，对应MATLAB的Ufun"""
    return k*((x-a)**m)*(x>a) + k*((-x-a)**m)*(x<-a)

def F12(x):
    """对应MATLAB的F12"""
    dim = len(x)
    term1 = (np.pi/dim)*(10*np.sin(np.pi*(1+(x[0]+1)/4))**2 +
                         np.sum((((x[:-1]+1)/4)**2)*(1+10*np.sin(np.pi*(1+(x[1:]+1)/4))**2)) +
                         ((x[-1]+1)/4)**2)
    term2 = np.sum(Ufun(x, 10, 100, 4))
    return term1 + term2

def F13(x):
    """对应MATLAB的F13"""
    dim = len(x)
    term1 = 0.1*(np.sin(3*np.pi*x[0])**2 +
                 np.sum((x[:-1]-1)**2*(1+np.sin(3*np.pi*x[1:])**2)) +
                 (x[-1]-1)**2*(1+np.sin(2*np.pi*x[-1])**2))
    term2 = np.sum(Ufun(x, 5, 100, 4))
    return term1 + term2

def F14(x):
    """对应MATLAB的F14"""
    aS = np.array([[-32,-16,0,16,32]*5, [-32]*5+[-16]*5+[0]*5+[16]*5+[32]*5])
    bS = np.sum((x[:,None]-aS)**6, axis=0)
    return (1/500 + np.sum(1/(np.arange(1,26)+bS)))**(-1)

def F15(x):
    """对应MATLAB的F15"""
    aK = np.array([.1957,.1947,.1735,.16,.0844,.0627,.0456,.0342,.0323,.0235,.0246])
    bK = 1/np.array([.25,.5,1,2,4,6,8,10,12,14,16])
    return np.sum((aK - (x[0]*(bK**2+x[1]*bK))/(bK**2+x[2]*bK+x[3]))**2)

def F16(x):
    """对应MATLAB的F16"""
    return 4*x[0]**2 - 2.1*x[0]**4 + x[0]**6/3 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4

def F17(x):
    """对应MATLAB的F17"""
    return (x[1] - 5.1*x[0]**2/(4*np.pi**2) + 5*x[0]/np.pi -6)**2 + 10*(1-1/(8*np.pi))*np.cos(x[0]) +10

def F18(x):
    """对应MATLAB的F18"""
    term1 = 1 + (x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)
    term2 = 30 + (2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2)
    return term1*term2

def F19(x):
    """对应MATLAB的F19"""
    aH = np.array([[3,10,30],[.1,10,35],[3,10,30],[.1,10,35]])
    cH = np.array([1,1.2,3,3.2])
    pH = np.array([[.3689,.117,.2673],[.4699,.4387,.747],[.1091,.8732,.5547],[.03815,.5743,.8828]])
    return -np.sum(cH * np.exp(-np.sum(aH*(x-pH)**2, axis=1)))

def F20(x):
    """对应MATLAB的F20"""
    aH = np.array([[10,3,17,3.5,1.7,8],[.05,10,17,.1,8,14],[3,3.5,1.7,10,17,8],[17,8,.05,10,.1,14]])
    cH = np.array([1,1.2,3,3.2])
    pH = np.array([[.1312,.1696,.5569,.0124,.8283,.5886],[.2329,.4135,.8307,.3736,.1004,.9991],
                   [.2348,.1415,.3522,.2883,.3047,.6650],[.4047,.8828,.8732,.5743,.1091,.0381]])
    return -np.sum(cH * np.exp(-np.sum(aH*(x-pH)**2, axis=1)))

def F21(x):
    """对应MATLAB的F21"""
    aSH = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7]])
    cSH = np.array([.1,.2,.2,.4,.4])
    return -np.sum([1/((x-aSH[i])@(x-aSH[i])+cSH[i]) for i in range(5)])

def F22(x):
    """对应MATLAB的F22"""
    aSH = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3]])
    cSH = np.array([.1,.2,.2,.4,.4,.6,.3])
    return -np.sum([1/((x-aSH[i])@(x-aSH[i])+cSH[i]) for i in range(7)])

def F23(x):
    """对应MATLAB的F23"""
    aSH = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH = np.array([.1,.2,.2,.4,.4,.6,.3,.7,.5,.5])
    return -np.sum([1/((x-aSH[i])@(x-aSH[i])+cSH[i]) for i in range(10)])

# ==================== 3. 测试函数详情获取 ====================
def get_functions_details(F):
    """
    获取测试函数的上下界、维度和函数句柄
    对应MATLAB文件：Get_Functions_details.m
    """
    n1 = 30  # 默认最大维度
    func_dict = {
        'F1': (F1, -100, 100, n1),
        'F2': (F2, -10, 10, n1),
        'F3': (F3, -100, 100, n1),
        'F4': (F4, -100, 100, n1),
        'F5': (F5, -30, 30, n1),
        'F6': (F6, -100, 100, n1),
        'F7': (F7, -1.28, 1.28, n1),
        'F8': (F8, -500, 500, n1),
        'F9': (F9, -5.12, 5.12, n1),
        'F10': (F10, -32, 32, n1),
        'F11': (F11, -600, 600, n1),
        'F12': (F12, -50, 50, n1),
        'F13': (F13, -50, 50, n1),
        'F14': (F14, -65.536, 65.536, 2),
        'F15': (F15, -5, 5, 4),
        'F16': (F16, -5, 5, 2),
        'F17': (F17, np.array([-5,0]), np.array([10,15]), 2),
        'F18': (F18, -5, 5, 2),
        'F19': (F19, 0, 1, 3),
        'F20': (F20, 0, 1, 6),
        'F21': (F21, 0, 10, 4),
        'F22': (F22, 0, 10, 4),
        'F23': (F23, 0, 10, 4),
    }
    if F not in func_dict:
        raise ValueError(f"未知测试函数: {F}")
    fobj, lb, ub, dim = func_dict[F]
    return lb, ub, dim, fobj

# ==================== 4. 探索阶段函数 ====================
def exploration(Sol, lb, ub, dim, nSol, CostFunction, BatchCostFunction=None, batch_tag=""):
    """
    探索阶段（全局搜索）
    对应MATLAB文件：Exploration.m
    """
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
        
        y = boundary_check(y, lb, ub)
        z = np.zeros_like(x)
        j0 = np.random.randint(dim)
        
        for j in range(dim):
            if j == j0 or np.random.rand() <= PCR:
                z[j] = y[j]
            else:
                z[j] = x[j]

        candidates[i] = z

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

# ==================== 5. 开发阶段函数 ====================
def exploitation(Sol, lb, ub, dim, nSol, Best, MaxIter, Iter, CostFunction, BatchCostFunction=None, batch_tag=""):
    """
    开发阶段（局部搜索）
    对应MATLAB文件：Exploitation.m
    """
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

        new_sol.X = boundary_check(new_sol.X, lb, ub)
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

# ==================== 6. 美洲豹优化算法主函数 ====================
def puma(nSol, MaxIter, lb, ub, dim, CostFunction, BatchCostFunction=None):
    """
    美洲豹优化算法主流程
    对应MATLAB文件：Puma.m
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
    
    # 种群初始化
    X_init = np.random.uniform(lb, ub, (nSol, dim))
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
    
    # -------------------- 无经验阶段（前3次迭代） --------------------
    for Iter in range(1, 4):
        # 执行探索和开发
        Sol_Explor = exploration(
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
        
        Sol_Exploit = exploitation(
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
            Sol = exploration(
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
            Sol = exploitation(
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
    
    return Best.X, Best.Cost, Convergence

def get_stelmor_sim_details():
    #确定所有工艺参数的上界
    ub = np.array([0.8870,0.23,0.55,0.015,0.01,0.058,0.257,0.0161, 920, 1.21,1.34,1.46,1.48,1.48,1.48,1.48,1.48,1.53,1.60, 100,100,100,100,100,100])
    
    #确定所有工艺参数的下界
    lb = np.array([0.81,0.18,0.47,0.0062,0.0025,0.006,0.005,0.0059, 840, 0.82,0.88,0.96,1.06,0.43,0.43,0.43,0.43,0.43,0.43, 0,0,0,0,0,0])

    # #确定所有工艺参数的上界
    # ub = np.array([0.82,0.21,0.52,0.011,0.006,0.012,0.007,0.01, 920, 1.21,1.34,1.46,1.48,1.48,1.48,1.48,1.48,1.53,1.60, 100,100,100,100,100,100])
    # #确定所有工艺参数的下界
    # lb = np.array([0.82,0.21,0.52,0.011,0.006,0.012,0.007,0.01, 840, 0.82,0.88,0.96,1.06,0.43,0.43,0.43,0.43,0.43,0.43, 0,0,0,0,0,0])

    #确定工艺参数的维度
    dim = len(lb)
    return lb, ub, dim

def stelmor_sim_CostFunction(x):
    return float(stelmor_batch_cost_function(np.asarray(x, dtype=np.float64).reshape(1, -1), batch_tag="Single")[0])



# ==================== 7. 主程序入口 ====================
def main():
    """
    主程序，对应MATLAB文件：main.m
    """
    # # 选择测试函数（F1-F23）
    # Function_name = 'F1'
    
    # # 获取测试函数信息
    # lb, ub, dim, fobj = get_functions_details(Function_name)
    
    # 算法参数
    MaxIter = 100   # 最大迭代次数，最小为3
    PopSize = 300   # 种群规模最小为7，因为算法中需要选择6个不同的个体
    
    lb, ub, dim = get_stelmor_sim_details()

    # 运行算法   
    best_pos, best_score, convergence = puma(
        PopSize,
        MaxIter,
        lb,
        ub,
        dim,
        stelmor_sim_CostFunction,
        BatchCostFunction=stelmor_batch_cost_function,
    )
    
    best_solution_text = (
        f"Best Score:{-best_score}\n"
        f"C:{best_pos[0]}\nSi:{best_pos[1]}\nMn:{best_pos[2]}\nP:{best_pos[3]}\nS:{best_pos[4]}\nCr:{best_pos[5]}\nNi:{best_pos[6]}\nCu:{best_pos[7]}\n"
        f"ORT:{best_pos[8]}\n"
        f"SPEED1:{best_pos[9]}\nSPEED2:{best_pos[10]}\nSPEED3:{best_pos[11]}\nSPEED4:{best_pos[12]}\nSPEED5:{best_pos[13]}\nSPEED6:{best_pos[14]}\nSPEED7:{best_pos[15]}\nSPEED8:{best_pos[16]}\nSPEED9:{best_pos[17]}\nSPEED10:{best_pos[18]}\n"
        f"FAN1:{best_pos[19]}\nFAN2:{best_pos[20]}\nFAN3:{best_pos[21]}\nFAN4:{best_pos[22]}\nFAN5:{best_pos[23]}\nFAN6:{best_pos[24]}"
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

    plt.title('Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    plt.xlim(0, max(len(curve) - 1, 1))
    plt.grid(False)
    plt.box(True)
    plt.show()

    print(best_solution_text)

if __name__ == "__main__":
    main()
