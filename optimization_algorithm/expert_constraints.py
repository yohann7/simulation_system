"""
专家约束评价模块 —— 将 plan.txt 3.9 节的冶金约束转化为可执行的评分函数。

使用方式：
    1. pre_check_bounds(params) → 快速预筛（无仿真）
    2. extract_from_state(state, basic_info) → 从仿真状态提取约束值
    3. evaluate_constraints(vals, pred_TS, pred_Z) → 硬/软约束 + 奖励
    4. compute_total_score(feasible, ek_penalty, ek_bonus) → 聚合评分
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# 约束阈值配置（文献推荐值，来源见 plan.txt 3.9 节）
# ═══════════════════════════════════════════════════════════════

CONSTRAINT_CFG = {
    # ── 硬约束阈值 ──
    "ORT":       (890, 940),       # °C, 吐丝温度
    "CR_PEARL":  (9.0, 12.0),      # °C/s, 珠光体温区平均冷却速率（索氏体最优区）
    "CR_550":    5.0,              # °C/s, 550°C 以下冷却速率上限（防马氏体）
    "PEARL_FRAC": 0.85,            # 珠光体转变量下限
    "SPEED":     (0.5, 1.6),       # m/s, 辊道速度（下限产能要求，上限设备能力）
    "SPEED_RATIO_MAX": 1.5,        # 相邻段辊速比硬上限（防堆叠/拉断）
    "FAN":       (0, 100),         # %, 风机开度

    # ── 软约束阈值 ──
    "DT_PEARL":  15.0,             # °C, 相变区搭接/非搭接温差
    "DT_MAX":    20.0,             # °C, 全程搭接/非搭接温差
    "CR_POST":   3.0,              # °C/s, 相变后冷却速率
    "CEM_FRAC":  0.03,             # 先共析渗碳体体积分数
    "SPEED_JUMP": 0.25,            # 相邻段辊速比偏离上限
    "FAN_JUMP":  40.0,             # %, 相邻段风机开度差上限
    "TS":        (980, 1120),      # MPa, 抗拉强度目标范围
    "Z_MIN":     38.0,             # %, 断面收缩率下限

    # ── 奖励阈值 ──
    "T_TRANS_TARGET": 655.0,       # °C, 目标相变温度（82A 索氏体区）
    "T_TRANS_TOL":    10.0,        # °C, 相变温度允许偏差（范围 645~665°C）
    "S0":       (0.12, 0.17),      # μm, 目标片层间距范围
    "PEARL_BONUS": 0.90,           # 索氏体率奖励阈值
}

# 评分权重系数
# Score = k_MP * MP_cost + k_EK * (EK_penalty - w_bonus * EK_bonus)
K_MP = 0.0   # 力学性能项系数（力学性能数据不完全，暂置 0）
K_EK = 1.0   # 专家知识惩罚项系数
W_BONUS = 0.3  # 专家知识奖励项权重（<1，防止奖励压倒惩罚）

# ═══════════════════════════════════════════════════════════════
# 1. 从 SimulationState 提取约束值
# ═══════════════════════════════════════════════════════════════

def extract_from_state(state, basic_info):
    """
    从一次仿真完成的 SimulationState 中提取所有约束相关物理量。

    参数：
        state: SimulationState
        basic_info: sim_T.basic_info 模块（提供 A1, Acm, Bs 等临界温度）

    返回：
        dict，键名对应约束变量：
            cr_pearl, cr_550, pearl_frac, cem_frac,
            max_dT_total, max_dT_pearl, cr_post,
            T_trans, phase, S0_est
    """
    T0 = np.array(state.history_T_0[-1], dtype=np.float64)  # 表面 非搭接
    T1 = np.array(state.history_T_1[-1], dtype=np.float64)  # 表面 搭接
    t = np.array(state.history_time, dtype=np.float64)

    A1 = float(basic_info.A1)
    Bs = float(basic_info.Bs)

    result = {}

    # --- 冷却速率（数值微分，°C/s） ---
    cr = _cooling_rate(t, T0)

    # 珠光体温区平均冷却速率 (A1 → Bs, 即 ~728 → ~560 °C)
    mask_pearl = (T0[:-1] <= A1) & (T0[:-1] >= Bs)
    if mask_pearl.sum() >= 2:
        result["cr_pearl"] = float(np.mean(np.abs(cr[mask_pearl])))
    else:
        result["cr_pearl"] = None  # 未进入此温区

    # 550°C 以下冷却速率（防马氏体）
    mask_low = T0[:-1] < 550.0
    if mask_low.sum() >= 2:
        result["cr_550"] = float(np.max(np.abs(cr[mask_low])))
    else:
        result["cr_550"] = 0.0

    # 相变后冷却速率（珠光体转变量 > 0.99 之后）
    pearl_hist = np.array(state.pearlite_0[-1], dtype=np.float64)
    if len(pearl_hist) > 0 and pearl_hist[-1] > 0.99:
        done_idx = int(np.argmax(pearl_hist > 0.99))
        if done_idx < len(cr):
            result["cr_post"] = float(np.mean(np.abs(cr[done_idx:])))
        else:
            result["cr_post"] = 0.0
    else:
        result["cr_post"] = 0.0

    # --- 珠光体最终转变量 ---
    if len(state.pearlite_0[-1]) > 0:
        result["pearl_frac"] = float(state.pearlite_0[-1][-1])
    else:
        result["pearl_frac"] = 0.0

    # --- 先共析渗碳体分数 ---
    # 82A 为过共析钢（C=0.82% > 0.77%），先共析相为渗碳体
    # 注意：f_total 是总转变量（≈0.99），其中绝大部分为珠光体片层中的渗碳体
    # 而非晶界先共析渗碳体。用杠杆定律估算先共析渗碳体：
    #   cem_pro = max(0, (C_pct - 0.77) / (6.67 - 0.77))
    # ferrite_final 为穿过 A1 时的铁素体量；若接近零则为过共析钢
    f_ferrite = float(state.ferrite_final_0[-1])
    if f_ferrite < 0.01:
        # 过共析钢：先共析渗碳体 ≈ (C% - 0.77) / 5.9
        C_pct = basic_info.ELM_C * 100
        cem_frac = max(0.0, (C_pct - 0.77) / 5.9)
    else:
        # 亚共析钢：先共析相为铁素体，无晶界渗碳体
        cem_frac = 0.0
    result["cem_frac"] = cem_frac
    result["phase"] = "Ferrite" if f_ferrite > 0.01 else "Cementite"

    # --- 搭接/非搭接温差 ---
    dT = np.abs(T0 - T1)
    result["max_dT_total"] = float(np.max(dT)) if len(dT) > 0 else 0.0

    # 相变区温差 (A1 → Bs)
    if mask_pearl.sum() >= 2:
        result["max_dT_pearl"] = float(np.max(dT[:-1][mask_pearl]))
    else:
        result["max_dT_pearl"] = result["max_dT_total"]

    # --- 相变温度（珠光体转变量达到 50% 时的温度，索氏体区最优 ~630°C）---
    if len(pearl_hist) > 0 and pearl_hist[-1] > 0.5:
        half_idx = int(np.argmax(pearl_hist > 0.5))
        if half_idx < len(T0):
            result["T_trans"] = float(T0[half_idx])
        else:
            result["T_trans"] = None
    else:
        result["T_trans"] = None

    # --- 珠光体片层间距估算（基于 V_COOL-spacing 强相关 r=-0.978）---
    # 数据拟合：ln(spacing) ≈ ln(0.197) - 0.38 * ln(V_COOL/4.2)
    # 简化：用 cr_pearl 替代 V_COOL
    cr_pearl_val = result.get("cr_pearl")
    if cr_pearl_val is not None and cr_pearl_val > 0:
        log_cr = np.log(max(cr_pearl_val, 0.5))
        log_S0 = np.log(0.197) - 0.38 * (log_cr - np.log(4.2))
        result["S0_est"] = float(np.exp(log_S0))
    else:
        result["S0_est"] = None

    return result


def _cooling_rate(time, temp):
    """数值微分 dT/dt (°C/s)，返回长度 len(time)-1 的数组。"""
    dt = np.diff(time)
    dT = np.diff(temp)
    safe_dt = np.where(dt > 0, dt, np.inf)
    return dT / safe_dt


# ── 82A 固定化学成分的临界温度 ──

def _compute_critical_temps(C=0.82, Si=0.25, Mn=0.50, Cr=0.20, Ni=0.05):
    """根据化学成分计算 A1 和 Bs 临界温度（sim_T 所用公式）。"""
    A1 = 727 - 10.7 * Mn - 16.9 * Ni + 16 * Cr + 29.1 * Si
    Bs = 830 - 270 * C - 90 * Mn - 37 * Ni - 70 * Cr
    return A1, Bs


def extract_from_state_data(state_data, C=0.82, Si=0.25, Mn=0.50, Cr=0.20, Ni=0.05):
    """
    从 worker 返回的扁平 state_data dict 中提取约束值（无需完整 SimulationState）。

    state_data 由 calculate_all_sim_T._worker_simulate 在 return_state=True 时产生，
    包含: time, T0, T1, pearlite_0_surface, ferrite_0_surface, ferrite_final_0, f_total_0

    化学成分参数用于计算临界温度 A1 / Bs。
    """
    A1, Bs = _compute_critical_temps(C, Si, Mn, Cr, Ni)

    T0 = np.asarray(state_data["T0"], dtype=np.float64)
    T1 = np.asarray(state_data["T1"], dtype=np.float64)
    t = np.asarray(state_data["time"], dtype=np.float64)
    pearl_hist = np.asarray(state_data["pearlite_0_surface"], dtype=np.float64)
    f_ferrite_final = float(state_data["ferrite_final_0"][-1])
    f_total = float(state_data["f_total_0"][-1])

    result = {}

    # 冷却速率
    cr = _cooling_rate(t, T0)

    # 珠光体温区 (A1 → Bs)
    mask_pearl = (T0[:-1] <= A1) & (T0[:-1] >= Bs)
    if mask_pearl.sum() >= 2:
        result["cr_pearl"] = float(np.mean(np.abs(cr[mask_pearl])))
    else:
        result["cr_pearl"] = None

    # 550°C 以下
    mask_low = T0[:-1] < 550.0
    if mask_low.sum() >= 2:
        result["cr_550"] = float(np.max(np.abs(cr[mask_low])))
    else:
        result["cr_550"] = 0.0

    # 相变后冷速
    if len(pearl_hist) > 0 and pearl_hist[-1] > 0.99:
        done_idx = int(np.argmax(pearl_hist > 0.99))
        if done_idx < len(cr):
            result["cr_post"] = float(np.mean(np.abs(cr[done_idx:])))
        else:
            result["cr_post"] = 0.0
    else:
        result["cr_post"] = 0.0

    # 珠光体转变量
    result["pearl_frac"] = float(pearl_hist[-1]) if len(pearl_hist) > 0 else 0.0

    # 先共析渗碳体分数（杠杆定律估算，非珠光体片层中的渗碳体）
    if f_ferrite_final < 0.01:
        cem_frac = max(0.0, (C - 0.77) / 5.9)
    else:
        cem_frac = 0.0
    result["cem_frac"] = cem_frac
    result["phase"] = "Ferrite" if f_ferrite_final > 0.01 else "Cementite"

    # 温差
    dT = np.abs(T0 - T1)
    result["max_dT_total"] = float(np.max(dT)) if len(dT) > 0 else 0.0
    if mask_pearl.sum() >= 2:
        result["max_dT_pearl"] = float(np.max(dT[:-1][mask_pearl]))
    else:
        result["max_dT_pearl"] = result["max_dT_total"]

    # 相变温度
    if len(pearl_hist) > 0 and pearl_hist[-1] > 0.5:
        half_idx = int(np.argmax(pearl_hist > 0.5))
        result["T_trans"] = float(T0[half_idx]) if half_idx < len(T0) else None
    else:
        result["T_trans"] = None

    # 片层间距估算
    cr_pearl_val = result.get("cr_pearl")
    if cr_pearl_val is not None and cr_pearl_val > 0:
        log_cr = np.log(max(cr_pearl_val, 0.5))
        log_S0 = np.log(0.197) - 0.38 * (log_cr - np.log(4.2))
        result["S0_est"] = float(np.exp(log_S0))
    else:
        result["S0_est"] = None

    return result


# ═══════════════════════════════════════════════════════════════
# 2. 预检查（无仿真，快速过滤明显不可行解）
# ═══════════════════════════════════════════════════════════════

def pre_check_bounds(params):
    """
    参数合法性预检查 —— 不运行仿真，快速拒绝越界候选解。

    参数：
        params: dict, 键: 'ORT', 'SPEED' (list[10]), 'FAN' (list[6])

    返回：
        (feasible: bool, penalty: float)
    """
    cfg = CONSTRAINT_CFG
    penalty = 0.0

    # H1: 吐丝温度
    ort = params.get("ORT")
    if ort is None or ort < cfg["ORT"][0] or ort > cfg["ORT"][1]:
        return False, 0.0

    # H6: 辊速范围
    speeds = params.get("SPEED", [])
    sp_lo, sp_hi = cfg["SPEED"]
    if any(s < sp_lo or s > sp_hi for s in speeds):
        return False, 0.0

    # H7: 相邻段辊速比硬上限（防止线材堆叠或拉断）
    for i in range(1, len(speeds)):
        if speeds[i-1] > 0 and speeds[i] > 0:
            r = max(speeds[i] / speeds[i-1], speeds[i-1] / speeds[i])
            if r > cfg["SPEED_RATIO_MAX"]:
                return False, 0.0

    # S5: 相邻段辊速跳跃（软约束），单次跳变惩罚上限 50
    for i in range(1, len(speeds)):
        if speeds[i-1] > 0 and speeds[i] > 0:
            ratio = abs(speeds[i] / speeds[i-1] - 1.0)
            if ratio > cfg["SPEED_JUMP"]:
                p = (ratio - cfg["SPEED_JUMP"]) ** 2 * 10.0
                penalty += min(p, 50.0)

    return True, penalty


# ═══════════════════════════════════════════════════════════════
# 3. 约束评价（需仿真状态 + ML 预测）
# ═══════════════════════════════════════════════════════════════

def evaluate_constraints(vals, pred_TS=None, pred_Z=None):
    """
    根据从 SimulationState 提取的物理量，施加硬约束/软约束/奖励项。

    参数：
        vals: dict, 来自 extract_from_state() 的输出
        pred_TS: float 或 None, ML 模型预测的抗拉强度 (MPa)
        pred_Z:  float 或 None, ML 模型预测的断面收缩率 (%)

    返回：
        (feasible: bool, penalty: float, bonus: float, details: dict)
        details 包含各约束项的拆分值，便于调试。
    """
    cfg = CONSTRAINT_CFG
    penalty = 0.0
    bonus = 0.0
    details = {}

    # ═══ 硬约束 ═══

    # H2: 珠光体温区冷却速率（软约束 —— 偏离理想区间 [9,12] 施加惩罚）
    cr_pearl = vals.get("cr_pearl")
    if cr_pearl is not None:
        cr_lo, cr_hi = cfg["CR_PEARL"]
        if cr_pearl < cr_lo:
            p = (cr_lo - cr_pearl) ** 2 * 5.0
            penalty += p
            details["H2_cr_pearl_low"] = p
        elif cr_pearl > cr_hi:
            p = (cr_pearl - cr_hi) ** 2 * 5.0
            penalty += p
            details["H2_cr_pearl_high"] = p
    else:
        # 未进入珠光体温区，给予固定惩罚
        penalty += 50.0
        details["H2_cr_pearl_none"] = 50.0

    # H3: 马氏体临界（550°C 以下冷速，软约束 —— 超标施加惩罚，不再拒绝）
    cr_550 = vals.get("cr_550", 0.0)
    if cr_550 >= cfg["CR_550"]:
        p = (cr_550 - cfg["CR_550"]) ** 2 * 3.0
        penalty += p
        details["H3_cr_550"] = p

    # H4: 珠光体转变量
    pearl_frac = vals.get("pearl_frac", 0.0)
    if pearl_frac < cfg["PEARL_FRAC"]:
        details["H4_pearl_frac"] = pearl_frac
        return False, 0.0, 0.0, details

    # ═══ 软约束 ═══

    # S1: 相变区温差
    dT_pearl = vals.get("max_dT_pearl", 0.0)
    if dT_pearl > cfg["DT_PEARL"]:
        p = (dT_pearl - cfg["DT_PEARL"]) ** 2 * 2.0
        penalty += p
        details["S1_dT_pearl"] = p

    # S2: 全程温差
    dT_total = vals.get("max_dT_total", 0.0)
    if dT_total > cfg["DT_MAX"]:
        p = (dT_total - cfg["DT_MAX"]) ** 2 * 1.0
        penalty += p
        details["S2_dT_total"] = p

    # S3: 相变后冷却速率
    cr_post = vals.get("cr_post", 0.0)
    if cr_post > cfg["CR_POST"]:
        p = (cr_post - cfg["CR_POST"]) ** 2 * 5.0
        penalty += p
        details["S3_cr_post"] = p

    # S4: 先共析渗碳体分数
    cem_frac = vals.get("cem_frac", 0.0)
    if cem_frac > cfg["CEM_FRAC"]:
        p = (cem_frac - cfg["CEM_FRAC"]) ** 2 * 20.0
        penalty += p
        details["S4_cem_frac"] = p

    # S7: TS 偏离目标
    if pred_TS is not None:
        ts_lo, ts_hi = cfg["TS"]
        if pred_TS < ts_lo:
            p = (ts_lo - pred_TS) ** 2 * 0.001
            penalty += p
            details["S7_TS_low"] = p
        elif pred_TS > ts_hi:
            p = (pred_TS - ts_hi) ** 2 * 0.001
            penalty += p
            details["S7_TS_high"] = p

    # S8: Z 低于标准
    if pred_Z is not None and pred_Z < cfg["Z_MIN"]:
        p = (cfg["Z_MIN"] - pred_Z) ** 2 * 5.0
        penalty += p
        details["S8_Z_low"] = p

    # ═══ 奖励项 ═══

    # B1: 相变温度接近 630 °C
    T_trans = vals.get("T_trans")
    if T_trans is not None and abs(T_trans - cfg["T_TRANS_TARGET"]) < cfg["T_TRANS_TOL"]:
        bonus += 5.0
        details["B1_T_trans"] = 5.0

    # B2: 珠光体片层间距在目标区间
    S0 = vals.get("S0_est")
    if S0 is not None and cfg["S0"][0] <= S0 <= cfg["S0"][1]:
        bonus += 3.0
        details["B2_S0"] = 3.0

    # B4: 索氏体率超越标准
    if pearl_frac >= cfg["PEARL_BONUS"]:
        bonus += 5.0
        details["B4_pearl_bonus"] = 5.0

    details["penalty"] = penalty
    details["bonus"] = bonus
    return True, penalty, bonus, details


# ═══════════════════════════════════════════════════════════════
# 4. 评分聚合
# ═══════════════════════════════════════════════════════════════

def compute_total_score(feasible, ek_penalty, ek_bonus, mp_cost=0.0, k_mp=None, k_ek=None):
    """
    聚合评分：Score = k_MP * MP_cost + k_EK * (EK_penalty - w_bonus * EK_bonus)

    奖励权重 w_bonus < 1，防止奖励项压倒惩罚项（如高惩罚+高奖励解优于低惩罚解）。

    参数：
        feasible: bool, 硬约束是否通过
        ek_penalty: float, 专家知识惩罚项
        ek_bonus: float, 专家知识奖励项
        mp_cost: float, 力学性能代价（暂为 0，待力学性能数据补充后启用）
        k_mp: float 或 None, 力学性能权重系数
        k_ek: float 或 None, 专家知识惩罚权重系数

    返回：
        float, 总分（不可行解返回 inf，越低越好）
    """
    if not feasible:
        return float("inf")

    _k_mp = k_mp if k_mp is not None else K_MP
    _k_ek = k_ek if k_ek is not None else K_EK

    ek_cost = ek_penalty - W_BONUS * ek_bonus
    return _k_mp * mp_cost + _k_ek * ek_cost
