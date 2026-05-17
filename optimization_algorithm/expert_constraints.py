"""
专家约束评价模块 —— 纯惩罚系统（v3）。

设计原则：
  1. 归一化偏差 d：d = 偏离专家知识边界的比例。相同 d → 相同"违规程度"
  2. 分级惩罚函数：T1 用 exp(k·d)-1（硬约束软化），T2/T3 用 d²（软约束）
  3. 纯惩罚评分：Score = penalty，无奖励项

使用方式：
  1. pre_check_bounds(params) → 预检查（速比约束，无仿真）
  2. extract_from_state(state, basic_info) → 从仿真状态提取约束值
  3. evaluate_constraints(vals, speeds, ort) → 约束评价（纯惩罚）
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════
# 分级权重配置
# ═══════════════════════════════════════════════════════════════

TIER_CFG = {
    "k":    2.0,    # T1 指数函数陡度
    "W_T1": 1.0,    # 致命级权重（exp 惩罚）
    "W_T2": 0.8,    # 关键级权重（d² 惩罚）
    "W_T3": 0.5,    # 辅助级权重（d² 惩罚）
}

# ═══════════════════════════════════════════════════════════════
# 约束规范定义
# ═══════════════════════════════════════════════════════════════

CONSTRAINT_SPECS = [
    # === T1 致命 — exp(k·d)-1 惩罚 ===
    {
        "name": "H2_cr_pearl",
        "tier": 1,
        "source": "vals",
        "key": "cr_pearl",
        "type": "bilateral",
        "lo": 9.0, "hi": 12.0,
        "norm_lo": 9.0, "norm_hi": 12.0,
    },
    {
        "name": "H3_cr_post",
        "tier": 2,
        "source": "vals",
        "key": "cr_550_segs",
        "type": "unilateral_upper",
        "limit": 5.0,
        "norm": 5.0,
        "per_segment": True,
    },
    {
        "name": "H4_pearl_frac",
        "tier": 1,
        "source": "vals",
        "key": "pearl_frac",
        "type": "unilateral_lower",
        "limit": 0.85,
        "norm": 0.85,
    },
    {
        "name": "S9_cr_stage1",
        "tier": 1,
        "source": "vals",
        "key": "cr_stage1_segs",
        "type": "bilateral",
        "lo": 20.0, "hi": 30.0,
        "norm_lo": 20.0, "norm_hi": 30.0,
        "per_segment": True,
    },

    # === T2 关键 — d² 惩罚 ===
    {
        "name": "S4_cem_frac",
        "tier": 2,
        "source": "vals",
        "key": "cem_frac",
        "type": "unilateral_upper",
        "limit": 0.012,
        "norm": 0.012,
    },
    {
        "name": "S13_cr_lowT",
        "tier": 2,
        "source": "vals",
        "key": "cr_lowT_segs",
        "type": "unilateral_upper",
        "limit": 5.0,
        "norm": 5.0,
        "per_segment": True,
    },
    {
        "name": "S5a_speed_decrease",
        "tier": 1,
        "source": "speed_decrease",
        "tol": 0.05,
        "norm": 0.05,
    },
    {
        "name": "S5b_speed_increase",
        "tier": 2,
        "source": "speed_increase",
        "tol": 0.20,
        "norm": 0.20,
    },
    {
        "name": "S1_dT_pearl",
        "tier": 2,
        "source": "vals",
        "key": "max_dT_pearl",
        "type": "unilateral_upper",
        "limit": 15.0,
        "norm": 15.0,
    },
    {
        "name": "S2_dT_total",
        "tier": 2,
        "source": "vals",
        "key": "max_dT_total",
        "type": "unilateral_upper",
        "limit": 20.0,
        "norm": 20.0,
    },
    {
        "name": "S10_time",
        "tier": 2,
        "source": "vals",
        "key": "time_total",
        "type": "bilateral",
        "lo": 75.0, "hi": 85.0,
        "norm_lo": 75.0, "norm_hi": 85.0,
    },

    # === T3 辅助 — d² 惩罚 ===
    {
        "name": "S11_T_end",
        "tier": 3,
        "source": "vals",
        "key": "T_end",
        "type": "unilateral_upper",
        "limit": 400.0,
        "norm": 400.0,
    },
    {
        "name": "S12_speed_dev",
        "tier": 3,
        "source": "speed_deviation",
        "ref": [1.173, 1.270, 1.369, 1.427, 1.423, 1.453, 1.450, 1.453, 1.459, 1.556],
        "tol": 0.08,
    },
    {
        "name": "S14_ort_dev",
        "tier": 3,
        "source": "ort",
        "target": 910.0,
        "norm": 20.0,
    },
]

# ═══════════════════════════════════════════════════════════════
# 1. 归一化偏差 + 分级惩罚
# ═══════════════════════════════════════════════════════════════

def normalized_deviation(value, spec):
    """计算归一化偏差 d。d=0 表示满足约束，d>0 表示违规。"""
    if spec["type"] == "unilateral_upper":
        return max(0.0, value - spec["limit"]) / spec["norm"]
    elif spec["type"] == "unilateral_lower":
        return max(0.0, spec["limit"] - value) / spec["norm"]
    elif spec["type"] == "bilateral":
        if value < spec["lo"]:
            return max(0.0, spec["lo"] - value) / spec["norm_lo"]
        elif value > spec["hi"]:
            return max(0.0, value - spec["hi"]) / spec["norm_hi"]
        return 0.0
    return 0.0


def apply_penalty(d, tier):
    """分级惩罚函数。T1 用 exp，T2/T3 用 d²。"""
    if d <= 0:
        return 0.0
    if tier == 1:
        return TIER_CFG["W_T1"] * (np.exp(TIER_CFG["k"] * d) - 1.0)
    elif tier == 2:
        return TIER_CFG["W_T2"] * d ** 2
    elif tier == 3:
        return TIER_CFG["W_T3"] * d ** 2
    return 0.0


# ═══════════════════════════════════════════════════════════════
# 2. 从 SimulationState 提取约束值（不变）
# ═══════════════════════════════════════════════════════════════

def extract_from_state(state, basic_info, roll_start_time=None):
    """从仿真完成的 SimulationState 中提取所有约束相关物理量。"""
    T0 = np.array(state.history_T_0[-1], dtype=np.float64)
    T1 = np.array(state.history_T_1[-1], dtype=np.float64)
    t = np.array(state.history_time, dtype=np.float64)
    Bs = float(basic_info.Bs)

    result = {}

    pearl_hist = np.array(state.pearlite_0[-1], dtype=np.float64)
    if len(pearl_hist) > 0 and pearl_hist[-1] > 0.99:
        t_trans_start = float(t[int(np.argmax(pearl_hist > 0.01))])
        t_trans_end = float(t[int(np.argmax(pearl_hist > 0.99))])
    else:
        t_trans_start = t[-1]
        t_trans_end = t[-1]

    below_Bs = np.where(T0 <= Bs)[0]
    t_Bs = float(t[below_Bs[0]]) if len(below_Bs) > 0 else t[-1]

    def _seg_cooling_rates(segments_T, segments_t, stat="mean"):
        all_rates = []
        for seg_T, seg_tm in zip(segments_T, segments_t):
            if len(seg_T) < 2:
                continue
            dt = np.diff(seg_tm)
            dT = np.diff(seg_T)
            with np.errstate(divide='ignore', invalid='ignore'):
                rates = np.abs(dT / np.where(dt > 0, dt, np.inf))
            all_rates.extend(rates.tolist())
        if not all_rates:
            return None
        return float(np.mean(all_rates)) if stat == "mean" else float(np.max(all_rates))

    if roll_start_time is not None and len(roll_start_time) >= 2:
        n_seg = len(roll_start_time) - 1
        seg_idx = np.searchsorted(roll_start_time, t, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, n_seg - 1)

        k = int(seg_idx[np.searchsorted(t, t_trans_start)]) if t_trans_start < t[-1] else 0
        m = int(seg_idx[np.searchsorted(t, t_Bs)]) if t_Bs < t[-1] else n_seg - 1

        def _seg_T_t(seg_range):
            segs_T, segs_t = [], []
            for s in seg_range:
                mask = seg_idx == s
                if mask.sum() >= 2:
                    segs_T.append(T0[mask])
                    segs_t.append(t[mask])
            return segs_T, segs_t

        s9_T, s9_t = _seg_T_t(range(0, k))
        h2_T, h2_t = _seg_T_t([k])
        h3_T, h3_t = _seg_T_t(range(k + 1, m + 1))
        s13_T, s13_t = _seg_T_t(range(m + 1, n_seg))

        result["cr_stage1_segs"] = [_seg_cooling_rates([T], [tm], "mean") for T, tm in zip(s9_T, s9_t)]
        result["cr_pearl_segs"] = [_seg_cooling_rates([T], [tm], "mean") for T, tm in zip(h2_T, h2_t)]
        result["cr_550_segs"] = [_seg_cooling_rates([T], [tm], "mean") or 0.0 for T, tm in zip(h3_T, h3_t)]
        result["cr_lowT_segs"] = [_seg_cooling_rates([T], [tm], "mean") or 0.0 for T, tm in zip(s13_T, s13_t)]

        result["cr_stage1"] = float(np.mean([c for c in result["cr_stage1_segs"] if c is not None])) if result["cr_stage1_segs"] else None
        result["cr_pearl"] = result["cr_pearl_segs"][0] if result["cr_pearl_segs"] else None
        result["cr_550"] = float(np.max([c for c in result["cr_550_segs"] if c is not None])) if result["cr_550_segs"] else 0.0
        result["cr_lowT"] = float(np.max([c for c in result["cr_lowT_segs"] if c is not None])) if result["cr_lowT_segs"] else 0.0

        dT = np.abs(T0 - T1)
        h2_mask = seg_idx == k
        result["max_dT_pearl"] = float(np.max(dT[h2_mask])) if h2_mask.sum() > 0 else float(np.max(dT))

        result["_trans_seg"] = k
        result["_Bs_seg"] = m
        result["_n_seg"] = n_seg
    else:
        t_start_idx = int(np.searchsorted(t, t_trans_start))
        t_end_idx = int(np.searchsorted(t, t_trans_end))

        def _simple_cr(seg_T, seg_t, stat="mean"):
            if len(seg_T) < 2: return None
            dt = np.diff(seg_t); dT = np.diff(seg_T)
            with np.errstate(divide='ignore', invalid='ignore'):
                rates = np.abs(dT / np.where(dt > 0, dt, np.inf))
            return float(np.mean(rates)) if stat=="mean" else float(np.max(rates))

        result["cr_stage1"] = _simple_cr(T0[:t_start_idx+1], t[:t_start_idx+1], "mean")
        result["cr_pearl"] = _simple_cr(T0[t_start_idx:t_end_idx+1], t[t_start_idx:t_end_idx+1], "mean") if t_end_idx>t_start_idx else None
        post_T = T0[t_end_idx:]; post_t = t[t_end_idx:]
        result["cr_550"] = _simple_cr(post_T, post_t, "max") or 0.0
        result["cr_lowT"] = 0.0

        dT = np.abs(T0 - T1)
        result["max_dT_pearl"] = float(np.max(dT[t_start_idx:t_end_idx+1])) if t_end_idx > t_start_idx else float(np.max(dT))

    result["time_total"] = float(t[-1]) if len(t) > 0 else 0.0
    result["T_end"] = float(T0[-1]) if len(T0) > 0 else 0.0

    dT_all = np.abs(T0 - T1)
    result["max_dT_total"] = float(np.max(dT_all)) if len(dT_all) > 0 else 0.0

    result["pearl_frac"] = float(pearl_hist[-1]) if len(pearl_hist) > 0 else 0.0

    f_ferrite = float(state.ferrite_final_0[-1])
    if f_ferrite < 0.01:
        C_pct = basic_info.ELM_C * 100
        cem_frac = max(0.0, (C_pct - 0.77) / 5.9)
    else:
        cem_frac = 0.0
    result["cem_frac"] = cem_frac

    if len(pearl_hist) > 0 and pearl_hist[-1] > 0.5:
        half_idx = int(np.argmax(pearl_hist > 0.5))
        result["T_trans"] = float(T0[half_idx]) if half_idx < len(T0) else None
    else:
        result["T_trans"] = None

    cr_pearl_val = result.get("cr_pearl")
    if cr_pearl_val is not None and cr_pearl_val > 0:
        log_cr = np.log(max(cr_pearl_val, 0.5))
        log_S0 = np.log(0.197) - 0.38 * (log_cr - np.log(4.2))
        result["S0_est"] = float(np.exp(log_S0))
    else:
        result["S0_est"] = None

    return result


def extract_from_state_data(state_data, C=0.82, Si=0.25, Mn=0.50, Cr=0.20, Ni=0.05):
    """从 worker 返回的扁平 state_data dict 中提取约束值。"""
    T0 = np.asarray(state_data["T0"], dtype=np.float64)
    T1 = np.asarray(state_data["T1"], dtype=np.float64)
    t = np.asarray(state_data["time"], dtype=np.float64)
    pearl_hist = np.asarray(state_data["pearlite_0_surface"], dtype=np.float64)
    f_ferrite_final = float(state_data["ferrite_final_0"][-1])
    roll_rt = state_data.get("roll_start_time", None)

    Bs = 830.0 - 270.0 * C - 90.0 * Mn - 37.0 * Ni - 70.0 * Cr

    result = {}

    if len(pearl_hist) > 0 and pearl_hist[-1] > 0.99:
        t_trans_start = float(t[int(np.argmax(pearl_hist > 0.01))])
        t_trans_end = float(t[int(np.argmax(pearl_hist > 0.99))])
    else:
        t_trans_start = t[-1]
        t_trans_end = t[-1]

    below_Bs = np.where(T0 <= Bs)[0]
    t_Bs = float(t[below_Bs[0]]) if len(below_Bs) > 0 else t[-1]

    def _seg_cooling_rates(segments_T, segments_t, stat="mean"):
        all_rates = []
        for seg_T, seg_tm in zip(segments_T, segments_t):
            if len(seg_T) < 2: continue
            dt = np.diff(seg_tm); dT = np.diff(seg_T)
            with np.errstate(divide='ignore', invalid='ignore'):
                rates = np.abs(dT / np.where(dt > 0, dt, np.inf))
            all_rates.extend(rates.tolist())
        if not all_rates: return None
        return float(np.mean(all_rates)) if stat=="mean" else float(np.max(all_rates))

    if roll_rt is not None and len(roll_rt) >= 2:
        n_seg = len(roll_rt) - 1
        seg_idx = np.searchsorted(roll_rt, t, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, n_seg - 1)

        k = int(seg_idx[np.searchsorted(t, t_trans_start)]) if t_trans_start < t[-1] else 0
        m = int(seg_idx[np.searchsorted(t, t_Bs)]) if t_Bs < t[-1] else n_seg - 1

        def _seg_T_t(seg_range):
            segs_T, segs_t = [], []
            for s in seg_range:
                mask = seg_idx == s
                if mask.sum() >= 2: segs_T.append(T0[mask]); segs_t.append(t[mask])
            return segs_T, segs_t

        s9_T, s9_t = _seg_T_t(range(0, k))
        h2_T, h2_t = _seg_T_t([k])
        h3_T, h3_t = _seg_T_t(range(k+1, m+1))
        s13_T, s13_t = _seg_T_t(range(m+1, n_seg))

        result["cr_stage1_segs"] = [_seg_cooling_rates([T], [tm], "mean") for T, tm in zip(s9_T, s9_t)]
        result["cr_pearl_segs"] = [_seg_cooling_rates([T], [tm], "mean") for T, tm in zip(h2_T, h2_t)]
        result["cr_550_segs"] = [_seg_cooling_rates([T], [tm], "mean") or 0.0 for T, tm in zip(h3_T, h3_t)]
        result["cr_lowT_segs"] = [_seg_cooling_rates([T], [tm], "mean") or 0.0 for T, tm in zip(s13_T, s13_t)]

        result["cr_stage1"] = float(np.mean([c for c in result["cr_stage1_segs"] if c is not None])) if result["cr_stage1_segs"] else None
        result["cr_pearl"] = result["cr_pearl_segs"][0] if result["cr_pearl_segs"] else None
        result["cr_550"] = float(np.max([c for c in result["cr_550_segs"] if c is not None])) if result["cr_550_segs"] else 0.0
        result["cr_lowT"] = float(np.max([c for c in result["cr_lowT_segs"] if c is not None])) if result["cr_lowT_segs"] else 0.0

        dT = np.abs(T0 - T1)
        h2_mask = seg_idx == k
        result["max_dT_pearl"] = float(np.max(dT[h2_mask])) if h2_mask.sum() > 0 else float(np.max(dT))
    else:
        t_start_idx = int(np.searchsorted(t, t_trans_start))
        t_end_idx = int(np.searchsorted(t, t_trans_end))

        def _s(seg_T, seg_t, s="mean"):
            if len(seg_T)<2: return None
            dt=np.diff(seg_t); dT=np.diff(seg_T)
            with np.errstate(divide='ignore', invalid='ignore'):
                r=np.abs(dT/np.where(dt>0,dt,np.inf))
            return float(np.mean(r)) if s=="mean" else float(np.max(r))

        result["cr_stage1"] = _s(T0[:t_start_idx+1], t[:t_start_idx+1], "mean")
        result["cr_pearl"] = _s(T0[t_start_idx:t_end_idx+1], t[t_start_idx:t_end_idx+1], "mean") if t_end_idx>t_start_idx else None
        result["cr_550"] = _s(T0[t_end_idx:], t[t_end_idx:], "max") or 0.0
        result["cr_lowT"] = 0.0
        dT = np.abs(T0 - T1)
        result["max_dT_pearl"] = float(np.max(dT[t_start_idx:t_end_idx+1])) if t_end_idx>t_start_idx else float(np.max(dT))

    result["time_total"] = float(t[-1]) if len(t) > 0 else 0.0
    result["T_end"] = float(T0[-1]) if len(T0) > 0 else 0.0
    result["max_dT_total"] = float(np.max(np.abs(T0 - T1))) if len(T0) > 0 else 0.0
    result["pearl_frac"] = float(pearl_hist[-1]) if len(pearl_hist) > 0 else 0.0

    if f_ferrite_final < 0.01:
        cem_frac = max(0.0, (C - 0.77) / 5.9)
    else:
        cem_frac = 0.0
    result["cem_frac"] = cem_frac

    if len(pearl_hist) > 0 and pearl_hist[-1] > 0.5:
        half_idx = int(np.argmax(pearl_hist > 0.5))
        result["T_trans"] = float(T0[half_idx]) if half_idx < len(T0) else None
    else:
        result["T_trans"] = None

    cr_pearl_val = result.get("cr_pearl")
    if cr_pearl_val is not None and cr_pearl_val > 0:
        log_cr = np.log(max(cr_pearl_val, 0.5))
        log_S0 = np.log(0.197) - 0.38 * (log_cr - np.log(4.2))
        result["S0_est"] = float(np.exp(log_S0))
    else:
        result["S0_est"] = None

    return result


# ═══════════════════════════════════════════════════════════════
# 3. 约束评价（纯惩罚，spec 驱动）
# ═══════════════════════════════════════════════════════════════

def evaluate_constraints(vals, speeds=None, ort=None):
    """
    对所有约束逐条评价，返回总惩罚。

    参数：
        vals:   dict, extract_from_state / extract_from_state_data 输出
        speeds: list[float] 或 None, 10 段辊速（用于 S5/S12）
        ort:    float 或 None, 吐丝温度（用于 S14）

    返回：
        (penalty: float, details: dict)
    """
    penalty = 0.0
    details = {}

    for spec in CONSTRAINT_SPECS:
        p = _eval_spec(spec, vals, speeds, ort)
        if p > 0:
            penalty += p
            details[spec["name"]] = p

    return penalty, details


def _eval_spec(spec, vals, speeds, ort):
    """对单个 spec 计算惩罚值。"""
    src = spec["source"]

    if src == "vals":
        value = vals.get(spec["key"])
        if value is None:
            return 0.0
        if spec.get("per_segment"):
            total = 0.0
            for seg_val in value:
                if seg_val is not None:
                    d = normalized_deviation(seg_val, spec)
                    total += apply_penalty(d, spec["tier"])
            return total
        else:
            d = normalized_deviation(value, spec)
            return apply_penalty(d, spec["tier"])

    elif src == "speed_ratio":
        # 对称速比（保留兼容）
        if speeds is None:
            return 0.0
        total = 0.0
        for i in range(1, len(speeds)):
            if speeds[i-1] > 0 and speeds[i] > 0:
                ratio_dev = abs(speeds[i] / speeds[i-1] - 1.0)
                d = max(0.0, ratio_dev - 0.25) / 0.25
                total += apply_penalty(d, spec["tier"])
        return total

    elif src == "speed_decrease":
        # 非对称降速（T1 exp 惩罚，容忍 5%）
        if speeds is None:
            return 0.0
        total = 0.0
        tol = spec["tol"]
        norm = spec["norm"]
        for i in range(1, len(speeds)):
            if speeds[i-1] > 0 and speeds[i] > 0:
                ratio = speeds[i] / speeds[i-1]
                if ratio < 1.0:
                    d = max(0.0, (1.0 - ratio) - tol) / norm
                    total += apply_penalty(d, spec["tier"])
        return total

    elif src == "speed_increase":
        # 非对称升速（T2 d² 惩罚，容忍 20%）
        if speeds is None:
            return 0.0
        total = 0.0
        tol = spec["tol"]
        norm = spec["norm"]
        for i in range(1, len(speeds)):
            if speeds[i-1] > 0 and speeds[i] > 0:
                ratio = speeds[i] / speeds[i-1]
                if ratio > 1.0:
                    d = max(0.0, (ratio - 1.0) - tol) / norm
                    total += apply_penalty(d, spec["tier"])
        return total

    elif src == "speed_deviation":
        if speeds is None:
            return 0.0
        total = 0.0
        ref = spec["ref"]
        tol = spec["tol"]
        for s, r in zip(speeds, ref):
            dev = abs(s - r)
            d = max(0.0, dev - tol) / tol
            total += apply_penalty(d, spec["tier"])
        return total

    elif src == "ort":
        if ort is None:
            return 0.0
        d = abs(ort - spec["target"]) / spec["norm"]
        return apply_penalty(d, spec["tier"])

    return 0.0


# ═══════════════════════════════════════════════════════════════
# 4. 预检查（速比约束，无仿真）
# ═══════════════════════════════════════════════════════════════

def pre_check_bounds(params):
    """
    预检查：非对称速比约束（降速 T1 exp，升速 T2 d²）。

    返回：penalty: float
    """
    speeds = params.get("SPEED", [])
    if not speeds:
        return 0.0

    penalty = 0.0
    for i in range(1, len(speeds)):
        if speeds[i-1] > 0 and speeds[i] > 0:
            ratio = speeds[i] / speeds[i-1]
            if ratio < 1.0:
                d = max(0.0, (1.0 - ratio) - 0.05) / 0.05
                penalty += apply_penalty(d, 1)  # T1 exp
            elif ratio > 1.0:
                d = max(0.0, (ratio - 1.0) - 0.20) / 0.20
                penalty += apply_penalty(d, 2)  # T2 d²

    return penalty


# ═══════════════════════════════════════════════════════════════
# 5. 评分聚合
# ═══════════════════════════════════════════════════════════════

def compute_total_score(penalty):
    """纯惩罚系统：Score = penalty。"""
    return float(penalty)
