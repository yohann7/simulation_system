"""
82A 卷帘钢斯太尔摩冷却线 —— 温度与组织相变仿真系统。

本模块提供单次温度仿真和组织成份仿真的全部核心功能函数，包括：
- 物理参数计算（导热系数、比热容、相变潜热）
- 换热系数计算（对流 + 辐射，含佳灵装置模型）
- 相变孕育期与 JMAK 动力学
- 相变热焓计算
- 有限差分温度场求解
- 仿真结果可视化

供其他模块调用的核心函数见模块末尾的 __all__ 列表。
"""

import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════
# 基础数据类（工艺参数、修正系数、辊道信息）
# ═══════════════════════════════════════════════════════════════


class basic_info:
    """基础工艺参数，不随仿真段变化。"""

    T_air = Ta = 25  # 环境温度，℃
    a = 29.228
    steel_grade = "82A"

    rho = 7823  # 密度，kg/m³
    phi = 5.5  # 线材直径，mm
    r = phi * 1e-3 / 2  # 线材半径，m

    ELM_C = 0.82 / 100  # 碳含量
    ELM_SI = 0.21 / 100  # 硅含量
    ELM_MN = 0.52 / 100  # 锰含量
    ELM_NI = 0.009858 / 100
    ELM_CR = 0.011815 / 100

    v_wire = 100  # 吐丝速度
    D_ring = 1.05  # 线环直径，m（A线1.05，B线1.075）

    Acm = 727 + 314.2 * (ELM_C * 100 - 0.77)  # 过共析钢上临界点
    A1 = 727 - 10.7 * (ELM_MN * 100) - 16.9 * (ELM_NI * 100) + 16 * (ELM_CR * 100) + 29.1 * (ELM_SI * 100)
    Bs = 830 - 270 * (ELM_C * 100) - 90 * (ELM_MN * 100) - 37 * (ELM_NI * 100) - 70 * (ELM_CR * 100)


class parameter_change:
    """仿真修正参数集。

    供 change_parameter_82A.py 优化使用，也供 sim_T 仿真直接加载。
    """

    def __init__(self, num):
        self.num = num
        self.xs_hc0 = 1.0  # 非搭接点对流换热修正
        self.xs_hc1 = 1.0  # 搭接点对流换热修正
        self.view_factor = 1.0  # 搭接点辐射遮挡修正
        self.xs_tauf = 1.0  # 铁素体孕育期修正
        self.xs_taup = 1.0  # 珠光体孕育期修正
        self.xs_dqp = 1.0  # 珠光体相变热焓修正
        self.xs_dqf = 1.0  # 铁素体相变热焓修正
        self.result = 0  # 结果评价指标，0-1 之间


class roll:
    """单段辊道参数。"""

    def __init__(self, roll_num):
        self.roll_num = roll_num
        self.roll_name = "name_undefined"
        self.roll_length = 9.252  # m
        self.roll_v = 0.97  # m/s
        self.t = self.roll_length / (self.roll_v / 60)  # 停留时间，s
        self.step = 0  # 仿真步数（初始化时根据 dt 计算）
        self.fan_air_volume = 0  # 风机风量，m³/s
        self.fan_status = 1  # 风机开度，0-1
        self.fan_area = 1.5 * 9.252 / 2  # 风机面积
        self.fan_speed = 0  # 风机风速，m/s
        self.thermal_cove = 0  # 保温罩状态
        self.optiflex_angle = 0  # 佳灵装置开合角度，deg
        self.pre_temp_0 = None  # 入口非搭接点温度
        self.pre_temp_1 = None  # 入口搭接点温度
        self.post_temp_0 = None  # 出口非搭接点温度
        self.post_temp_1 = None  # 出口搭接点温度


class data_loader:
    """辊道工艺数据加载。"""

    @staticmethod
    def load_roll_data():
        """加载斯太尔摩冷却线 26 段辊道参数。"""
        # 被调用: calculate_all_sim_T.py, change_parameter_82A.py, calculate_the_effect.py
        n = 26
        rolls = [roll(i) for i in range(n)]

        # --- IN 入口段 ---
        rolls[0].roll_name = "IN"
        rolls[0].t = 3.0
        rolls[0].roll_length = 3
        rolls[0].roll_v = 1
        rolls[0].fan_status = 0

        # --- 风机段 1-8 (含佳灵装置) ---
        # 前 4 台风机: optiflex_angle = 3.0°
        # 后 4 台风机: optiflex_angle = 1.5°
        fan_config = [
            # (idx, name, v, t_half, status, optiflex)
            (1, "1-1", 1.1, 8.411, 0.99, 3.0),
            (2, "1-2", 1.1, 8.411, 0.99, 3.0),
            (3, "2-1", 1.199, 7.716, 0.95, 3.0),
            (4, "2-2", 1.199, 7.716, 0.95, 3.0),
            (5, "3-1", 1.283, 7.211, 0, 3.0),
            (6, "3-2", 1.283, 7.211, 0, 3.0),
            (7, "4-1", 1.347, 6.869, 0, 3.0),
            (8, "4-2", 1.347, 6.869, 0, 3.0),
            (9, "5-1", 1.347, 6.869, 0, 1.5),
            (10, "5-2", 1.347, 6.869, 0, 1.5),
            (11, "6-1", 1.428, 6.479, 0, 1.5),
            (12, "6-2", 1.428, 6.479, 0, 1.5),
            (13, "7-1", 1.428, 6.479, 0, 1.5),
            (14, "7-2", 1.428, 6.479, 0, 1.5),
            (15, "8-1", 1.428, 6.479, 0, 1.5),
            (16, "8-2", 1.428, 6.479, 0, 1.5),
        ]
        for idx, name, v, t_half, status, optiflex in fan_config:
            r = rolls[idx]
            r.roll_name = name
            r.roll_length = 9.252 / 2
            r.roll_v = v
            r.t = t_half / 2
            r.fan_air_volume = 53.889
            r.fan_status = status
            r.fan_speed = r.fan_air_volume * r.fan_status / r.fan_area
            r.optiflex_angle = optiflex

        # --- 后段 (风机全关) ---
        post_config = [
            (17, "9-1", 1.428, 6.479),
            (18, "9-2", 1.428, 6.479),
            (19, "10-1", 1.428, 6.479),
            (20, "10-2", 1.428, 6.479),
            (21, "11-1", 1.228, 7.534),
            (22, "11-2", 1.228, 7.534),
            (23, "12-1", 1.007, 9.188),
            (24, "12-2", 1.007, 9.188),
        ]
        for idx, name, v, t_half in post_config:
            r = rolls[idx]
            r.roll_name = name
            r.roll_length = 9.252 / 2
            r.roll_v = v
            r.t = t_half / 2
            r.fan_air_volume = 0
            r.fan_status = 0

        # --- OUT 出口段 ---
        rolls[25].roll_name = "OUT"
        rolls[25].roll_length = 3
        rolls[25].roll_v = 0.906
        rolls[25].t = 3.311
        rolls[25].fan_air_volume = 0
        rolls[25].fan_status = 0

        # 根据仿真步长计算每段步数
        for r in rolls:
            r.step = int(r.t / _default_dt)

        return rolls, n


# ═══════════════════════════════════════════════════════════════
# 参数文件读写
# ═══════════════════════════════════════════════════════════════

PARAMETER_FILE = Path(__file__).with_name("parameter.txt")
PARAMETER_KEYS = [
    "xs_hc0", "xs_hc1", "view_factor",
    "xs_tauf", "xs_taup", "xs_dqp", "xs_dqf",
]

# 默认仿真步长（供 data_loader 使用，运行时可通过参数覆盖）
_default_dt = 0.01


def set_default_dt(dt):
    """设置全局默认仿真步长，影响 data_loader.load_roll_data() 中的 step 计算。"""
    global _default_dt
    _default_dt = float(dt)


def _parse_parameter_lines(lines):
    """解析 parameter.txt 的内容行。"""
    params = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        params[key.strip()] = float(value.strip())
    return params


def read_parameter_file(file_path=PARAMETER_FILE):
    """读取 parameter.txt 并返回参数字典。"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"parameter.txt not found at {path}. Run change_parameter_82A.py to generate it."
        )
    content = path.read_text(encoding="utf-8").splitlines()
    return _parse_parameter_lines(content)


def build_parameter_from_file(num=0, file_path=PARAMETER_FILE):
    """从 parameter.txt 构建 parameter_change 对象。

    被调用: calculate_all_sim_T.py, change_parameter_82A.py, calculate_the_effect.py
    """
    params_dict = read_parameter_file(file_path)
    param_obj = parameter_change(num)
    for name in PARAMETER_KEYS:
        if name in params_dict:
            setattr(param_obj, name, float(params_dict[name]))
    return param_obj


# ═══════════════════════════════════════════════════════════════
# 仿真状态（单次仿真中所有可变状态）
# ═══════════════════════════════════════════════════════════════


class SimulationState:
    """封装单次温度仿真的全部可变状态。

    每个 SimulationState 实例代表一次独立仿真，支持多实例并行。
    """

    def __init__(self, N=50, dt=0.01):
        self.N = N
        self.dt = dt
        self.dr = basic_info.r / N
        self.current_params = None  # parameter_change 对象或 None
        self._init_arrays()

    def _init_arrays(self):
        """初始化所有仿真状态数组。"""
        n = self.N

        # 总转变量
        self.f_total_0 = np.zeros(n)
        self.f_total_1 = np.zeros(n)

        # 铁素体孕育期
        self.suma_f0 = np.zeros(n)
        self.mark_sf_0 = np.zeros(n, dtype=int)
        self.mark_ef_0 = np.zeros(n, dtype=int)
        self.suma_f1 = np.zeros(n)
        self.mark_sf_1 = np.zeros(n, dtype=int)
        self.mark_ef_1 = np.zeros(n, dtype=int)

        # 珠光体孕育期
        self.suma_p0 = np.zeros(n)
        self.mark_sp_0 = np.zeros(n, dtype=int)
        self.mark_ep_0 = np.zeros(n, dtype=int)
        self.suma_p1 = np.zeros(n)
        self.mark_sp_1 = np.zeros(n, dtype=int)
        self.mark_ep_1 = np.zeros(n, dtype=int)

        # 历史记录
        self.history_time = []
        self.history_T_0 = [[] for _ in range(n)]
        self.history_T_1 = [[] for _ in range(n)]
        self.history_Q_0 = [[] for _ in range(n)]
        self.history_Q_1 = [[] for _ in range(n)]
        self.history_h_0 = [[] for _ in range(3)]
        self.history_h_1 = [[] for _ in range(3)]
        self.pearlite_0 = [[] for _ in range(n)]
        self.pearlite_1 = [[] for _ in range(n)]
        self.ferrite_0 = [[] for _ in range(n)]
        self.ferrite_1 = [[] for _ in range(n)]
        self.ferrite_final_0 = np.zeros(n)
        self.ferrite_final_1 = np.zeros(n)
        self.ferrite_recorded_0 = np.zeros(n, dtype=bool)
        self.ferrite_recorded_1 = np.zeros(n, dtype=bool)

    def reset(self):
        """重置仿真状态，保留 N/dt 等结构参数。"""
        self._init_arrays()


# ═══════════════════════════════════════════════════════════════
# 物理参数计算
# ═══════════════════════════════════════════════════════════════


def calculate_physical_parameters(T_current):
    """根据当前温度计算随温度变化的物理参数。

    返回: (导热系数 k [W/(m·K)], 比热容 c [J/(kg·K)], 珠光体潜热 Hap [J/kg])
    参考: Yafei S. et al. (2009) CCDC.
    """
    C_pct = basic_info.ELM_C * 100

    # 导热系数
    k = 57.4 - 0.0237 * T_current - 10.3 * C_pct - 1.84e-5 * T_current**2 + 0.0108 * T_current * C_pct

    # 比热容
    c = 499 + 0.0006 * T_current**2 + 0.000892 * (T_current / C_pct) - 2.61 * (1 / C_pct)

    # 珠光体相变潜热
    Hap = 120848 - 52.42 * T_current - 0.158 * T_current * T_current

    return k, c, Hap


def calculate_optiflex_parameters(optiflex_angle):
    """计算佳灵装置 (Optiflex) 的风速分配修正因子。

    参考: Deng T.W. et al. (2025), J. Mater. Eng. Perform., Vol.34, pp.11212-11225.
          DOI: 10.1007/s11665-024-09898-2

    参数:
        optiflex_angle: 佳灵装置开合角度 (deg)，0=关闭，范围 0-10

    返回:
        (搭接点风速因子 >=1, 非搭接点风速因子 <=1)
    """
    angle = max(0.0, min(10.0, optiflex_angle))

    # 风口宽度与开合角度的线性关系 (基于论文 Table 10 数据拟合)
    w_lap_0 = 325.0  # 0° 时搭接风口宽，mm
    w_nonlap_0 = 780.0  # 0° 时非搭接风口宽，mm
    dw_dtheta = 11.113  # 斜率，mm/deg

    w_lap = w_lap_0 + dw_dtheta * angle
    w_nonlap = w_nonlap_0 - 2.0 * dw_dtheta * angle

    return w_lap / w_lap_0, w_nonlap / w_nonlap_0


def _get_params_attr(current_params, attr_name, default=1.0):
    """安全获取修正参数值。"""
    if current_params is not None:
        return getattr(current_params, attr_name, default)
    return default


def calculate_heat_transfer(
    T_0_surface, T_1_surface, T_air, phi,
    fan_air_volume, fan_status, fan_area,
    optiflex_angle=0, current_params=None,
):
    """计算风冷线换热系数（对流 + 辐射）。

    包含:
    - 佳灵装置风速分配
    - 搭接点被动换热折减 (blocking_factor)
    - Žukauskas 强制对流 + Churchill & Chu 自然对流
    - 辐射换热（Stefan-Boltzmann 线性化）

    返回: (h_conv_0, h_conv_1, h_rad_0, h_rad_1)
    """
    epsilon = 1e-6

    # 标度化温度（用于辐射公式）
    T_air_K_sc = (T_air + 273.15) / 100.0
    T_0_K_sc = (T_0_surface + 273.15) / 100.0
    T_1_K_sc = (T_1_surface + 273.15) / 100.0

    # --- 空气物性 (Sutherland 定律) ---
    def _k_air(T_k):
        return 2.495e-3 * (T_k**1.5) / (T_k + 194.0)

    def _vair(T_k):
        return 4.13e-9 * (T_k**2.5) / (T_k + 110.4)

    # --- 流场 ---
    u_avg = fan_air_volume * fan_status / fan_area

    # 搭接点被动换热折减
    blocking_factor = 0.88

    # 佳灵装置风速分配
    if optiflex_angle > 0:
        wf_lap, wf_nonlap = calculate_optiflex_parameters(optiflex_angle)
    else:
        wf_lap, wf_nonlap = 1.0, 1.0

    u_0 = u_avg * wf_nonlap
    u_1 = u_avg * wf_lap

    # --- 膜温度 ---
    T_surf_K_0 = T_0_surface + 273.15
    T_surf_K_1 = T_1_surface + 273.15
    T_air_K = T_air + 273.15

    T_film_0 = (T_surf_K_0 + T_air_K) / 2.0
    T_film_1 = (T_surf_K_1 + T_air_K) / 2.0
    T_film_avg = (T_film_0 + T_film_1) / 2.0

    ka_film = _k_air(T_film_avg)
    vair_film = _vair(T_film_avg)
    Prf = 0.7
    Prs = 0.7

    # --- Žukauskas 强制对流 ---
    def _nu_forced(Re):
        if Re < 40:
            C, m = 0.75, 0.4
        elif Re < 1000:
            C, m = 0.51, 0.5
        elif Re < 200000:
            C, m = 0.26, 0.6
        else:
            C, m = 0.076, 0.7
        return C * (Re**m) * (Prf**0.37) * ((Prf / Prs) ** 0.25)

    Re_0 = (u_0 * phi) / vair_film
    Re_1 = (u_1 * phi) / vair_film
    Nu_forced_0 = _nu_forced(Re_0)
    Nu_forced_1 = _nu_forced(Re_1)

    # --- Churchill & Chu 自然对流 ---
    g = 9.81
    beta = 1.0 / T_film_avg
    delta_T_avg = max(abs((T_0_surface - T_air + T_1_surface - T_air) / 2.0), epsilon)
    Gr = g * beta * delta_T_avg * (phi**3) / (vair_film**2)
    Ra = Gr * Prf

    if Ra > 0:
        Nu_nat = (0.60 + 0.387 * (Ra ** (1.0 / 6.0))
                  / ((1.0 + (0.559 / Prf) ** (9.0 / 16.0)) ** (8.0 / 27.0))) ** 2
    else:
        Nu_nat = 0.0

    Nu_0 = max(Nu_forced_0, Nu_nat)
    Nu_1 = max(Nu_forced_1, Nu_nat)

    # --- 对流换热系数 ---
    hc0 = (ka_film / phi) * Nu_0
    hc1 = (ka_film / phi) * Nu_1 * blocking_factor

    # 手动修正系数
    xs_hc0 = _get_params_attr(current_params, "xs_hc0", 1.0)
    xs_hc1 = _get_params_attr(current_params, "xs_hc1", 1.0)
    view_factor = _get_params_attr(current_params, "view_factor", 1.0)

    h_conv_0 = 1.5 * xs_hc0 * hc0
    h_conv_1 = 1.5 * xs_hc1 * hc1

    # --- 辐射换热系数 ---
    rad_coeff = 4.536

    d_T0 = T_0_surface - T_air
    h_rad_0 = 0 if abs(d_T0) < epsilon else rad_coeff * (pow(T_0_K_sc, 4) - pow(T_air_K_sc, 4)) / d_T0

    d_T1 = T_1_surface - T_air
    h_rad_1 = 0 if abs(d_T1) < epsilon else (
        view_factor * rad_coeff * (pow(T_1_K_sc, 4) - pow(T_air_K_sc, 4)) / d_T1
    )

    return h_conv_0, h_conv_1, h_rad_0, h_rad_1


# ═══════════════════════════════════════════════════════════════
# 相变孕育期与动力学
# ═══════════════════════════════════════════════════════════════


def get_incubation_time_pearlite(T_current, current_params=None):
    """计算珠光体等温孕育期 tau [s]。

    被调用: Q_calculation (内部)
    """
    if not (basic_info.Bs < T_current <= basic_info.A1):
        return 1e9

    kp = np.exp(10.164 - 16.002 * basic_info.ELM_C - 0.9797 * basic_info.ELM_MN
                + 0.00791 * T_current - 3.5067e-5 * T_current**2)
    tt = -0.91732 * np.log(kp) + 20 * np.log(T_current) + 1.9559 * 10000 / T_current - 157.45

    xs_taup = _get_params_attr(current_params, "xs_taup", 1.0)
    return np.exp(tt) * xs_taup * 0.5


def get_incubation_time_ferrite(T_current, current_params=None):
    """计算铁素体等温孕育期 tau [s]。

    被调用: Q_calculation (内部)
    """
    if not (basic_info.A1 < T_current <= basic_info.Acm):
        return 1e9

    Kf = 14.2 * math.exp(-(T_current - 620) / 25.1)
    tt = -1.6454 * np.log(Kf) + 20 * np.log(T_current) + 3.265 * 10000 / T_current - 173.89

    xs_tauf = _get_params_attr(current_params, "xs_tauf", 1.0)
    return np.exp(tt) * xs_tauf


# ═══════════════════════════════════════════════════════════════
# 相变热焓计算（JMAK 动力学）
# ═══════════════════════════════════════════════════════════════


def calculate_phase_change_heat(state, T_vec, time, position):
    """计算单侧（搭接/非搭接点）的相变热焓。

    参数:
        state: SimulationState 实例
        T_vec: 开尔文温度数组 [N]
        time: 当前仿真时间
        position: 0=非搭接点, 1=搭接点

    返回:
        Q_source [N], f_phase [N]（转变量）
    """
    N = state.N
    dt = state.dt
    params = state.current_params

    # 按位置选择状态数组
    if position == 0:
        suma_p, f_p, mark_sp, mark_ep = state.suma_p0, state.f_total_0, state.mark_sp_0, state.mark_ep_0
        suma_f, f_f, mark_sf, mark_ef = state.suma_f0, state.f_total_0, state.mark_sf_0, state.mark_ef_0
    else:
        suma_p, f_p, mark_sp, mark_ep = state.suma_p1, state.f_total_1, state.mark_sp_1, state.mark_ep_1
        suma_f, f_f, mark_sf, mark_ef = state.suma_f1, state.f_total_1, state.mark_sf_1, state.mark_ef_1

    Q_source = np.zeros(N)

    for i in range(N):
        T_k = T_vec[i]
        T_c = T_k - 273.15  # JMAK 和潜热必须使用摄氏度
        dq = 0

        # --- 记录铁素体最终转变量（穿过 A1 进入珠光体区时捕获）---
        if T_c <= basic_info.A1:
            if position == 0 and not state.ferrite_recorded_0[i]:
                state.ferrite_final_0[i] = f_p[i]
                state.ferrite_recorded_0[i] = True
            elif position == 1 and not state.ferrite_recorded_1[i]:
                state.ferrite_final_1[i] = f_p[i]
                state.ferrite_recorded_1[i] = True

        # --- 超过 Acm：无相变 ---
        if T_c > basic_info.Acm:
            pass

        # --- 先共析渗碳体/铁素体 (A1 < T <= Acm) ---
        elif T_c > basic_info.A1:
            if mark_ef[i] == 1:
                dq = 0
            elif mark_ef[i] == 0:
                if mark_sf[i] == 0:
                    tau = get_incubation_time_ferrite(T_c, params)
                    suma_f[i] += dt / tau
                    if suma_f[i] >= 0.99:
                        mark_sf[i] = 1
                elif mark_sf[i] == 1:
                    f_old = f_p[i]

                    n_jmak = 0.701
                    k_T = 14.2 * math.exp(-(T_c - 620) / 25.1)

                    if f_old > 0:
                        val = max(-math.log(1 - f_old) / k_T, 0.0)
                        tn = math.pow(val, 1.0 / n_jmak)
                    else:
                        tn = 0.0

                    tA = tn + dt
                    f_new = 1.0 - math.exp(-k_T * math.pow(tA, n_jmak))

                    Haf_current = 20789 - 15.62 * T_c - 0.24 * (T_c**2)

                    f_f[i] = f_new
                    real_df = f_new - f_old
                    dq = Haf_current * (real_df / dt) * basic_info.rho

                    xs_dqf = _get_params_attr(params, "xs_dqf", 1.0)
                    dq *= xs_dqf
                    dq = max(0, dq)

                    if f_new >= 0.99:
                        mark_ef[i] = 1

        # --- 珠光体 (Bs < T <= A1) ---
        elif T_c > basic_info.Bs:
            if mark_ep[i] == 1:
                dq = 0
            elif mark_ep[i] == 0:
                if mark_sp[i] == 0:
                    tau = get_incubation_time_pearlite(T_c, params)
                    suma_p[i] += dt / tau
                    if suma_p[i] >= 0.99:
                        mark_sp[i] = 1
                elif mark_sp[i] == 1:
                    f_old = f_p[i]

                    n_jmak = 2
                    k_T = np.exp(10.164 - 16.002 * basic_info.ELM_C - 0.9797 * basic_info.ELM_MN
                                 + 0.00791 * T_c - 3.5067e-5 * T_c**2)

                    if f_old > 0:
                        val = max(-math.log(1 - f_old) / k_T, 0.0)
                        tn = math.pow(val, 1.0 / n_jmak)
                    else:
                        tn = 0.0

                    tA = tn + dt
                    f_new = 1.0 - math.exp(-k_T * math.pow(tA, n_jmak))

                    Hap_current = 120848 - 52.42 * T_c - 0.158 * (T_c**2)

                    f_p[i] = f_new
                    real_df = f_new - f_old
                    dq = Hap_current * (real_df / dt) * basic_info.rho

                    xs_dqp = _get_params_attr(params, "xs_dqp", 1.0)
                    dq *= xs_dqp * 1.5
                    dq = max(0, dq)

                    if f_new >= 0.99:
                        mark_ep[i] = 1

        Q_source[i] = dq

    # 回写状态
    if position == 0:
        state.suma_p0, state.mark_sp_0, state.mark_ep_0 = suma_p, mark_sp, mark_ep
        state.suma_f0, state.mark_sf_0, state.mark_ef_0 = suma_f, mark_sf, mark_ef
        if T_c <= basic_info.A1 and T_c > basic_info.Bs:
            state.f_total_0 = f_p
        elif T_c > basic_info.A1 and T_c <= basic_info.Acm:
            state.f_total_0 = f_f
    else:
        state.suma_p1, state.mark_sp_1, state.mark_ep_1 = suma_p, mark_sp, mark_ep
        state.suma_f1, state.mark_sf_1, state.mark_ef_1 = suma_f, mark_sf, mark_ef
        if T_c <= basic_info.A1 and T_c > basic_info.Bs:
            state.f_total_1 = f_p
        elif T_c > basic_info.A1 and T_c <= basic_info.Acm:
            state.f_total_1 = f_f

    return Q_source, f_p


def calculate_total_phase_heat(state, T_vec_0, T_vec_1, Hap_0, Hap_1, time):
    """计算双侧相变热焓（非搭接点 + 搭接点）。

    被调用: cooling_calculation
    返回: Q_0, Q_1, f_p0, f_p1
    """
    Q_0, f_p0 = calculate_phase_change_heat(state, T_vec_0, time, position=0)
    Q_1, f_p1 = calculate_phase_change_heat(state, T_vec_1, time, position=1)
    return Q_0, Q_1, f_p0, f_p1


# ═══════════════════════════════════════════════════════════════
# 有限差分温度求解
# ═══════════════════════════════════════════════════════════════


def _solve_heat_conduction_step(T_current, h_val, Q_src, k_val, c, state):
    """单步隐式有限差分求解 1D 径向热传导。

    返回: 更新后的温度场 [N]
    """
    N = state.N
    dt = state.dt
    dr = state.dr
    r_edge = basic_info.r

    area_out = 2 * np.pi * r_edge
    area_in = 2 * np.pi * (r_edge - dr)
    vol_edge = np.pi * (r_edge**2 - (r_edge - dr)**2)

    alpha = k_val / (basic_info.rho * c)
    Fo = alpha * dt / (dr**2)

    A = np.zeros((N, N))
    B = np.zeros(N)
    source_term = Q_src * dt / (basic_info.rho * c)

    # 内部节点
    indices = np.arange(1, N - 1)
    A[indices, indices - 1] = -Fo * (1 - 1 / (2 * indices))
    A[indices, indices] = 1 + 2 * Fo
    A[indices, indices + 1] = -Fo * (1 + 1 / (2 * indices))
    B[indices] = T_current[indices] + source_term[indices]

    # 中心节点 (i=0)
    A[0, 0] = 1 + 4 * Fo
    A[0, 1] = -4 * Fo
    B[0] = T_current[0] + source_term[0]

    # 表面节点
    term_cond = k_val * area_in / dr
    term_conv = h_val * area_out
    term_cap = basic_info.rho * c * vol_edge / dt

    A[-1, -2] = -term_cond
    A[-1, -1] = term_cap + term_cond + term_conv
    B[-1] = term_cap * T_current[-1] + term_conv * (basic_info.Ta + 273.15) + Q_src[-1] * vol_edge

    return np.linalg.solve(A, B)


def _record_history(state, current_time, T_field_0, T_field_1, Q_0, Q_1, f_p0, f_p1, h_r0, h_c0, h_0, h_r1, h_c1, h_1):
    """记录当前时间步的仿真历史数据。"""
    state.history_time.append(current_time)
    for row, val in zip(state.history_T_0, T_field_0 - 273.15):
        row.append(val)
    for row, val in zip(state.history_T_1, T_field_1 - 273.15):
        row.append(val)
    for row, val in zip(state.history_Q_0, Q_0):
        row.append(val)
    for row, val in zip(state.history_Q_1, Q_1):
        row.append(val)
    # 铁素体和珠光体分别记录
    f_ferrite_0 = np.where(state.ferrite_recorded_0, state.ferrite_final_0, f_p0)
    f_pearlite_0 = np.maximum(0, f_p0 - f_ferrite_0)
    f_ferrite_1 = np.where(state.ferrite_recorded_1, state.ferrite_final_1, f_p1)
    f_pearlite_1 = np.maximum(0, f_p1 - f_ferrite_1)
    for row, val in zip(state.ferrite_0, f_ferrite_0):
        row.append(val)
    for row, val in zip(state.pearlite_0, f_pearlite_0):
        row.append(val)
    for row, val in zip(state.ferrite_1, f_ferrite_1):
        row.append(val)
    for row, val in zip(state.pearlite_1, f_pearlite_1):
        row.append(val)
    state.history_h_0[0].append(h_r0)
    state.history_h_0[1].append(h_c0)
    state.history_h_0[2].append(h_0)
    state.history_h_1[0].append(h_r1)
    state.history_h_1[1].append(h_c1)
    state.history_h_1[2].append(h_1)


def cooling_calculation(state, roll_obj, time_offset=0.0):
    """对单段辊道执行冷却计算。

    被调用: run_full_simulation, calculate_all_sim_T.py

    参数:
        state: SimulationState 实例
        roll_obj: roll 实例，其 pre_temp_0/1 为入口温度
        time_offset: 该段开始时的全局累计时间，s（保证跨段历史时间戳连续）

    更新:
        state 内部的历史记录数组
        roll_obj.post_temp_0/1 为出口温度
    """
    # 拷贝入口温度（避免修改传入数组）
    T_field_0 = roll_obj.pre_temp_0.copy() + 273.15
    T_field_1 = roll_obj.pre_temp_1.copy() + 273.15

    dt = state.dt

    for j in range(roll_obj.step):
        local_time = (j + 1) * dt
        global_time = time_offset + local_time  # 跨段连续的全局时间戳

        # --- 步骤 A: 基于当前温度更新物性 ---
        T_surf_0 = T_field_0[-1] - 273.15
        T_surf_1 = T_field_1[-1] - 273.15
        T_avg_0 = np.mean(T_field_0) - 273.15
        T_avg_1 = np.mean(T_field_1) - 273.15

        k_val_0, c_0, Hap_0 = calculate_physical_parameters(T_avg_0)
        k_val_1, c_1, Hap_1 = calculate_physical_parameters(T_avg_1)

        h_c0, h_c1, h_r0, h_r1 = calculate_heat_transfer(
            T_surf_0, T_surf_1, basic_info.T_air,
            basic_info.phi * 1e-3,
            roll_obj.fan_air_volume, roll_obj.fan_status, roll_obj.fan_area,
            roll_obj.optiflex_angle, state.current_params,
        )
        h_0 = h_c0 + h_r0
        h_1 = h_c1 + h_r1

        # --- 步骤 B: 相变热源 ---
        Q_0, Q_1, f_p0, f_p1 = calculate_total_phase_heat(
            state, T_field_0, T_field_1, Hap_0, Hap_1, global_time,
        )

        # --- 步骤 C: 差分求解 ---
        T_field_0 = _solve_heat_conduction_step(T_field_0, h_0, Q_0, k_val_0, c_0, state)
        T_field_1 = _solve_heat_conduction_step(T_field_1, h_1, Q_1, k_val_1, c_1, state)

        # --- 步骤 D: 记录历史 ---
        _record_history(state, global_time, T_field_0, T_field_1, Q_0, Q_1, f_p0, f_p1,
                        h_r0, h_c0, h_0, h_r1, h_c1, h_1)

    # 保存出口温度
    roll_obj.post_temp_0 = T_field_0 - 273.15
    roll_obj.post_temp_1 = T_field_1 - 273.15


# ═══════════════════════════════════════════════════════════════
# 高层仿真接口
# ═══════════════════════════════════════════════════════════════


def run_full_simulation(rolls, tem1=850, tem0=830, params=None, dt=0.01):
    """运行一次完整的斯太尔摩冷却线仿真。

    被调用: sim_T.__main__, calculate_all_sim_T.py, predict.py

    参数:
        rolls: roll 对象列表
        tem1: 搭接点入口温度，℃
        tem0: 非搭接点入口温度，℃
        params: parameter_change 对象或 None
        dt: 仿真时间步长，s

    返回:
        (state, roll_start_time) — SimulationState 实例和辊道起始时间列表
    """
    set_default_dt(dt)
    state = SimulationState(dt=dt)
    state.current_params = params

    num_rolls = len(rolls)

    # 重新计算每段停留时间和步数（roll_v 可能已被外部修改）
    for r in rolls:
        r.t = r.roll_length / r.roll_v
        r.step = int(r.t / dt)

    temp_trans_1 = np.full(state.N, tem1)
    temp_trans_0 = np.full(state.N, tem0)

    roll_start_time = [0.0]
    cumulative_time = 0.0
    for i in range(num_rolls):
        current_roll = rolls[i]
        current_roll.pre_temp_0 = temp_trans_0
        current_roll.pre_temp_1 = temp_trans_1
        cooling_calculation(state, current_roll, time_offset=cumulative_time)
        temp_trans_0 = current_roll.post_temp_0
        temp_trans_1 = current_roll.post_temp_1
        cumulative_time += float(current_roll.t)
        roll_start_time.append(cumulative_time)

    return state, roll_start_time


def run_simulation_with_params(params, dt=0.1):
    """以指定修正参数运行一次仿真，返回测温点温度。

    供 change_parameter_82A.py, calculate_the_effect.py 调用的便捷接口。

    返回: (搭接点测温温度数组, 非搭接点测温温度数组)
    """
    # 被调用: change_parameter_82A.py, calculate_the_effect.py
    set_default_dt(dt)
    rolls, num_rolls = data_loader.load_roll_data()
    state, roll_start_time = run_full_simulation(rolls, tem1=850, tem0=830, params=params, dt=dt)
    return get_measure_point_T_results(state, roll_start_time)


# ═══════════════════════════════════════════════════════════════
# 测温点数据与时间
# ═══════════════════════════════════════════════════════════════

# 现场实测测温点温度（搭接点 8 点、非搭接点 8 点）
MEASURE_POINT_T_REAL_1 = [845, 822, 718, 656, 650, 595, 575, 558]
MEASURE_POINT_T_REAL_0 = [830, 815, 705, 645, 630, 595, 568, 558]


def get_measure_point_times(roll_start_time):
    """从辊道起始时间计算测温点时间序列（8 个点）。

    被调用: change_parameter_82A.py, calculate_the_effect.py, get_measure_point_T_results
    """
    big_idx = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22]
    small_idx = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    big_times = [roll_start_time[i] for i in big_idx if i < len(roll_start_time)]
    small_times = [roll_start_time[i] for i in small_idx if i < len(roll_start_time)]

    measure_point_time = [
        big_times[0], big_times[1], small_times[0],
        big_times[2], small_times[1], big_times[3],
        small_times[2], big_times[4], small_times[3],
        big_times[5], small_times[4], big_times[6],
        small_times[5], big_times[7], small_times[6],
    ]
    return measure_point_time[:8]


# ═══════════════════════════════════════════════════════════════
# 结果提取与采样
# ═══════════════════════════════════════════════════════════════


def get_measure_point_T_results(state, roll_start_time):
    """从仿真历史中提取测温点温度。

    被调用: change_parameter_82A.py, calculate_the_effect.py, predict.py

    返回: (搭接点测温温度, 非搭接点测温温度) — 各 8 个点
    """
    measure_point_time = get_measure_point_times(roll_start_time)

    if len(state.history_T_1[-1]) == 0:
        raise ValueError("history_T_1 为空，尚未完成仿真，无法提取测温点温度。")

    measure_point_T_sim1 = np.zeros(len(measure_point_time))
    measure_point_T_sim0 = np.zeros(len(measure_point_time))
    max_idx = len(state.history_T_1[-1]) - 1

    for i, t in enumerate(measure_point_time):
        idx = int(t / state.dt)
        idx = max(0, min(idx, max_idx))
        measure_point_T_sim1[i] = state.history_T_1[-1][idx]
        measure_point_T_sim0[i] = state.history_T_0[-1][idx]

    return measure_point_T_sim1, measure_point_T_sim0


def sample_with_step(history_time_raw, history_t0_raw, history_t1_raw, roll_start_time=None, n=1):
    """按辊道起始时间及相邻区间均匀采样。

    被调用: calculate_all_sim_T.py, predict.py

    n=1 时等价于取中点。
    """
    if n < 1:
        raise ValueError("n 必须大于等于 1。")

    min_len = min(len(history_time_raw), len(history_t0_raw), len(history_t1_raw))
    if min_len == 0:
        return [], [], [], []

    time_raw = np.asarray(history_time_raw[:min_len], dtype=float)
    t0_raw = np.asarray(history_t0_raw[:min_len], dtype=float)
    t1_raw = np.asarray(history_t1_raw[:min_len], dtype=float)

    if not roll_start_time or len(roll_start_time) < 2:
        return time_raw.tolist(), t0_raw.tolist(), t1_raw.tolist(), list(range(len(time_raw)))

    sample_time_candidates = []
    for i in range(len(roll_start_time) - 1):
        t_start = float(roll_start_time[i])
        t_end = float(roll_start_time[i + 1])
        sample_time_candidates.append(t_start)
        if n == 1:
            sample_time_candidates.append((t_start + t_end) / 2.0)
        else:
            sample_time_candidates.extend(np.linspace(t_start, t_end, n + 2)[1:-1].tolist())
    sample_time_candidates.append(float(roll_start_time[-1]))

    # 去重排序
    sample_times = []
    for t in sorted(sample_time_candidates):
        if not sample_times or abs(t - sample_times[-1]) > 1e-9:
            sample_times.append(t)

    sampled_time, sampled_t0, sampled_t1 = [], [], []
    max_idx = len(time_raw) - 1

    for target_time in sample_times:
        insert_pos = int(np.searchsorted(time_raw, target_time, side="left"))
        if insert_pos <= 0:
            idx = 0
        elif insert_pos > max_idx:
            idx = max_idx
        else:
            left_idx = insert_pos - 1
            right_idx = insert_pos
            idx = left_idx if abs(time_raw[left_idx] - target_time) < abs(time_raw[right_idx] - target_time) else right_idx

        sampled_time.append(float(target_time))
        sampled_t0.append(float(t0_raw[idx]))
        sampled_t1.append(float(t1_raw[idx]))

    return sampled_time, sampled_t0, sampled_t1, list(range(len(sampled_time)))


def calculate_importance_for_real_point(X, Y, peak_weight_ratio=0.5, lambda_smooth=0.9):
    """高度可调的权重计算算法。

    被调用: change_parameter_82A.py, sim_T.py (绘图)

    参数:
        X, Y: 原始数据的横纵坐标向量
        peak_weight_ratio: 0~1，越大越侧重捕捉梯度峰值（回温）
        lambda_smooth: 0~1，越大权重分配越均匀
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    n = len(X)

    if n < 3:
        return np.ones(n) / n

    # 前向和后向差分
    m_b = np.zeros(n)
    m_f = np.zeros(n)
    m_b[1:] = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])
    m_b[0] = m_b[1]
    m_f[:-1] = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])
    m_f[-1] = m_f[-2]

    # 特征 A: 差分不一致性（转折点）
    s_discord = np.abs(m_f - m_b)

    # 特征 B: 梯度峰值（回温/潜热）
    s_peak = np.zeros(n)
    diff_left = m_b[1:-1] - m_b[:-2]
    diff_right = m_b[1:-1] - m_b[2:]
    s_peak[1:-1] = (np.maximum(0, diff_left) * np.maximum(0, diff_right)) ** 2

    def _safe_norm(vec):
        v_min, v_max = np.min(vec), np.max(vec)
        return (vec - v_min) / (v_max - v_min + 1e-9)

    norm_discord = _safe_norm(s_discord)
    norm_peak = _safe_norm(s_peak)

    combined_score = (1 - peak_weight_ratio) * norm_discord + peak_weight_ratio * norm_peak
    feature_impact = np.sqrt(combined_score + 1e-9)
    feature_weight = feature_impact / np.sum(feature_impact)

    uniform_weight = np.ones(n) / n
    return lambda_smooth * uniform_weight + (1 - lambda_smooth) * feature_weight


# ═══════════════════════════════════════════════════════════════
# 可视化函数
# ═══════════════════════════════════════════════════════════════


def _get_segment_times(roll_start_time):
    """从辊道起始时间提取大段/小段分割时刻。"""
    big_idx = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26]
    small_idx = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]

    big_times = [roll_start_time[i] for i in big_idx if i < len(roll_start_time)]
    small_times = [roll_start_time[i] for i in small_idx if i < len(roll_start_time)]
    return big_times, small_times


def _add_segment_background(ax, big_times):
    """为大段区域添加交替背景色和分割线。"""
    bg_colors = ['#f6f8fb', '#fbf8f3']
    for i in range(len(big_times) - 1):
        ax.axvspan(big_times[i], big_times[i + 1], facecolor=bg_colors[i % 2], alpha=1, zorder=0)


def _add_segment_lines(ax, big_times, small_times):
    """添加大小段分割虚线。"""
    ymin, ymax = ax.get_ylim()
    if small_times:
        ax.vlines(small_times, ymin=ymin, ymax=ymax, linestyles='--', colors='gray', alpha=0.7)
    if big_times:
        ax.vlines(big_times, ymin=ymin, ymax=ymax, linestyles='--', colors='black', alpha=0.7)


def plot_T_results(state, roll_start_time):
    """绘制温度仿真结果曲线。"""
    big_times, small_times = _get_segment_times(roll_start_time)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(state.history_time, state.history_T_0[-1], label='Surface Temp Non-Overlap (°C)')
    plt.plot(state.history_time, state.history_T_0[0], label='Center Temp Non-Overlap (°C)')
    plt.plot(state.history_time, state.history_T_1[-1], label='Surface Temp Overlap (°C)')
    plt.plot(state.history_time, state.history_T_1[0], label='Center Temp Overlap (°C)')

    _add_segment_background(ax, big_times)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Profiles During Cooling')
    plt.legend()
    plt.grid()
    _add_segment_lines(ax, big_times, small_times)


def plot_H_results(state, roll_start_time):
    """绘制换热系数变化曲线。"""
    big_times, small_times = _get_segment_times(roll_start_time)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(state.history_time, state.history_h_0[0], label='h_r Non-Overlap')
    plt.plot(state.history_time, state.history_h_0[1], label='h_c Non-Overlap')
    plt.plot(state.history_time, state.history_h_1[0], label='h_r Overlap')
    plt.plot(state.history_time, state.history_h_1[1], label='h_c Overlap')

    _add_segment_background(ax, big_times)
    plt.xlabel('Time (s)')
    plt.ylabel('Heat Transfer Coefficient (W/m²·K)')
    plt.title('Heat Transfer Coefficient During Cooling')
    plt.legend()
    plt.grid()
    _add_segment_lines(ax, big_times, small_times)


def plot_Q_results(state, roll_start_time):
    """绘制相变热焓变化曲线。"""
    big_times, small_times = _get_segment_times(roll_start_time)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    plt.plot(state.history_time, state.history_Q_0[-1], label='surface_Q Non-Overlap')
    plt.plot(state.history_time, state.history_Q_1[-1], label='surface_Q Overlap')

    _add_segment_background(ax, big_times)
    plt.xlabel('Time (s)')
    plt.ylabel('Q (W/m³)')
    plt.title('Q During Cooling')
    plt.legend()
    plt.grid()
    _add_segment_lines(ax, big_times, small_times)


def plot_trans_stage(state, roll_start_time):
    """绘制相变转变量曲线 —— 铁素体与珠光体分别表示。"""
    big_times, small_times = _get_segment_times(roll_start_time)

    fig, (ax_f, ax_p) = plt.subplots(1, 2, figsize=(16, 6))

    # 铁素体子图
    ax_f.plot(state.history_time, state.ferrite_0[-1], label='Surface Non-Overlap')
    ax_f.plot(state.history_time, state.ferrite_1[-1], label='Surface Overlap')
    ax_f.plot(state.history_time, state.ferrite_0[0], label='Center Non-Overlap')
    ax_f.plot(state.history_time, state.ferrite_1[0], label='Center Overlap')
    _add_segment_background(ax_f, big_times)
    ax_f.set_xlabel('Time (s)')
    ax_f.set_ylabel('Ferrite Fraction')
    ax_f.set_title('Ferrite Transformation')
    ax_f.legend(fontsize=8)
    ax_f.grid()
    _add_segment_lines(ax_f, big_times, small_times)

    # 珠光体子图
    ax_p.plot(state.history_time, state.pearlite_0[-1], label='Surface Non-Overlap')
    ax_p.plot(state.history_time, state.pearlite_1[-1], label='Surface Overlap')
    ax_p.plot(state.history_time, state.pearlite_0[0], label='Center Non-Overlap')
    ax_p.plot(state.history_time, state.pearlite_1[0], label='Center Overlap')
    _add_segment_background(ax_p, big_times)
    ax_p.set_xlabel('Time (s)')
    ax_p.set_ylabel('Pearlite Fraction')
    ax_p.set_title('Pearlite Transformation')
    ax_p.legend(fontsize=8)
    ax_p.grid()
    _add_segment_lines(ax_p, big_times, small_times)

    fig.suptitle('Phase Transformation During Cooling', fontsize=13)
    fig.tight_layout()


def plot_measure_point_T_results(state, roll_start_time, metric_mode="MAE"):
    """绘制测温点仿真值与实测值对比。

    被调用: sim_T.__main__
    """
    measure_point_time = get_measure_point_times(roll_start_time)

    measure_point_time_sim = state.history_time[:int(30 / state.dt)]
    measure_point_T_sim1 = state.history_T_1[-1][:int(30 / state.dt)]
    measure_point_T_sim0 = state.history_T_0[-1][:int(30 / state.dt)]

    measure_point_T_real_1 = MEASURE_POINT_T_REAL_1
    measure_point_T_real_0 = MEASURE_POINT_T_REAL_0

    mode = str(metric_mode).upper()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(measure_point_time_sim, measure_point_T_sim1, label='Simulated Temp (°C)', marker='.', color='gray')

    if mode == "WMAE":
        x1 = np.array(measure_point_time[:len(measure_point_T_real_1)], dtype=np.float64)
        x0 = np.array(measure_point_time[:len(measure_point_T_real_0)], dtype=np.float64)
        weights1 = np.asarray(calculate_importance_for_real_point(x1, measure_point_T_real_1), dtype=np.float64)
        weights0 = np.asarray(calculate_importance_for_real_point(x0, measure_point_T_real_0), dtype=np.float64)
        vmin = min(np.min(weights1), np.min(weights0))
        vmax = max(np.max(weights1), np.max(weights0))
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("RdYlGn_r")

        sc1 = axs[0].scatter(x1, measure_point_T_real_1, label='Actual Temp (°C)',
                             marker='o', c=weights1, cmap=cmap, norm=norm,
                             edgecolors='black', linewidths=0.6, s=80, zorder=3)
        for x, y, w in zip(x1, measure_point_T_real_1, weights1):
            axs[0].text(x + 0.05, y + 2.0, f"{w:.3f}", fontsize=8, color='black')
    else:
        axs[0].scatter(measure_point_time[:len(measure_point_T_real_1)], measure_point_T_real_1,
                       label='Actual Temp (°C)', marker='*', color='red', zorder=3)

    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Temperature (°C)')
    axs[0].set_title('Temperature at Measurement Points 1')
    axs[0].legend(loc='upper right')

    axs[1].plot(measure_point_time_sim, measure_point_T_sim0, label='Simulated Temp (°C)', marker='.', color='gray')

    if mode == "WMAE":
        sc0 = axs[1].scatter(x0, measure_point_T_real_0, label='Actual Temp (°C)',
                             marker='o', c=weights0, cmap=cmap, norm=norm,
                             edgecolors='black', linewidths=0.6, s=80, zorder=3)
        for x, y, w in zip(x0, measure_point_T_real_0, weights0):
            axs[1].text(x + 0.05, y + 2.0, f"{w:.3f}", fontsize=8, color='black')
        cbar = fig.colorbar(sc0, ax=axs.ravel().tolist(), shrink=0.92, pad=0.02)
        cbar.set_label('Real-point Weight (low=green, high=red)')
    else:
        axs[1].scatter(measure_point_time[:len(measure_point_T_real_0)], measure_point_T_real_0,
                       label='Actual Temp (°C)', marker='*', color='red', zorder=3)

    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Temperature (°C)')
    axs[1].set_title('Temperature at Measurement Points 0')
    axs[1].legend(loc='upper right')


def save_sim_data_as_csv(state):
    """将仿真结果保存为 CSV 文件。

    输出列: 时间, 非搭接点表面温度, 搭接点表面温度
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "sim_data.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(state.history_time)
        writer.writerow(state.history_T_0[-1])
        writer.writerow(state.history_T_1[-1])

    print(f"文件保存成功！路径：{csv_path}")


# ═══════════════════════════════════════════════════════════════
# 向后兼容别名
# ═══════════════════════════════════════════════════════════════

# simulation_model 保留为命名空间，提供向后兼容的类方法访问
class simulation_model:
    """向后兼容层，代理到新的模块级函数。

    新代码应直接使用模块级函数和 SimulationState。
    """
    N = 50
    dt = 0.01
    dr = basic_info.r / 50

    # 类级别委托属性
    _state = None

    @staticmethod
    def Calculate_physical_parameters(T_current):
        return calculate_physical_parameters(T_current)

    @staticmethod
    def Calculate_optitflex_parameters(optiflex_angle):
        return calculate_optiflex_parameters(optiflex_angle)

    @staticmethod
    def H_calculation(*args, **kwargs):
        return calculate_heat_transfer(*args, **kwargs)

    @staticmethod
    def get_incubation_time_pearlite(T_current):
        return get_incubation_time_pearlite(T_current)

    @staticmethod
    def get_incubation_time_ferrite(T_current):
        return get_incubation_time_ferrite(T_current)

    @staticmethod
    def Q_calculation(T_vec_0, T_vec_1, Hap_0, Hap_1, time):
        if simulation_model._state is None:
            simulation_model._state = SimulationState()
        return calculate_total_phase_heat(simulation_model._state, T_vec_0, T_vec_1, Hap_0, Hap_1, time)

    @staticmethod
    def Cooling_calculation(roll_obj):
        if simulation_model._state is None:
            simulation_model._state = SimulationState()
        return cooling_calculation(simulation_model._state, roll_obj)

    @staticmethod
    def get_measure_point_T_results():
        if simulation_model._state is None:
            simulation_model._state = SimulationState()
        # 需要 roll_start_time，从全局获取
        rt = globals().get("roll_start_time", [0.0])
        return get_measure_point_T_results(simulation_model._state, rt)

    @staticmethod
    def sample_with_step(*args, **kwargs):
        return sample_with_step(*args, **kwargs)

    @staticmethod
    def calculate_importance_for_real_point(*args, **kwargs):
        return calculate_importance_for_real_point(*args, **kwargs)

    @staticmethod
    def plot_T_results():
        if simulation_model._state is not None:
            rt = globals().get("roll_start_time", [0.0])
            plot_T_results(simulation_model._state, rt)

    @staticmethod
    def plot_H_results():
        if simulation_model._state is not None:
            rt = globals().get("roll_start_time", [0.0])
            plot_H_results(simulation_model._state, rt)

    @staticmethod
    def plot_Q_results():
        if simulation_model._state is not None:
            rt = globals().get("roll_start_time", [0.0])
            plot_Q_results(simulation_model._state, rt)

    @staticmethod
    def plot_measure_point_T_results(metric_mode="MAE"):
        if simulation_model._state is not None:
            rt = globals().get("roll_start_time", [0.0])
            plot_measure_point_T_results(simulation_model._state, rt, metric_mode)

    @staticmethod
    def plot_trans_stage():
        if simulation_model._state is not None:
            rt = globals().get("roll_start_time", [0.0])
            plot_trans_stage(simulation_model._state, rt)

    @staticmethod
    def save_sim_data_as_csv():
        if simulation_model._state is not None:
            save_sim_data_as_csv(simulation_model._state)


# ═══════════════════════════════════════════════════════════════
# 主程序入口
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 加载修正参数
    params = build_parameter_from_file(0)

    # 加载辊道数据
    rolls, num_rolls = data_loader.load_roll_data()

    # 运行完整仿真
    state, roll_start_time = run_full_simulation(rolls, tem1=850, tem0=830, params=params, dt=0.01)

    # 供旧代码兼容访问
    simulation_model._state = state

    # 提取大段/小段分割时刻（供绘图使用）
    big_idx = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 26]
    small_idx = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    big_start_time = [roll_start_time[i] for i in big_idx if i < len(roll_start_time)]
    small_start_time = [roll_start_time[i] for i in small_idx if i < len(roll_start_time)]

    # 绘图
    plot_T_results(state, roll_start_time)
    plot_H_results(state, roll_start_time)
    plot_Q_results(state, roll_start_time)
    plot_measure_point_T_results(state, roll_start_time, metric_mode="WMAE")
    plot_trans_stage(state, roll_start_time)
    plt.show()
