'''
1.未解决-dq计算值偏小，是否考虑实际体积大小?采用cpp的相变计算方法
2.已解决-风速计算时的风量的单位未统一. 
3.未解决-换热系数的计算方法是cpp的,搭接点和非搭接点的区分方式:考虑搭接点的风速只有非搭接点的0.2。
4.未解决-佳灵装置的仿真计算
5.未解决-保温罩的仿真计算
6.未解决-未考虑渗碳体、贝氏体的相变计算
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import csv
import os
from pathlib import Path

#基础工艺类
class basic_info:
    #基础参数，不随仿真段变化
    T_air = Ta = 25  #环境温度，单位℃
    a = 29.228

    steel_grade = '82A'
          
    rho = 7823       # 密度, kg/m^3
    phi = 5.5  #线材直径，单位mm
    r = phi * 1e-3 / 2  #线材半径，单位m
    ELM_C = 0.82 / 100 #碳含量，单位%   
    ELM_SI = 0.21 / 100 #硅含量，单位%
    ELM_MN = 0.52 / 100 #锰含量，单位%
    ELM_NI = 0.009858 / 100
    ELM_CR = 0.011815 / 100
    # YIELD = 270  #下屈服强度，单位MPa
    # UTS = 410    #抗拉强度，单位MPa
    # BREAK_EL = 36  #断后伸长率，单位%
    # EXT = 51  #断面收缩

    v_wire = 100    # 吐丝速度，猜的
    D_ring = 1.05    # 线环直径，A线1.05m，B线1.075m
    Acm = 727 + 314.2 * (ELM_C * 100 - 0.77)  # 过共析钢上临界点，替代原A3=820
    A1 = 727 - 10.7*(ELM_MN*100) - 16.9*(ELM_NI*100) + 16*(ELM_CR*100) + 29.1*(ELM_SI*100)
    Bs = 830 - 270*(ELM_C*100) - 90*(ELM_MN*100) - 37*(ELM_NI*100) - 70*(ELM_CR*100)

class parameter_change:
    #修正仿真过程的参数
    def __init__(self, num):
        self.num = num
        self.xs_hc0 = 1.0    #非搭接点对流换热修正系数
        self.xs_hc1 = 1.0    #搭接点对流换热修正系数
        self.view_factor = 1.0   #搭接点辐射遮挡修正系数
        self.xs_tauf = 1.0   #铁素体孕育期修正系数
        self.xs_taup = 1.0   #珠光体孕育期修正系数
        self.xs_dqp = 1.0    #珠光体相变热焓修正系数
        self.xs_dqf = 1.0    #铁素体相变热焓修正系数
        self.result = 0      #结果记录变量，0-1之间，数值越大表示约接近真实值


PARAMETER_FILE = Path(__file__).with_name("parameter.txt")
PARAMETER_KEYS = [
    "xs_hc0",
    "xs_hc1",
    "view_factor",
    "xs_tauf",
    "xs_taup",
    "xs_dqp",
    "xs_dqf",
]


def _parse_parameter_lines(lines):
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
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"parameter.txt not found at {path}. Run change_parameter_82A.py to generate it."
        )
    content = path.read_text(encoding="utf-8").splitlines()
    return _parse_parameter_lines(content)


def build_parameter_from_file(num=0, file_path=PARAMETER_FILE):
    params_dict = read_parameter_file(file_path)
    param_obj = parameter_change(num)
    for name in PARAMETER_KEYS:
        if name in params_dict:
            setattr(param_obj, name, float(params_dict[name]))
    return param_obj


#仿真每段辊道相关   
class roll:
    def __init__(self, roll_num):
        self.roll_num = roll_num
        self.roll_name = 'name_undefined'  #辊道名称
        self.roll_length = 9.252 #辊道长度，单位m
        self.roll_v = 0.97 #辊道运行速度，单位m/s
        self.t = self.roll_length / (self.roll_v / 60)  # 辊道停留时间，单位s
        self.step = int(self.t / simulation_model.dt)  # 仿真步数
        self.fan_air_volume = 0 #风机风量，单位m3/s
        self.fan_status = 1 #风机开度，0-100之间
        self.fan_area = 1.5*9.252/2 #风机面积
        self.fan_speed = 0 #风机风速，单位m/s
        self.thermal_cove = 0 #保温罩状态，0=不使用，1=使用
        self.optiflex_angle = 0  # 佳灵装置开合角度 (deg)，0=关闭/无装置
        #该段前数据
        self.pre_temp_0 = np.zeros(simulation_model.N)   #开始前非搭接点温度单位℃
        self.pre_temp_1 = np.zeros(simulation_model.N)   #开始前搭接点温度单位℃
        #该段后数据
        self.post_temp_0 = np.zeros(simulation_model.N)   #结束后非搭接点温度单位℃
        self.post_temp_1 = np.zeros(simulation_model.N)   #结束后搭接点温度单位℃
        
#数据加载类
class data_loader:
    @staticmethod
    def load_roll_data():
        n = 26  #26段小段
        rolls = [roll(i) for i in range(n)]
        rolls[0].roll_name = 'IN'
        rolls[0].t = 3.0
        rolls[0].roll_length = 9.252/2
        rolls[0].roll_v = 0.85
        rolls[0].step = int(rolls[0].t / simulation_model.dt)
        rolls[0].fan_status = 0

        rolls[1].roll_name = '1-1'
        rolls[1].roll_length = 9.252/2
        rolls[1].roll_v = 0.97
        rolls[1].t = 8.411/2
        rolls[1].step = int(rolls[1].t / simulation_model.dt)
        rolls[1].fan_air_volume = 53.889
        rolls[1].fan_status = 0.99
        rolls[1].fan_speed = rolls[1].fan_air_volume * rolls[1].fan_status / rolls[1].fan_area
        rolls[1].optiflex_angle = 3.0  # 前4台风机佳灵装置开度 3deg (Deng et al., 2025)
        rolls[2].roll_name = '1-2'
        rolls[2].roll_length = 9.252/2
        rolls[2].roll_v = 0.97
        rolls[2].t = 8.411/2
        rolls[2].step = int(rolls[2].t / simulation_model.dt)
        rolls[2].fan_air_volume = 53.889 
        rolls[2].fan_status = 0.99
        rolls[2].fan_speed = rolls[2].fan_air_volume * rolls[2].fan_status / rolls[2].fan_area
        rolls[2].optiflex_angle = 3.0

        rolls[3].roll_name = '2-1'
        rolls[3].roll_length = 9.252/2
        rolls[3].roll_v = 1.05
        rolls[3].t = 7.716/2
        rolls[3].step = int(rolls[3].t / simulation_model.dt)
        rolls[3].fan_air_volume = 53.889
        rolls[3].fan_status = 0.95
        rolls[3].fan_speed = rolls[3].fan_air_volume * rolls[3].fan_status / rolls[3].fan_area
        rolls[3].optiflex_angle = 3.0
        rolls[4].roll_name = '2-2'
        rolls[4].roll_length = 9.252/2
        rolls[4].roll_v = 1.05
        rolls[4].t = 7.716/2
        rolls[4].step = int(rolls[4].t / simulation_model.dt)
        rolls[4].fan_air_volume = 53.889
        rolls[4].fan_status = 0.95
        rolls[4].fan_speed = rolls[4].fan_air_volume * rolls[4].fan_status / rolls[4].fan_area
        rolls[4].optiflex_angle = 3.0

        rolls[5].roll_name = '3-1'
        rolls[5].roll_length = 9.252/2
        rolls[5].roll_v = 1.15
        rolls[5].t = 7.211/2
        rolls[5].step = int(rolls[5].t / simulation_model.dt)
        rolls[5].fan_air_volume = 53.889
        rolls[5].fan_status = 0
        rolls[5].fan_speed = rolls[5].fan_air_volume * rolls[5].fan_status / rolls[5].fan_area
        rolls[5].optiflex_angle = 3.0
        rolls[6].roll_name = '3-2'
        rolls[6].roll_length = 9.252/2
        rolls[6].roll_v = 1.15
        rolls[6].t = 7.211/2
        rolls[6].step = int(rolls[6].t / simulation_model.dt)
        rolls[6].fan_air_volume = 53.889
        rolls[6].fan_status = 0
        rolls[6].fan_speed = rolls[6].fan_air_volume * rolls[6].fan_status / rolls[6].fan_area
        rolls[6].optiflex_angle = 3.0

        rolls[7].roll_name = '4-1'
        rolls[7].roll_length = 9.252/2
        rolls[7].roll_v = 1.20
        rolls[7].t = 6.869/2
        rolls[7].step = int(rolls[7].t / simulation_model.dt)
        rolls[7].fan_air_volume = 53.889
        rolls[7].fan_status = 0
        rolls[7].fan_speed = rolls[7].fan_air_volume * rolls[7].fan_status / rolls[7].fan_area
        rolls[7].optiflex_angle = 3.0
        rolls[8].roll_name = '4-2'
        rolls[8].roll_length = 9.252/2
        rolls[8].roll_v = 1.20
        rolls[8].t = 6.869/2
        rolls[8].step = int(rolls[8].t / simulation_model.dt)
        rolls[8].fan_air_volume = 53.889
        rolls[8].fan_status = 0
        rolls[8].fan_speed = rolls[8].fan_air_volume * rolls[8].fan_status / rolls[8].fan_area
        rolls[8].optiflex_angle = 3.0  # 前4台风机完

        rolls[9].roll_name = '5-1'
        rolls[9].roll_length = 9.252/2
        rolls[9].roll_v = 1.24
        rolls[9].t = 6.869/2
        rolls[9].step = int(rolls[9].t / simulation_model.dt)
        rolls[9].fan_air_volume = 53.889
        rolls[9].fan_status = 0
        rolls[9].fan_speed = rolls[9].fan_air_volume * rolls[9].fan_status / rolls[9].fan_area
        rolls[9].optiflex_angle = 1.5  # 后4台风机佳灵装置开度 1.5deg (Deng et al., 2025)
        rolls[10].roll_name = '5-2'
        rolls[10].roll_length = 9.252/2
        rolls[10].roll_v = 1.24
        rolls[10].t = 6.869/2
        rolls[10].step = int(rolls[10].t / simulation_model.dt)
        rolls[10].fan_air_volume = 53.889
        rolls[10].fan_status = 0
        rolls[10].fan_speed = rolls[10].fan_air_volume * rolls[10].fan_status / rolls[10].fan_area
        rolls[10].optiflex_angle = 1.5

        rolls[11].roll_name = '6-1'
        rolls[11].roll_length = 9.252/2
        rolls[11].roll_v = 1.11
        rolls[11].t = 6.479/2
        rolls[11].step = int(rolls[11].t / simulation_model.dt)
        rolls[11].fan_air_volume = 53.889
        rolls[11].fan_status = 0
        rolls[11].fan_speed = rolls[11].fan_air_volume * rolls[11].fan_status / rolls[11].fan_area
        rolls[11].optiflex_angle = 1.5
        rolls[12].roll_name = '6-2'
        rolls[12].roll_length = 9.252/2
        rolls[12].roll_v = 1.11
        rolls[12].t = 6.479/2
        rolls[12].step = int(rolls[12].t / simulation_model.dt)
        rolls[12].fan_air_volume = 53.889
        rolls[12].fan_status = 0
        rolls[12].fan_speed = rolls[12].fan_air_volume * rolls[12].fan_status / rolls[12].fan_area
        rolls[12].optiflex_angle = 1.5

        rolls[13].roll_name = '7-1'
        rolls[13].roll_length = 9.252/2
        rolls[13].roll_v = 1.10
        rolls[13].t = 6.479/2
        rolls[13].step = int(rolls[13].t / simulation_model.dt)
        rolls[13].fan_air_volume = 53.889
        rolls[13].fan_status = 0
        rolls[13].fan_speed = rolls[13].fan_air_volume * rolls[13].fan_status / rolls[13].fan_area
        rolls[13].optiflex_angle = 1.5
        rolls[14].roll_name = '7-2'
        rolls[14].roll_length = 9.252/2
        rolls[14].roll_v = 1.10
        rolls[14].t = 6.479/2
        rolls[14].step = int(rolls[14].t / simulation_model.dt)
        rolls[14].fan_air_volume = 53.889
        rolls[14].fan_status = 0
        rolls[14].fan_speed = rolls[14].fan_air_volume * rolls[14].fan_status / rolls[14].fan_area
        rolls[14].optiflex_angle = 1.5

        rolls[15].roll_name = '8-1'
        rolls[15].roll_length = 9.252/2
        rolls[15].roll_v = 0.88
        rolls[15].t = 6.479/2
        rolls[15].step = int(rolls[15].t / simulation_model.dt)
        rolls[15].fan_air_volume = 53.889
        rolls[15].fan_status = 0
        rolls[15].fan_speed = rolls[15].fan_air_volume * rolls[15].fan_status / rolls[15].fan_area
        rolls[15].optiflex_angle = 1.5
        rolls[16].roll_name = '8-2'
        rolls[16].roll_length = 9.252/2
        rolls[16].roll_v = 0.88
        rolls[16].t = 6.479/2
        rolls[16].step = int(rolls[16].t / simulation_model.dt)
        rolls[16].fan_air_volume = 53.889
        rolls[16].fan_status = 0
        rolls[16].fan_speed = rolls[16].fan_air_volume * rolls[16].fan_status / rolls[16].fan_area
        rolls[16].optiflex_angle = 1.5  # 后4台风机完

        rolls[17].roll_name = '9-1'
        rolls[17].roll_length = 9.252/2
        rolls[17].roll_v = 0.74
        rolls[17].t = 6.479/2
        rolls[17].step = int(rolls[17].t / simulation_model.dt)
        rolls[17].fan_air_volume = 0
        rolls[17].fan_status = 0
        rolls[17].fan_speed = rolls[17].fan_air_volume * rolls[17].fan_status / rolls[17].fan_area
        rolls[18].roll_name = '9-2'
        rolls[18].roll_length = 9.252/2
        rolls[18].roll_v = 0.74
        rolls[18].t = 6.479/2
        rolls[18].step = int(rolls[18].t / simulation_model.dt)
        rolls[18].fan_air_volume = 0
        rolls[18].fan_status = 0
        rolls[18].fan_speed = rolls[18].fan_air_volume * rolls[18].fan_status / rolls[18].fan_area

        rolls[19].roll_name = '10-1'
        rolls[19].roll_length = 9.252/2
        rolls[19].roll_v = 0.75
        rolls[19].t = 6.479/2
        rolls[19].step = int(rolls[19].t / simulation_model.dt)
        rolls[19].fan_air_volume = 0
        rolls[19].fan_status = 0
        rolls[19].fan_speed = rolls[19].fan_air_volume * rolls[19].fan_status / rolls[19].fan_area
        rolls[20].roll_name = '10-2'
        rolls[20].roll_length = 9.252/2
        rolls[20].roll_v = 0.75
        rolls[20].t = 6.479/2
        rolls[20].step = int(rolls[20].t / simulation_model.dt)
        rolls[20].fan_air_volume = 0
        rolls[20].fan_status = 0
        rolls[20].fan_speed = rolls[20].fan_air_volume * rolls[20].fan_status / rolls[20].fan_area

        rolls[21].roll_name = '11-1'
        rolls[21].roll_length = 9.252/2
        rolls[21].roll_v = 0.75
        rolls[21].t = 7.534/2
        rolls[21].step = int(rolls[21].t / simulation_model.dt)
        rolls[21].fan_air_volume = 0
        rolls[21].fan_status = 0
        rolls[21].fan_speed = rolls[21].fan_air_volume * rolls[21].fan_status / rolls[21].fan_area
        rolls[22].roll_name = '11-2'
        rolls[22].roll_length = 9.252/2
        rolls[22].roll_v = 0.75
        rolls[22].t = 7.534/2
        rolls[22].step = int(rolls[22].t / simulation_model.dt)
        rolls[22].fan_air_volume = 0
        rolls[22].fan_status = 0
        rolls[22].fan_speed = rolls[22].fan_air_volume * rolls[22].fan_status / rolls[22].fan_area

        rolls[23].roll_name = '12-1'
        rolls[23].roll_length = 9.252/2
        rolls[23].roll_v = 0.75
        rolls[23].t = 9.188/2
        rolls[23].step = int(rolls[23].t / simulation_model.dt)
        rolls[23].fan_air_volume = 0
        rolls[23].fan_status = 0
        rolls[23].fan_speed = rolls[23].fan_air_volume * rolls[23].fan_status / rolls[23].fan_area
        rolls[24].roll_name = '12-2'
        rolls[24].roll_length = 9.252/2
        rolls[24].roll_v = 0.75
        rolls[24].t = 9.188/2
        rolls[24].step = int(rolls[24].t / simulation_model.dt)
        rolls[24].fan_air_volume = 0
        rolls[24].fan_status = 0
        rolls[24].fan_speed = rolls[24].fan_air_volume * rolls[24].fan_status / rolls[24].fan_area

        rolls[25].roll_name = 'OUT'
        rolls[25].roll_length = 9.252
        rolls[25].roll_v = 0.81
        rolls[25].t = 3.311
        rolls[25].step = int(rolls[25].t / simulation_model.dt)
        rolls[25].fan_air_volume = 0
        rolls[25].fan_status = 0
        return rolls, n        

class simulation_model:  
    #仿真模型修改参数
    current_params = None


    #仿真算法参数
    N = 50  #算法网格数
    dt = 0.01   #算法时间步长，单位s
    dr = basic_info.r / N  #径向网格步长，单位m
    # latent_heat = 2.7 * 10**8  # 固态相变潜热, J/m^3
        
    #仿真过程参数
    #总体转变量
    f_total_0 = np.zeros(N)  #非搭接点总转变量
    f_total_1 = np.zeros(N)  #搭接点总转变量
    suma_f0 = np.zeros(N)  # 非搭接点铁素体孕育期累计
    #f_f0 = np.zeros(N)     # 非搭接点铁素体转变量
    mark_sf_0 = np.zeros(N, dtype=int)  #相变标记变量
    mark_ef_0 = np.zeros(N, dtype=int)

    suma_f1 = np.zeros(N)  # 搭接点铁素体孕育期累计
    #f_f1 = np.zeros(N)     # 搭接点铁素体转变量
    mark_sf_1 = np.zeros(N, dtype=int)  #相变标记变量
    mark_ef_1 = np.zeros(N, dtype=int)

    suma_p0 = np.zeros(N)  # 非搭接点珠光体孕育期累计
    #f_p0 = np.zeros(N)     # 非搭接点珠光体转变量
    mark_sp_0 = np.zeros(N, dtype=int)  #相变标记变量
    mark_ep_0 = np.zeros(N, dtype=int)

    suma_p1 = np.zeros(N)  # 搭接点珠光体孕育期累计
    #f_p1 = np.zeros(N)     # 搭接点珠光体转变量
    mark_sp_1 = np.zeros(N, dtype=int)  #相变标记变量
    mark_ep_1 = np.zeros(N, dtype=int)
    

    # 结果记录   
    history_time = []
    history_T_0 = [[] for _ in range(N)]  # 每个径向节点的温度历史
    history_T_1 = [[] for _ in range(N)]  # 每个径向节点的温度历史
    history_Q_0 = [[] for _ in range(N)]  # 每个径向节点的相变热焓历史
    history_Q_1 = [[] for _ in range(N)]  # 每个径向节点的相变热焓历史
    history_h_0 = [[] for _ in range(3)]  # 第一行为辐射换热，第二行为对流换热，第三行为总换热系数
    history_h_1 = [[] for _ in range(3)]  # 第一行为辐射换热，第二行为对流换热，第三行为总换热系数
    #珠光体转变量
    pearlite_0 = [[] for _ in range(N)]  #非搭接点珠光体转变量
    pearlite_1 = [[] for _ in range(N)]  #搭接点珠光体转变量

    @staticmethod
    def sample_with_step(history_time_raw, history_t0_raw, history_t1_raw, n=1):
        """按辊道起始时间及其相邻区间均匀采样。

        n 表示每两个相邻辊道起始时间之间额外均匀采样的点数，n=1 时等价于取中点。
        """
        # 该函数供其他程序调用。
        if n < 1:
            raise ValueError("n 必须大于等于 1。")

        min_len = min(len(history_time_raw), len(history_t0_raw), len(history_t1_raw))
        if min_len == 0:
            return [], [], [], []

        time_raw = np.asarray(history_time_raw[:min_len], dtype=float)
        t0_raw = np.asarray(history_t0_raw[:min_len], dtype=float)
        t1_raw = np.asarray(history_t1_raw[:min_len], dtype=float)

        roll_times = globals().get("roll_start_time", None)
        if not roll_times or len(roll_times) < 2:
            history_time = time_raw.tolist()
            history_t0 = t0_raw.tolist()
            history_t1 = t1_raw.tolist()
            num = list(range(len(history_time)))
            return history_time, history_t0, history_t1, num

        sample_time_candidates = []
        for i in range(len(roll_times) - 1):
            t_start = float(roll_times[i])
            t_end = float(roll_times[i + 1])
            sample_time_candidates.append(t_start)
            if n == 1:
                sample_time_candidates.append((t_start + t_end) / 2.0)
            else:
                sample_time_candidates.extend(np.linspace(t_start, t_end, n + 2)[1:-1].tolist())
        sample_time_candidates.append(float(roll_times[-1]))

        sample_time_candidates = sorted(sample_time_candidates)
        sample_times = []
        for t in sample_time_candidates:
            if not sample_times or abs(t - sample_times[-1]) > 1e-9:
                sample_times.append(t)

        sampled_time = []
        sampled_t0 = []
        sampled_t1 = []
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
                if abs(time_raw[right_idx] - target_time) < abs(time_raw[left_idx] - target_time):
                    idx = right_idx
                else:
                    idx = left_idx

            sampled_time.append(float(target_time))
            sampled_t0.append(float(t0_raw[idx]))
            sampled_t1.append(float(t1_raw[idx]))

        num = list(range(len(sampled_time)))
        return sampled_time, sampled_t0, sampled_t1, num

    @staticmethod
    def Calculate_physical_parameters(T_current):
        """
        更据当前温度计算随温度变化的物理参数（导热系数和比热容、空气运动粘度），单位摄氏度
        """
        #导热系数
        #[1]	Yafei S, Yongjun T, Jing S, et al. Effect of temperature and composition on thermal properties of carbon steel; proceedings of the 2009 Chinese Control and Decision Conference, F, 2009 [C]. IEEE.
        k = 57.4 - 0.0237*T_current - 10.3*basic_info.ELM_C*100 - 1.84e-5*T_current**2 + 0.0108*T_current*basic_info.ELM_C*100
        #比热容
        c = 499 + 0.0006*T_current**2 + 0.000892*(T_current/(basic_info.ELM_C*100)) - 2.61*(1/(basic_info.ELM_C*100))
        # k = 59.92- 0.0221*T_current-5.4e-5*T_current**2+4.3e-8*T_current**3
        # #珠光体比热容
        # c = 480.57 + 0.1226*T_current+8e-4*T_current**2-5e-7*T_current**3
        #------------------------------------------------------------------------------------------------------------------
        #相变热焓
        #珠光体相变潜热
        Hap = 120848 - 52.42*T_current - 0.158*T_current*T_current
        return k, c, Hap
    
    @staticmethod
    def Calculate_optitflex_parameters(optiflex_angle):
        """
        计算佳灵装置(Optiflex)的风速分配修正因子。

        参考: Deng T.W. et al. (2025), "Finite Element Simulation and Parameter
        Optimization of SWRH82B Wire Rod in Stelmor Cooling Process",
        J. Mater. Eng. Perform., Vol.34, pp.11212-11225.
        DOI: 10.1007/s11665-024-09898-2

        佳灵装置通过调节开合角度改变风道挡板位置，重新分配搭接点/非搭接点的
        风口宽度和风量比例。搭接点线材密集、自然散热较慢，佳灵装置增大搭接点
        风口宽度以增加该区域风量，使搭接点冷却速率接近非搭接点。

        参数:
        optiflex_angle : 佳灵装置开合角度 (deg)，范围 0-10
                         0 = 关闭(均匀分配), 3 = 论文推荐值, 10 = 最大开度

        返回:
        wind_factor_lap    : 搭接点风速修正因子 (>=1, 增大搭接点风速)
        wind_factor_nonlap : 非搭接点风速修正因子 (<=1, 减小非搭接点风速)
        """
        # 开合角度限幅 [0, 10]
        angle = max(0.0, min(10.0, optiflex_angle))

        # 风口宽度与开合角度的线性关系 (基于论文 Table 10 数据拟合)
        # 基准 (0deg): 搭接风口宽 325 mm, 非搭接风口宽 780 mm, 总宽 1430 mm
        # 斜率: dw/dtheta = (436.13 - 325) / 10 = 11.113 mm/deg
        w_lap_0 = 325.0
        w_nonlap_0 = 780.0
        dw_dtheta = 11.113  # mm/deg

        w_lap = w_lap_0 + dw_dtheta * angle
        w_nonlap = w_nonlap_0 - 2.0 * dw_dtheta * angle

        # 风量与风口宽度成正比，喷嘴面积不变
        # V(theta) / V(0deg) = w(theta) / w(0deg)
        wind_factor_lap = w_lap / w_lap_0
        wind_factor_nonlap = w_nonlap / w_nonlap_0

        return wind_factor_lap, wind_factor_nonlap
        

    @staticmethod
    def Calculate_thermal_cove_parameters(thermal_cove_status):
        """
        根据保温罩状态和温差计算保温罩相关参数，
        """
        pass
        # 保温罩的换热系数等参数计算逻辑
        # 这里需要根据保温罩的设计参数和工作原理进行具体实现
        # 例如，可以根据温差计算保温罩的绝热效果、换热系数等参数
        

    @staticmethod
    def H_calculation(T_0_surface, T_1_surface, T_air, phi, fan_air_volume, fan_status, fan_area, optiflex_angle=0):
        """
        [修正版] 计算风冷线换热系数。
        考虑佳灵装置(Optiflex)对搭接/非搭接点风速的主动调节，
        以及搭接点线材密集导致的被动换热折减(blocking_factor)。

        参数:
        T_0_surface    : 非搭接点表面温度 (C)
        T_1_surface    : 搭接点表面温度 (C)
        T_air          : 空气温度 (C)
        phi            : 线材直径 (m)
        fan_air_volume : 风机风量 (m3/s)
        fan_status     : 风机开启率 (0.0~1.0)
        fan_area       : 冷却面积 (m2)
        optiflex_angle : 佳灵装置开合角度 (deg)，0=无佳灵装置

        返回:
        h_conv_0 (非搭接对流), h_conv_1 (搭接对流),
        h_rad_0  (非搭接辐射), h_rad_1  (搭接辐射)
        """
        
        # === 1. 物理参数准备 ===
        epsilon = 1e-6

        # 标度化温度 (用于辐射公式)
        T_air_K_sc = (T_air + 273.15) / 100.0
        T_0_K_sc   = (T_0_surface + 273.15) / 100.0
        T_1_K_sc   = (T_1_surface + 273.15) / 100.0

        # 空气物性函数 (Sutherland定律)
        # 来源: Anderson (2006) Fundamentals of Aerodynamics; ANSYS FLUENT User Guide
        def calculate_k_air(T_kelvin):
            '''输入开尔文温度，返回空气导热系数 W/(m·K)'''
            return 2.495e-3 * (T_kelvin**1.5) / (T_kelvin + 194.0)

        def calculate_vair(T_kelvin):
            '''输入开尔文温度，返回空气运动粘度 m²/s'''
            return 4.13e-9 * (T_kelvin**2.5) / (T_kelvin + 110.4)

        # === 2. 流场计算 ===

        # 基础平均风速 (风机出口)
        u_avg = (fan_air_volume) * fan_status / fan_area

        # 搭接点被动换热折减系数 (无佳灵装置时搭接点线材密集、散热较慢)
        blocking_factor = 0.9

        # 佳灵装置(Optiflex)风速分配 — 主动调节搭接/非搭接风速
        # 来源: Deng et al. (2025), DOI: 10.1007/s11665-024-09898-2
        if optiflex_angle > 0:
            wf_lap, wf_nonlap = simulation_model.Calculate_optitflex_parameters(optiflex_angle)
        else:
            wf_lap, wf_nonlap = 1.0, 1.0

        u_0 = u_avg * wf_nonlap  # 非搭接点：经佳灵装置调节后的风速
        u_1 = u_avg * wf_lap     # 搭接点：经佳灵装置增大后的风速

        # === 3. 膜温度及空气物性 (Žukauskas关联式要求物性在膜温度评估) ===
        T_surf_K_0 = T_0_surface + 273.15
        T_surf_K_1 = T_1_surface + 273.15
        T_air_K = T_air + 273.15

        T_film_0 = (T_surf_K_0 + T_air_K) / 2.0
        T_film_1 = (T_surf_K_1 + T_air_K) / 2.0
        T_film_avg = (T_film_0 + T_film_1) / 2.0

        ka_film = calculate_k_air(T_film_avg)
        vair_film = calculate_vair(T_film_avg)

        # === 4. 普朗特数 (空气 Pr≈0.70, 在300-1100K变化<3%) ===
        Prf = 0.7
        Prs = 0.7

        # === 5. 强制对流: Zukauskas (1972) 单圆柱横流关联式 ===
        # 来源: Zukauskas, A. "Heat Transfer from Tubes in Crossflow."
        #       Advances in Heat Transfer, Vol.8, pp.93-160 (1972).
        #       DOI: 10.1016/S0065-2717(08)70038-8
        # 搭接点和非搭接点风速不同，分别计算 Re 和 Nu

        def calc_nu_zukauskas(Re):
            if Re < 40:
                C_zuk, m_zuk = 0.75, 0.4
            elif Re < 1000:
                C_zuk, m_zuk = 0.51, 0.5
            elif Re < 200000:
                C_zuk, m_zuk = 0.26, 0.6
            else:
                C_zuk, m_zuk = 0.076, 0.7
            return C_zuk * (Re**m_zuk) * (Prf**0.37) * ((Prf / Prs)**0.25)

        Re_0 = (u_0 * phi) / vair_film
        Re_1 = (u_1 * phi) / vair_film

        Nu_forced_0 = calc_nu_zukauskas(Re_0)
        Nu_forced_1 = calc_nu_zukauskas(Re_1)

        # === 6. 自然对流: Churchill & Chu (1975) 水平长圆柱 ===
        # 来源: Churchill, S.W. & Chu, H.H.S. "Correlating equations for laminar
        #       and turbulent free convection from a horizontal cylinder."
        #       Int. J. Heat Mass Transfer, Vol.18, pp.1049-1053 (1975).
        #       DOI: 10.1016/0017-9310(75)90222-7
        # 用于风机关闭段或低风速时的对流换热计算
        g = 9.81
        beta = 1.0 / T_film_avg  # 理想气体体积膨胀系数
        delta_T_avg = ((T_0_surface - T_air) + (T_1_surface - T_air)) / 2.0
        delta_T_avg = max(abs(delta_T_avg), epsilon)

        Gr = g * beta * delta_T_avg * (phi**3) / (vair_film**2)
        Ra = Gr * Prf

        if Ra > 0:
            # Churchill & Chu (1975) — 全 Ra 范围适用
            Nu_nat = (0.60 + 0.387 * (Ra**(1.0/6.0))
                      / ((1.0 + (0.559/Prf)**(9.0/16.0))**(8.0/27.0)))**2
        else:
            Nu_nat = 0.0

        # 取强制和自然对流中主导者 (风机关闭段强制对流弱时由自然对流接管)
        Nu_0 = max(Nu_forced_0, Nu_nat)
        Nu_1 = max(Nu_forced_1, Nu_nat)

        # === 7. 对流换热系数 ===
        # 非搭接点 (全风速, 无折减)
        hc0 = (ka_film / phi) * Nu_0

        # 搭接点 (佳灵装置调节风速 + blocking_factor 折减换热系数)
        # blocking_factor 反映搭接点线材密集导致的被动换热效率降低
        hc1 = (ka_film / phi) * Nu_1 * blocking_factor

        h_conv_0 = hc0
        h_conv_1 = hc1

        #手动修正
        params = simulation_model.current_params
        if params is not None:
            # 应用修正系数
            xs_hc0 = params.xs_hc0
            xs_hc1 = params.xs_hc1
            view_factor = params.view_factor #修正: 增加辐射遮挡系数
            # ... 其他计算 ...
        else:
            # 如果没传入参数，可以使用默认逻辑
            xs_hc0 = 1.0
            xs_hc1 = 1.0
            view_factor = 1.0 
        h_conv_0 = 1.5 * xs_hc0 * h_conv_0
        h_conv_1 = 1.5 * xs_hc1 * h_conv_1

        # === 4. 辐射换热系数计算 (h_rad) ===
        rad_coeff = 4.536
        
        # 非搭接点 (View Factor = 1.0)
        d_T0 = T_0_surface - T_air
        if abs(d_T0) < epsilon: h_rad_0 = 0
        else: h_rad_0 = rad_coeff * (pow(T_0_K_sc, 4) - pow(T_air_K_sc, 4)) / d_T0
        
        d_T1 = T_1_surface - T_air
        if abs(d_T1) < epsilon: h_rad_1 = 0
        else: h_rad_1 = view_factor * rad_coeff * (pow(T_1_K_sc, 4) - pow(T_air_K_sc, 4)) / d_T1

        return h_conv_0, h_conv_1, h_rad_0, h_rad_1
        
    
    @staticmethod
    def get_incubation_time_pearlite(T_current):
        """
        功能：计算珠光体的等温孕育期 (tau)。87B钢种
        T_current: 当前温度，单位摄氏度
        """
        if T_current > basic_info.Bs and T_current <= basic_info.A1:
            #kp = np.exp(10.164-16.002*basic_info.ELM_C-0.9797*basic_info.ELM_MN+0.0079*T_current-2.313/100000*T_current*T_current)
            kp = np.exp(10.164-16.002*basic_info.ELM_C -0.9797*basic_info.ELM_MN+0.00791*(T_current)-3.5067e-5 * (T_current)**2)
            tt = -0.91732*np.log(kp)+20*np.log(T_current)+1.9559*10000/(T_current)-157.45

            params = simulation_model.current_params
            if params is not None:
                # 应用修正系数
                xs_taup = params.xs_taup
                # ... 其他计算 ...
            else:
                # 如果没传入参数，可以使用默认逻辑
                xs_taup = 1

            tau = np.exp(tt) * xs_taup *0.5
            return tau
        else:
            return 1e9  # 不在珠光体转变温度范围内，不发生相变
        
    @staticmethod
    def get_incubation_time_ferrite(T_current):
        """
        功能：计算铁素体的等温孕育期 (tau)。87B钢种
        T_current: 当前温度，单位摄氏度
        """
        if T_current > basic_info.A1 and T_current <= basic_info.Acm:
            # Kf = np.exp(4.7766-13.339*basic_info.ELM_C-1.1922*basic_info.ELM_MN+0.02505*T_current-3.5067/100000*T_current*T_current)
            Kf = 14.2 * math.exp(-(T_current - 620) / 25.1)  # 简化K-V模型，缺少晶粒度和合金抑制项，对过共析钢影响小
            tt = -1.6454*np.log(Kf)+20*np.log(T_current) +3.265*10000/T_current -173.89

            params = simulation_model.current_params
            if params is not None:
                # 应用修正系数
                xs_tauf = params.xs_tauf
                # ... 其他计算 ...
            else:
                # 如果没传入参数，可以使用默认逻辑
                xs_tauf = 1

            tau = np.exp(tt) *xs_tauf
            return tau
        else:
            return 1e9  # 不在铁素体转变温度范围内，不发生相变

    @staticmethod
    def Q_calculation(T_vec_0, T_vec_1, Hap_0, Hap_1, time):
        """
        计算相变热焓Q，单位W/m^3
        T_vec 为开尔文温度数组
        Haf, Hap 为对应的相变潜热 (J/kg)
        """

        def process_phase_change(T_vec, time, positon):
            if positon == 0:
                suma_p = simulation_model.suma_p0
                f_p = simulation_model.f_total_0
                mark_sp = simulation_model.mark_sp_0
                mark_ep = simulation_model.mark_ep_0
                suma_f = simulation_model.suma_f0
                f_f = simulation_model.f_total_0
                mark_sf = simulation_model.mark_sf_0
                mark_ef = simulation_model.mark_ef_0
            elif positon == 1:
                suma_p = simulation_model.suma_p1
                f_p = simulation_model.f_total_1
                mark_sp = simulation_model.mark_sp_1
                mark_ep = simulation_model.mark_ep_1
                suma_f = simulation_model.suma_f1
                f_f = simulation_model.f_total_1
                mark_sf = simulation_model.mark_sf_1
                mark_ef = simulation_model.mark_ef_1

            else:
                print("Error: Invalid position for phase change processing.")

            Q_source = np.zeros(simulation_model.N)
            
            for i in range(simulation_model.N):
                T_k = T_vec[i]
                T_c = T_k - 273.15  # 必须使用摄氏度计算 JMAK 和 潜热
                dq = 0
                
                # --- 0. 超过Acm温度，无相变 ---
                if T_c > basic_info.Acm:
                    pass

                # --- 1. 先共析渗碳体/铁素体相变逻辑 (A1 < T <= Acm) ---
                elif T_c > basic_info.A1 and T_c <= basic_info.Acm:
                    if mark_ef[i] == 1: # 已经结束相变，确保热源为0
                        dq = 0
                    elif mark_ef[i] == 0: # 还未结束相变
                        # 孕育期判断
                        if mark_sf[i] == 0: # 如果没开始
                            tau = simulation_model.get_incubation_time_ferrite(T_c)
                            suma_f[i] += simulation_model.dt / tau
                            if suma_f[i] >= 0.99:
                                mark_sf[i] = 1 #标志开始相变
                                # sp_time = time #记录开始相变时间
                        
                        # 相变进行中 (应用 C++ 的 JMAK 逻辑)
                        elif mark_sf[i] == 1:
                            f_old = f_p[i]
                            
                            #Gemini铁素体JMAK公式: k(T) = 14.2 * exp(-(T - 620) / 25.1)，
                            n_jmak = 0.701
                            k_T_current = 14.2 * math.exp(-(T_c - 620) / 25.1)

                            # #孙的公式，k(T) = exp(4.7766-13.339*C-1.1922*Mn+0.02505*T-3.5067e-5*T^2)，这个公式在整个铁素体转变过程存在问题
                            # n_jmak = 1.0
                            # k_T_current = math.exp(4.7766-13.339*basic_info.ELM_C-1.1922*basic_info.ELM_MN+0.02505*T_c-3.5067e-5*T_c**2)
                            
                            ddd = 1.0 / n_jmak

                            # --- A. 计算等效时间 tn ---
                            if f_old > 0:
                                # A. 逆推等效时间 tn，必须用 k_T_current
                                # JMAK公式: f = 1 - exp(-k * t^n) -> t = [ -ln(1-f) / k ]^(1/n)
                                val = -math.log(1 - f_old) / k_T_current
                                val = max(val, 0.0) # 防止浮点误差
                                tn = math.pow(val, ddd)
                            else:
                                tn = 0.0
                            
                            # --- B. 计算经过当前步长后的总等效时间 tA ---
                            tA = tn + simulation_model.dt
                            
                            # --- C. 计算新的相变体积分数 f_new ---
                            ccc = n_jmak
                            k_T_fnew = k_T_current
                            f_new_calc = 1.0 - math.exp(-k_T_fnew * math.pow(tA, ccc))
                            
                            # 保证转变量单调递增且不超过上限
                            f_new = f_new_calc
                            
                            # --- D. 动态计算当前温度下的相变潜热 Haf ---
                            # 该公式仅在 T<263°C 时为正，在铁素体转变温区(727~Acm)恒为负值，
                            # 被下方 max(0,dq) 强制归零。对82A过共析钢影响小（先共析渗碳体量极少）。
                            # 亚共析钢需更换为正确的 γ→α 潜热公式（文献值约 16-20 kJ/kg）。
                            Haf_current = 20789 - 15.62 * T_c - 0.24 * (T_c ** 2)

                            # --- E. 更新记录并计算热源 dq ---
                            f_f[i] = f_new
                            real_df = f_new - f_old
                            
                            # 对应 CPP 的 qt = p*Haf*(Xf_new - Xf_old)/t0
                            dq = Haf_current * (real_df / simulation_model.dt) * basic_info.rho

                            #相变热焓修正系数
                            params = simulation_model.current_params
                            if params is not None:
                                # 应用修正系数
                                xs_dqf = params.xs_dqf
                                # ... 其他计算 ...
                            else:
                                # 如果没传入参数，可以使用默认逻辑
                                xs_dqf = 1
                            dq *= xs_dqf
                            
                            # 终极保护：哪怕极端温度导致 Haf 变负，也强制将其归 0 防止吸热崩溃
                            dq = max(0, dq)
                            
                            if f_new >= 0.99:
                                mark_ef[i] = 1 # 标记相变终了
                    

                # --- 2. 珠光体相变逻辑 (Bs < T <= A1) ---
                elif T_c > basic_info.Bs and T_c <= basic_info.A1:

                    if mark_ep[i] == 1: # 已经结束相变，确保热源为0
                        dq = 0
                    elif mark_ep[i] == 0: # 还未结束相变
                        # 孕育期判断
                        if mark_sp[i] == 0: # 如果没开始
                            tau = simulation_model.get_incubation_time_pearlite(T_c)
                            suma_p[i] += simulation_model.dt / tau
                            if suma_p[i] >= 0.99:
                                mark_sp[i] = 1 #标志开始相变
                                # sp_time = time #记录开始相变时间
                        
                        # 相变进行中 (应用 C++ 的 JMAK 逻辑)
                        elif mark_sp[i] == 1:
                            f_old = f_p[i]
                            
                            #Gemini珠光体JMAK公式: k(T) = 65.3 * exp(-(T - 595) / 4.2)，但这个公式在高温下会导致k过大，转变量瞬间达到1，失去物理意义
                            #n_jmak = 1.901
                            #k_T_current = 65.3 * math.exp(-(T_c - 595) / 4.2)

                            #孙公式：k(T) = exp(10.164-16.002*C-0.9797*Mn+0.00791*T-3.5067e-5*T^2)，这个公式在整个珠光体转变温度范围内都能得到合理的k值，转变量平滑过渡
                            n_jmak = 2
                            k_T_current = np.exp(10.164-16.002*basic_info.ELM_C -0.9797*basic_info.ELM_MN+0.00791*(T_c)-3.5067e-5 * (T_c)**2)
                            
                            ddd = 1.0 / n_jmak
                            
                            # --- A. 计算等效时间 tn ---
                            if f_old > 0:
                                # A. 逆推等效时间 tn，必须用 k_T_current
                                # JMAK公式: f = 1 - exp(-k * t^n) -> t = [ -ln(1-f) / k ]^(1/n)
                                val = -math.log(1 - f_old) / k_T_current
                                val = max(val, 0.0) # 防止浮点误差
                                tn = math.pow(val, ddd)
                            else:
                                tn = 0.0
                            
                            # --- B. 计算经过当前步长后的总等效时间 tA ---
                            tA = tn + simulation_model.dt
                            
                            # --- C. 计算新的相变体积分数 f_new ---
                            ccc = n_jmak
                            k_T_fnew = k_T_current
                            f_new_calc = 1.0 - math.exp(-k_T_fnew * math.pow(tA, ccc))
                            
                            # 保证转变量单调递增且不超过上限
                            f_new = f_new_calc
                            
                            # --- D. 动态计算当前温度下的珠光体相变潜热 Hap ---
                            Hap_current = 120848 - 52.42 * T_c - 0.158 * (T_c ** 2)

                            # --- E. 更新记录并计算热源 dq ---
                            f_p[i] = f_new
                            real_df = f_new - f_old

                            # 对应 CPP 的 qt = p*Hap*(Xp_new - Xp_old)/t0
                            dq = Hap_current * (real_df / simulation_model.dt) * basic_info.rho

                            #相变热焓修正系数
                            params = simulation_model.current_params
                            if params is not None:
                                # 应用修正系数
                                xs_dqp = params.xs_dqp
                                # ... 其他计算 ...
                            else:
                                # 如果没传入参数，可以使用默认逻辑
                                xs_dqp = 1
                
                            dq *= xs_dqp *1.5 # 原有额外 ×2 乘数已移除，如需补偿请通过 xs_dqp 修正系数调整
                            
                            # 终极保护：哪怕极端温度导致 Haf 变负，也强制将其归 0 防止吸热崩溃
                            dq = max(0, dq)
                                                    
                            if f_new >= 0.99:
                                mark_ep[i] = 1 # 标记相变终了
                    
                else:
                    pass  # 不在相变温度范围内，无相变    

                Q_source[i] = dq

            if positon == 0:
                simulation_model.suma_p0 = suma_p
                simulation_model.mark_sp_0 = mark_sp
                simulation_model.mark_ep_0 = mark_ep
                simulation_model.suma_f0 = suma_f   
                simulation_model.mark_sf_0 = mark_sf
                simulation_model.mark_ef_0 = mark_ef
                if T_c > basic_info.A1 and T_c <= basic_info.Acm:
                    simulation_model.f_total_0 = f_f
                elif T_c > basic_info.Bs and T_c <= basic_info.A1:
                    simulation_model.f_total_0 = f_p


            elif positon == 1:
                simulation_model.suma_p1 = suma_p
                simulation_model.mark_sp_1 = mark_sp
                simulation_model.mark_ep_1 = mark_ep
                simulation_model.suma_f1 = suma_f
                simulation_model.mark_sf_1 = mark_sf
                simulation_model.mark_ef_1 = mark_ef
                if T_c > basic_info.A1 and T_c <= basic_info.Acm:
                    simulation_model.f_total_1 = f_f
                elif T_c > basic_info.Bs and T_c <= basic_info.A1:
                    simulation_model.f_total_1 = f_p

            else:
                print("Error: Invalid position for phase change processing.")   

            return Q_source, f_p

        # 处理非搭接点 (0)
        Q_source_0 , f_p0= process_phase_change(T_vec_0, time, positon=0)
        
        # 处理搭接点 (1)
        Q_source_1 , f_p1= process_phase_change(T_vec_1, time, positon=1)

        return Q_source_0, Q_source_1, f_p0, f_p1


    @staticmethod
    def Cooling_calculation(roll):
        global each_roll_time
        
        # 1. 初始化当前温度场 (拷贝入口温度)
        # 注意：这里必须使用 .copy()，否则会修改传入的源数组
        T_field_0 = roll.pre_temp_0.copy() + 273.15
        T_field_1 = roll.pre_temp_1.copy() + 273.15
        
        # 预计算几何参数，避免循环内重复计算
        r_edge = basic_info.r
        area_out = 2 * np.pi * r_edge
        area_in = 2 * np.pi * (basic_info.r - simulation_model.dr)
        vol_edge = np.pi * (r_edge**2 - (basic_info.r - simulation_model.dr)**2) 

        for j in range(roll.step):
            current_time = (j + 1) * simulation_model.dt
            
            # --- 步骤 A: 物理参数更新 (基于当前时刻温度) ---
            # 使用当前表面的温度来计算换热系数
            T_surf_0 = T_field_0[-1] - 273.15
            T_surf_1 = T_field_1[-1] - 273.15
            
            # 使用当前平均温度计算物性 (k, c)
            T_avg_0 = np.mean(T_field_0) - 273.15 # 简化：两部分物性近似一致，或者分别计算
            T_avg_1 = np.mean(T_field_1) - 273.15
            k_val_0, c_0, Hap_0 = simulation_model.Calculate_physical_parameters(T_avg_0)
            k_val_1, c_1, Hap_1 = simulation_model.Calculate_physical_parameters(T_avg_1)
            
            # 计算换热系数
            h_c0, h_c1, h_r0, h_r1 = simulation_model.H_calculation(
                T_surf_0, T_surf_1, basic_info.T_air, basic_info.phi * 1e-3,
                roll.fan_air_volume, roll.fan_status, roll.fan_area, roll.optiflex_angle
            )
            h_0 = h_c0 + h_r0
            h_1 = h_c1 + h_r1

            # if simulation_model.mark_sp_0[1] ==1 and simulation_model.mark_ep_0[1] == 0:
            #     h_0 *= 0.5  # 珠光体相变进行中，表面温度被相变潜热钳制，强制换热系数为0
            # else:
            #     h_0 = h_0  # 正常计算换热系数

            # if simulation_model.mark_sp_1[1] ==1 and simulation_model.mark_ep_1[1] == 0:
            #     h_1 *= 0.5  # 珠光体相变进行中，表面温度被相变潜热钳制，强制换热系数为0
            # else:
            #     h_1 = h_1  # 正常计算换热系数


            # 计算相变热源 (基于当前温度场)
            # 注意：这里的输入应为摄氏度，且 Q_calculation 内部会更新 phase fraction 状态
            Q_0, Q_1 ,f_p0, f_p1= simulation_model.Q_calculation(T_field_0, T_field_1, Hap_0, Hap_1, each_roll_time + current_time)


            # --- 步骤 B: 差分求解 (分别求解 Non-overlap 和 Overlap) ---
            # 为了减少重复代码，定义一个内部求解过程
            def solve_step(T_current, h_val, Q_src, k_val, c):
                alpha = k_val / (basic_info.rho * c)
                Fo = alpha * simulation_model.dt / (simulation_model.dr**2)
                
                # 构建矩阵 (建议优化：提至循环外，仅更新边界)
                A = np.zeros((simulation_model.N, simulation_model.N))
                B = np.zeros(simulation_model.N)
                source_term = Q_src * simulation_model.dt / (basic_info.rho * c)

                # 内部节点
                indices = np.arange(1, simulation_model.N - 1)
                A[indices, indices-1] = -Fo * (1 - 1 / (2 * indices))
                A[indices, indices]   = 1 + 2 * Fo
                A[indices, indices+1] = -Fo * (1 + 1 / (2 * indices))
                B[indices] = T_current[indices] + source_term[indices]

                # 中心节点
                A[0, 0] = 1 + 4 * Fo
                A[0, 1] = -4 * Fo
                B[0]    = T_current[0] + source_term[0]

                # 表面节点
                term_cond = k_val * area_in / simulation_model.dr
                term_conv = h_val * area_out
                term_cap  = basic_info.rho * c * vol_edge / simulation_model.dt

                A[-1, -2] = -term_cond
                A[-1, -1] = term_cap + term_cond + term_conv
                B[-1]     = term_cap * T_current[-1] + \
                            term_conv * (basic_info.Ta + 273.15) + \
                            Q_src[-1] * vol_edge
                
                return np.linalg.solve(A, B)

            # 求解并更新状态
            T_field_0 = solve_step(T_field_0, h_0, Q_0, k_val_0, c_0)
            T_field_1 = solve_step(T_field_1, h_1, Q_1, k_val_1, c_1)

            # --- 步骤 C: 记录历史 ---
            # 记录时间 (为防止数据量过大，建议每隔几步记录一次，比如 if j % 10 == 0)
            simulation_model.history_time.append(each_roll_time + current_time)
            for row, val in zip(simulation_model.history_T_0, T_field_0 - 273.15):
                row.append(val)
            for row, val in zip(simulation_model.history_T_1, T_field_1 - 273.15):
                row.append(val)
            for row, val in zip(simulation_model.history_Q_0, Q_0):
                row.append(val)
            for row, val in zip(simulation_model.history_Q_1, Q_1):
                row.append(val)
            for row, val in zip(simulation_model.pearlite_0, f_p0):
                row.append(val)
            for row, val in zip(simulation_model.pearlite_1, f_p1):
                row.append(val)
            simulation_model.history_h_0[0].append(h_r0)
            simulation_model.history_h_0[1].append(h_c0)
            simulation_model.history_h_0[2].append(h_0)
            simulation_model.history_h_1[0].append(h_r1)
            simulation_model.history_h_1[1].append(h_c1)
            simulation_model.history_h_1[2].append(h_1)

        # 循环结束，保存该段出口温度
        roll.post_temp_0 = T_field_0 - 273.15
        roll.post_temp_1 = T_field_1 - 273.15
        
        # 更新全局时间（按实际仿真步数累加，保证时间戳为 dt 的整数倍）
        each_roll_time += roll.step * simulation_model.dt


    def plot_T_results():
        """
        绘制温度仿真结果
        """
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        plt.plot(simulation_model.history_time, simulation_model.history_T_0[-1], label='Surface Temp Non-Overlap (°C)')
        plt.plot(simulation_model.history_time, simulation_model.history_T_0[0], label='Center Temp Non-Overlap (°C)')
        plt.plot(simulation_model.history_time, simulation_model.history_T_1[-1], label='Surface Temp Overlap (°C)')
        plt.plot(simulation_model.history_time, simulation_model.history_T_1[0], label='Center Temp Overlap (°C)')

        # 使用 big_start_time 相邻分段添加交替浅色背景，便于分区观察且不干扰曲线
        bg_colors = ['#f6f8fb', '#fbf8f3']
        for i in range(len(big_start_time) - 1):
            ax.axvspan(big_start_time[i], big_start_time[i + 1],
                       facecolor=bg_colors[i % 2], alpha=1, zorder=0)

        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Profiles During Cooling')
        plt.legend()
        plt.grid()
        plt.vlines(small_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
           linestyles='--', colors='gray', alpha=0.7)
        plt.vlines(big_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
           linestyles='--', colors='black', alpha=0.7)
        #plt.show()

    def plot_H_results():
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        plt.plot(simulation_model.history_time, simulation_model.history_h_0[0], label='h_r Non-Overlap (°C)')
        plt.plot(simulation_model.history_time, simulation_model.history_h_0[1], label='h_c Non-Overlap (°C)')
        plt.plot(simulation_model.history_time, simulation_model.history_h_1[0], label='h_r Overlap (°C)')
        plt.plot(simulation_model.history_time, simulation_model.history_h_1[1], label='h_c Overlap (°C)')

        # 使用 big_start_time 相邻分段添加交替浅色背景，便于分区观察且不干扰曲线
        bg_colors = ['#f6f8fb', '#fbf8f3']
        for i in range(len(big_start_time) - 1):
            ax.axvspan(big_start_time[i], big_start_time[i + 1],
                       facecolor=bg_colors[i % 2], alpha=1, zorder=0)
            
        plt.xlabel('Time (s)')
        plt.ylabel('Heat Transfer Coefficient (W/m²·K)')
        plt.title('Heat Transfer Coefficient During Cooling')
        plt.legend()
        plt.grid()
        plt.vlines(small_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
           linestyles='--', colors='gray', alpha=0.7)
        plt.vlines(big_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
           linestyles='--', colors='black', alpha=0.7)
        #plt.show()

    def plot_Q_results():
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        plt.plot(simulation_model.history_time, simulation_model.history_Q_0[-1], label='surface_Q Non-Overlap (°C)')
        plt.plot(simulation_model.history_time, simulation_model.history_Q_1[-1], label='surface_Q Overlap (°C)')

         # 使用 big_start_time 相邻分段添加交替浅色背景，便于分区观察且不干扰曲线
        bg_colors = ['#f6f8fb', '#fbf8f3']
        for i in range(len(big_start_time) - 1):
            ax.axvspan(big_start_time[i], big_start_time[i + 1],
                       facecolor=bg_colors[i % 2], alpha=1, zorder=0)
            
        plt.xlabel('Time (s)')
        plt.ylabel('Q (W/m³)')
        plt.title('Q During Cooling')
        plt.legend()
        plt.grid()
        plt.vlines(small_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
           linestyles='--', colors='gray', alpha=0.7)
        plt.vlines(big_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
           linestyles='--', colors='black', alpha=0.7)
        #plt.show()

    def get_measure_point_T_results():
        '''
        该函数本体不调用，仅提供一个接口，方便外部调用获取每个测温点的仿真结果用于对比实际测量数据
        '''
        #记录每个大段的开始时间
        big_start_time = [roll_start_time[0], roll_start_time[1], roll_start_time[3], roll_start_time[5], roll_start_time[7], roll_start_time[9], roll_start_time[11], roll_start_time[13], roll_start_time[15], roll_start_time[17], roll_start_time[19], roll_start_time[21], roll_start_time[22]]

        #记录每个小段的开始时间
        small_start_time = [roll_start_time[2], roll_start_time[4], roll_start_time[6], roll_start_time[8], roll_start_time[10], roll_start_time[12], roll_start_time[14], roll_start_time[16], roll_start_time[18], roll_start_time[20]]

        #获得每个每个测温点的仿真结果
        measure_point_time = [big_start_time[0], big_start_time[1], small_start_time[0], big_start_time[2], small_start_time[1], big_start_time[3], small_start_time[2], big_start_time[4], small_start_time[3], big_start_time[5], small_start_time[4], big_start_time[6], small_start_time[5], big_start_time[7], small_start_time[6]]

        measure_point_time = measure_point_time[:8]

        # measure_point_time = [0.00, 3.77, 4.00, 5.00, 6.00, 7.00, 7.97, 8.00, 9.00, 10.00, 11.00, 12.00, 12.18, 13.00, 14.00, 15.00, 16.00, 16.04, 17.00, 18.00, 19.00, 19.89, 20.00, 21.00, 22.00, 23.00, 23.50, 24.00, 25.00, 26.00, 27.00, 27.10]

        if len(simulation_model.history_T_1[-1]) == 0:
            raise ValueError("history_T_1 为空，尚未完成仿真，无法提取测温点温度。")

        measure_point_T_sim1 = np.zeros(len(measure_point_time))
        measure_point_T_sim0 = np.zeros(len(measure_point_time))
        max_idx = len(simulation_model.history_T_1[-1]) - 1
        for i, t in enumerate(measure_point_time):
            idx = int(t / simulation_model.dt)
            if idx < 0:
                idx = 0
            elif idx > max_idx:
                idx = max_idx
            measure_point_T_sim1[i] = simulation_model.history_T_1[-1][idx]
            measure_point_T_sim0[i] = simulation_model.history_T_0[-1][idx]

        return measure_point_T_sim1, measure_point_T_sim0
    

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

    def plot_measure_point_T_results(metric_mode="MAE"):
        #记录每个实际测温点的时间
        measure_point_time = [big_start_time[0], big_start_time[1], small_start_time[0], big_start_time[2], small_start_time[1], big_start_time[3], small_start_time[2], big_start_time[4]]#, small_start_time[3], big_start_time[5], small_start_time[4], big_start_time[6], small_start_time[5], big_start_time[7], small_start_time[6], big_start_time[8], small_start_time[7], big_start_time[9], small_start_time[8], big_start_time[10], small_start_time[9], big_start_time[11], small_start_time[10], big_start_time[12], small_start_time[11]]

        measure_point_time_sim = simulation_model.history_time[:int(30/ simulation_model.dt)]  # 取前30秒的时间点，确保覆盖所有测温点的时间范围
        simulation_model.history_T_0[-1]
        measure_point_T_sim1 = simulation_model.history_T_1[-1][:int(30/ simulation_model.dt)]
        measure_point_T_sim0 = simulation_model.history_T_0[-1][:int(30/ simulation_model.dt)]
        #每个测温点的仿真结果
        # measure_point_T_sim1 = np.zeros(len(measure_point_time))
        # for i in range(len(measure_point_time)):
        #     measure_point_T_sim1[i] = simulation_model.history_T_1[-1][int(measure_point_time[i]/simulation_model.dt)]

        # measure_point_T_sim0 = np.zeros(len(measure_point_time))
        # for i in range(len(measure_point_time)):
        #     measure_point_T_sim0[i] = simulation_model.history_T_0[-1][int(measure_point_time[i]/simulation_model.dt)]

        #每个搭接点测温点的实际测量结果
        measure_point_T_real_1 = [845, 822, 718, 656, 650, 595, 575, 558]
        #每个非搭接点测温点的实际测量结果
        measure_point_T_real_0 = [830, 815, 705, 645, 630, 595, 568, 558]

        mode = str(metric_mode).upper()
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].plot(measure_point_time_sim, measure_point_T_sim1, label='Simulated Temp (°C)', marker='.', color='gray')

        if mode == "WMAE":
            x1 = np.array(measure_point_time[:len(measure_point_T_real_1)], dtype=np.float64)
            x0 = np.array(measure_point_time[:len(measure_point_T_real_0)], dtype=np.float64)
            weights1 = np.asarray(
                simulation_model.calculate_importance_for_real_point(x1, measure_point_T_real_1),
                dtype=np.float64,
            )
            weights0 = np.asarray(
                simulation_model.calculate_importance_for_real_point(x0, measure_point_T_real_0),
                dtype=np.float64,
            )
            norm = plt.Normalize(
                vmin=min(np.min(weights1), np.min(weights0)),
                vmax=max(np.max(weights1), np.max(weights0)),
            )
            cmap = plt.get_cmap("RdYlGn_r")  # 低权重=绿，高权重=红

            sc1 = axs[0].scatter(
                x1,
                measure_point_T_real_1,
                label='Actual Temp (°C)',
                marker='o',
                c=weights1,
                cmap=cmap,
                norm=norm,
                edgecolors='black',
                linewidths=0.6,
                s=80,
                zorder=3,
            )
            for x, y, w in zip(x1, measure_point_T_real_1, weights1):
                axs[0].text(x + 0.05, y + 2.0, f"{w:.3f}", fontsize=8, color='black')
        else:
            axs[0].scatter(
                measure_point_time[:len(measure_point_T_real_1)],
                measure_point_T_real_1,
                label='Actual Temp (°C)',
                marker='*',
                color='red',
                zorder=3,
            )

        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Temperature (°C)')
        axs[0].set_title('Temperature at Measurement Points 1')
        # plt.vlines(small_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
        #     linestyles='--', colors='gray', alpha=0.7)
        # plt.vlines(big_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
        #     linestyles='--', colors='black', alpha=0.7)
        axs[1].plot(measure_point_time_sim, measure_point_T_sim0, label='Simulated Temp (°C)', marker='.', color='gray')
        if mode == "WMAE":
            sc0 = axs[1].scatter(
                x0,
                measure_point_T_real_0,
                label='Actual Temp (°C)',
                marker='o',
                c=weights0,
                cmap=cmap,
                norm=norm,
                edgecolors='black',
                linewidths=0.6,
                s=80,
                zorder=3,
            )
            for x, y, w in zip(x0, measure_point_T_real_0, weights0):
                axs[1].text(x + 0.05, y + 2.0, f"{w:.3f}", fontsize=8, color='black')
            cbar = fig.colorbar(sc0, ax=axs.ravel().tolist(), shrink=0.92, pad=0.02)
            cbar.set_label('Real-point Weight (low=green, high=red)')
        else:
            axs[1].scatter(
                measure_point_time[:len(measure_point_T_real_0)],
                measure_point_T_real_0,
                label='Actual Temp (°C)',
                marker='*',
                color='red',
                zorder=3,
            )

        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Temperature (°C)')
        axs[1].set_title('Temperature at Measurement Points 0')
        axs[0].legend(loc='upper right')
        axs[1].legend(loc='upper right')
        #plt.show()


    def plot_trans_stage():
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        plt.plot(simulation_model.history_time, simulation_model.pearlite_0[-1], label='Surface trans Non-Overlap')
        plt.plot(simulation_model.history_time, simulation_model.pearlite_1[-1], label='Surface trans Overlap')
        plt.plot(simulation_model.history_time, simulation_model.pearlite_0[0], label='Center trans Non-Overlap')
        plt.plot(simulation_model.history_time, simulation_model.pearlite_1[0], label='Center trans Overlap')

         # 使用 big_start_time 相邻分段添加交替浅色背景，便于分区观察且不干扰曲线
        bg_colors = ['#f6f8fb', '#fbf8f3']
        for i in range(len(big_start_time) - 1):
            ax.axvspan(big_start_time[i], big_start_time[i + 1],
                       facecolor=bg_colors[i % 2], alpha=1, zorder=0)
            
        plt.xlabel('Time (s)')
        plt.ylabel('X')
        plt.title('trans_stage During Cooling')
        plt.legend()
        plt.grid()
        plt.vlines(small_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
           linestyles='--', colors='gray', alpha=0.7)
        plt.vlines(big_start_time, ymin=plt.ylim()[0], ymax=plt.ylim()[1],
           linestyles='--', colors='black', alpha=0.7)
        
    def save_sim_data_as_csv():
        "将仿真结果保存为 CSV 文件，方便后续分析和对比，时间；非搭接点表面；搭接点表面"
        # 获取当前 .py 文件所在的文件夹路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接出完整文件路径
        csv_path = os.path.join(script_dir, "sim_data.csv")
        # 以写入模式打开文件，newline='' 避免CSV产生多余空行，utf-8编码兼容中文
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            # 创建CSV写入器
            writer = csv.writer(csv_file)

            history_time = simulation_model.history_time
            history_T_0 = simulation_model.history_T_0[-1]
            history_T_1 = simulation_model.history_T_1[-1]
            
            # 第一行：写入 history_time 数组（直接传入数组即可）
            writer.writerow(history_time)
            # 第二行：写入 history_T_0 的最后一个元素（包装成列表，保证是一行）
            writer.writerow(history_T_0)
            # 第三行：写入 history_T_1 的最后一个元素
            writer.writerow(history_T_1)
            
            # print(type(simulation_model.history_T_0[-1]))

        print("文件保存成功！路径：当前目录下的 sim_data.csv")



if __name__ == "__main__":
    # 仿真模型修改参数（从类定义迁移到程序入口，便于集中管理）
    simulation_model.current_params = build_parameter_from_file(0)

    rolls, num_rolls = data_loader.load_roll_data()
    tem1 = 850  # 入口温度
    tem0 = 830
    temp_trans_1 = np.full(simulation_model.N, tem1)
    temp_trans_0 = np.full(simulation_model.N, tem0)

    each_roll_time = 0
    #记录每个辊道的开始时间
    roll_start_time = []
    roll_start_time.append(each_roll_time)

    for i in range(num_rolls):
        current_roll = rolls[i]
        current_roll.pre_temp_0 = temp_trans_0
        current_roll.pre_temp_1 = temp_trans_1
        simulation_model.Cooling_calculation(current_roll)
        temp_trans_0 = current_roll.post_temp_0
        temp_trans_1 = current_roll.post_temp_1

        #记录该辊道开始和结束时间
        roll_start_time.append(each_roll_time)
    
    #记录每个大段的开始时间
    big_start_time = [roll_start_time[0], roll_start_time[1], roll_start_time[3], roll_start_time[5], roll_start_time[7], roll_start_time[9], roll_start_time[11], roll_start_time[13], roll_start_time[15], roll_start_time[17], roll_start_time[19], roll_start_time[21], roll_start_time[23], roll_start_time[25], roll_start_time[26]]

    #记录每个小段的开始时间
    small_start_time = [roll_start_time[2], roll_start_time[4], roll_start_time[6], roll_start_time[8], roll_start_time[10], roll_start_time[12], roll_start_time[14], roll_start_time[16], roll_start_time[18], roll_start_time[20], roll_start_time[22], roll_start_time[24]]

    # simulation_model.save_sim_data_as_csv()
    simulation_model.plot_T_results()
    simulation_model.plot_H_results()
    simulation_model.plot_Q_results()
    simulation_model.plot_measure_point_T_results(metric_mode="WMAE")
    simulation_model.plot_trans_stage()
    plt.show()

    
