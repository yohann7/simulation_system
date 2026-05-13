# sim_T.py 代码改动与参考文献文档

> 改动日期：2026-05-13
> 基于：Q_calculation_formula_analysis.md / 82A_steel_phase_transformation_references.md
> 参考论文：Deng et al. (2025), J. Mater. Eng. Perform., DOI: 10.1007/s11665-024-09898-2

---

## 第一部分：所有改动内容

### 1. 临界温度公式修正（`basic_info` 类，第41-43行）

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| 上临界点 | `A3 = 820`（固定值，物理含义错误） | `Acm = 727 + 314.2*(ELM_C*100 - 0.77)`（过共析钢上临界点） |
| A1 温度 | `727 - 10.7*ELM_MN - 16.9*ELM_NI + 16*ELM_CR + 29.1*ELM_SI`（元素含量输入单位错误：质量分数而非 wt%） | `727 - 10.7*(ELM_MN*100) - 16.9*(ELM_NI*100) + 16*(ELM_CR*100) + 29.1*(ELM_SI*100)`（乘以 100 转换为 wt%） |
| Bs 温度 | `Bs = 500`（固定值，无成分依赖） | `Bs = 830 - 270*(ELM_C*100) - 90*(ELM_MN*100) - 37*(ELM_NI*100) - 70*(ELM_CR*100)`（Steven-Haynes 成分依赖公式） |

全局所有 `basic_info.A3` 引用（共5处：第660、718、722、889、902行）同步替换为 `basic_info.Acm`。

---

### 2. 相变潜热公式修正（`Q_calculation` 方法）

| 位置 | 修改前 | 修改后 |
|------|--------|--------|
| 珠光体潜热 ×2（第869行） | `dq *= xs_dqp * 2` | `dq *= xs_dqp`（移除无物理依据的额外乘数，注释说明如需补偿可通过 xs_dqp 修正系数调整） |
| 珠光体潜热变量名（第849-857行） | 变量名 `Haf_current`（与铁素体混淆），注释引用 `Xf_new` | 变量名 `Hap_current`，注释引用 `Xp_new` |
| 铁素体潜热公式（第771行） | 无注释说明 | 添加注释：该公式在 T<263°C 才为正，铁素体转变温区 (727°C~Acm) 恒为负值，被 `max(0,dq)` 强制归零。对 82A 过共析钢影响小。亚共析钢需更换为正确的 γ→α 潜热公式（文献值约 16-20 kJ/kg） |
| 铁素体孕育期 Kf 公式（第662行） | 无注释说明 | 添加注释：简化 K-V 模型，缺少晶粒度和合金抑制项，对过共析钢影响小 |

---

### 3. 换热系数计算公式重写（`H_calculation` 方法，第558-680行）

#### 3.1 空气运动粘度 — Sutherland 系数修正

```python
# 修改前
vair = 4.02e-10 * T^2.5 / (T + 110.4)   # 系数偏小约 10 倍

# 修改后
vair = 4.13e-9 * T^2.5 / (T + 110.4)    # 标准 Sutherland 运动粘度
```

#### 3.2 空气导热系数 — 从常数改为温度依赖

```python
# 修改前
ka = 0.026  # 固定值 (仅 25°C 正确)

# 修改后
k_air(T) = 2.495e-3 * T^1.5 / (T + 194.0)   # Sutherland 导热系数，在膜温度评估
```

#### 3.3 Žukauskas 强制对流关联式 — 恢复标准系数

```python
# 修改前
Nu = 0.10 * Re^0.6 * Pr^0.38 * (Pr/Prs)^0.25   # C=0.10 无文献依据

# 修改后 (按 Re 分段恢复标准 Žukauskas (1972) 系数)
Re < 40:        C=0.75, m=0.4
40 ≤ Re < 1000:   C=0.51, m=0.5
1000 ≤ Re < 2×10⁵: C=0.26, m=0.6
Re ≥ 2×10⁵:      C=0.076, m=0.7
Nu = C * Re^m * Pr^0.37 * (Pr/Prs)^0.25
```

#### 3.4 物性评估温度 — 从空气温度改为膜温度

```python
# 修改前
vairf = calculate_vair(T_air)   # 在环境温度 25°C 评估

# 修改后
T_film_avg = [(T_s0+T_air)/2 + (T_s1+T_air)/2] / 2 [K]
ka_film = calculate_k_air(T_film_avg)
vair_film = calculate_vair(T_film_avg)
```

#### 3.5 新增自然对流（Churchill & Chu, 1975）

```python
# 风机关闭或低风速段由自然对流接管
Ra = Gr · Pr
Nu_nat = (0.60 + 0.387·Ra^(1/6) / (1+(0.559/Pr)^(9/16))^(8/27))²
Nu = max(Nu_forced, Nu_nat)
```

#### 3.6 搭接点/非搭接点分别计算 Re 和 Nu

```python
# 修改前
u_0 = u_avg; u_1 = u_avg * blocking_factor
Re = (u_avg * D) / ν  (仅计算一个 Re)
hc0 = hc; hc1 = hc * blocking_factor

# 修改后 (支持佳灵装置不同风速)
u_0 = u_avg * wf_nonlap; u_1 = u_avg * wf_lap
Re_0 = (u_0 * D) / ν; Re_1 = (u_1 * D) / ν
Nu_0 = f(Re_0); Nu_1 = f(Re_1)
hc0 = k/D · Nu_0
hc1 = k/D · Nu_1 · blocking_factor
```

---

### 4. 佳灵装置 (Optiflex) 实现

#### 4.1 `roll` 类新增属性（第115行）

```python
self.optiflex_angle = 0  # 佳灵装置开合角度 (deg)，0=关闭/无装置
```

#### 4.2 `Calculate_optitflex_parameters` 函数实现（第502-543行）

从空的骨架函数实现为完整的风速因子计算：

```python
w_lap(θ)     = 325 + 11.113 × θ       (mm)
w_nonlap(θ)  = 780 − 22.226 × θ       (mm)
wf_lap       = w_lap(θ) / 325         (≥ 1)
wf_nonlap    = w_nonlap(θ) / 780      (≤ 1)
```

#### 4.3 `data_loader` 佳灵装置默认值

| 辊道段 | 对应风机 | optiflex_angle |
|--------|---------|----------------|
| 1-1 ~ 4-2 | 前 4 台风机 (200,000 m³/h) | **3.0°** |
| 5-1 ~ 8-2 | 后 4 台风机 (168,000 m³/h) | **1.5°** |
| IN, 9-1 ~ OUT | 无风机 | 0（默认） |

#### 4.4 `Cooling_calculation` 传递 optiflex_angle（第1041行）

```python
h_c0, h_c1, h_r0, h_r1 = simulation_model.H_calculation(
    T_surf_0, T_surf_1, basic_info.T_air, basic_info.phi * 1e-3,
    roll.fan_air_volume, roll.fan_status, roll.fan_area, roll.optiflex_angle
)
```

#### 4.5 heat transfer 机制说明

- **blocking_factor = 0.9**：作用于搭接点换热系数 `hc1`，代表线材密集导致的**被动**换热效率降低（自然物理现象）
- **optiflex_angle**：通过佳灵装置挡板角度**主动**调节搭接点/非搭接点的风速分配，增大搭接点风速以补偿密度效应
- 两者共同作用使 `hc_lap ≈ hc_nonlap`，实现均匀冷却

---

## 第二部分：改动后所用公式的参考文献

### [1] A1 临界温度 — Andrews 经验公式

- **标题**: Equations for the Calculation of the A1 and A3 Temperatures
- **作者**: Andrews, K.W.
- **期刊**: *Journal of The Iron and Steel Institute*, Vol. 203, pp. 721-727 (1965)
- **公式**: `Ac1(°C) = 723 − 10.7·Mn − 16.9·Ni + 29.1·Si + 16.9·Cr`

### [2] Acm 临界温度 — Fe-C 相图

- **来源**: Fe-C 二元相图 ES 线（渗碳体在奥氏体中的溶解度曲线）
- **公式**: `Acm(°C) ≈ 727 + 314.2 × (C_wt% − 0.77)`

### [3] Bs 贝氏体开始温度 — Steven-Haynes 公式

- **标题**: The Temperature of Formation of Martensite and Bainite in Low-Alloy Steels
- **作者**: Steven, W. & Haynes, A.G.
- **期刊**: *Journal of The Iron and Steel Institute*, Vol. 183, pp. 349-359 (1956)
- **公式**: `Bs(°C) = 830 − 270·C − 90·Mn − 37·Ni − 70·Cr − 83·Mo`

### [4] 铁素体相变动力学 — Kirkaldy-Venugopalan 模型

- **标题**: Prediction of Microstructure and Hardenability in Low Alloy Steels
- **作者**: Kirkaldy, J.S. & Venugopalan, D.
- **会议**: *International Conference on Phase Transformations in Ferrous Alloys*, Philadelphia, pp. 125-148 (1983)

### [5] 导热系数与比热容 — Sun Yafei et al.

- **标题**: Effect of Temperature and Composition on Thermal Properties of Carbon Steel
- **作者**: Yafei, S., Yongjun, T., Jing, S. et al.
- **会议**: *2009 Chinese Control and Decision Conference (CCDC)*, pp. 3756-3760 (2009)
- **DOI**: [10.1109/CCDC.2009.5191721](https://doi.org/10.1109/CCDC.2009.5191721)

### [6] 珠光体相变潜热 — Agarwal & Brimacombe

- **标题**: Mathematical Model of Heat Flow and Austenite-Pearlite Transformation in Eutectoid Carbon Steel Rods for Wire
- **作者**: Agarwal, P.K. & Brimacombe, J.K.
- **期刊**: *Metallurgical Transactions B*, Vol. 12, pp. 121-133 (1981)
- **DOI**: [10.1007/BF02674765](https://doi.org/10.1007/BF02674765)
- **参考值**: γ→P 总潜热约 77-85 kJ/kg

### [7] 空气运动粘度 — Sutherland 定律

- **原始文献**: Sutherland, W. (1893), "The Viscosity of Gases and Molecular Force", *Philosophical Magazine*, Series 5, Vol. 36, pp. 507-531
- **标准形式**: `μ = 1.458×10⁻⁶ × T^1.5 / (T + 110.4)` [kg/(m·s)]
- **来源**: ANSYS FLUENT User Guide §7.5.2; Anderson, J.D. (2006), *Fundamentals of Aerodynamics*, 5th ed., McGraw-Hill
- **代码形式**（运动粘度）: `ν = 4.13×10⁻⁹ × T^2.5 / (T + 110.4)` [m²/s]

### [8] 空气导热系数 — Sutherland 定律

- **来源**: COMSOL CFD Module User's Guide; OpenFOAM Sutherland transport model
- **标准形式**: `k = 2.495×10⁻³ × T^1.5 / (T + 194)` [W/(m·K)]
- **Anderson, J.D.** (2006), *Fundamentals of Aerodynamics*, 5th ed., McGraw-Hill

### [9] 强制对流 — Žukauskas 圆柱横流关联式

- **标题**: Heat Transfer from Tubes in Crossflow
- **作者**: Žukauskas, A.
- **期刊/书**: *Advances in Heat Transfer*, Vol. 8, pp. 93-160 (1972)
- **DOI**: [10.1016/S0065-2717(08)70038-8](https://doi.org/10.1016/S0065-2717(08)70038-8)
- **公式**: `Nu_D = C · Re_D^m · Pr^n · (Pr/Pr_s)^0.25`
  - Re < 40: C=0.75, m=0.4
  - 40 ≤ Re < 1000: C=0.51, m=0.5
  - 10³ ≤ Re < 2×10⁵: C=0.26, m=0.6
  - Re ≥ 2×10⁵: C=0.076, m=0.7
  - n = 0.37 (Pr ≤ 10)

### [10] 自然对流 — Churchill & Chu 水平圆柱关联式

- **标题**: Correlating Equations for Laminar and Turbulent Free Convection from a Horizontal Cylinder
- **作者**: Churchill, S.W. & Chu, H.H.S.
- **期刊**: *International Journal of Heat and Mass Transfer*, Vol. 18, pp. 1049-1053 (1975)
- **DOI**: [10.1016/0017-9310(75)90222-7](https://doi.org/10.1016/0017-9310(75)90222-7)
- **公式**: `Nu = [0.60 + 0.387·Ra^(1/6) / (1 + (0.559/Pr)^(9/16))^(8/27)]²`

### [11] 辐射换热 — Stefan-Boltzmann 定律 + 氧化钢发射率

- **Stefan-Boltzmann 常数**: σ = 5.67×10⁻⁸ W/(m²·K⁴)
- **氧化钢表面发射率**: ε ≈ 0.8（工程常用值）
- **代码实现**: `rad_coeff = 4.536 = ε × σ × 10⁸`，配合 T/100 温度标度化

### [12] 佳灵装置 (Optiflex) — Deng et al.

- **标题**: Finite Element Simulation and Parameter Optimization of SWRH82B Wire Rod in Stelmor Cooling Process
- **作者**: Deng, T.W., Cui, F., Tang, Z.Y., Cao, D.D., Tang, W. & Zeng, M.
- **期刊**: *Journal of Materials Engineering and Performance*, Vol. 34, pp. 11212-11225 (2025)
- **DOI**: [10.1007/s11665-024-09898-2](https://doi.org/10.1007/s11665-024-09898-2)
- **关键结论**:
  - 前 4 台风机佳灵装置开度 3° → 搭接/非搭接温差仅 4°C
  - 后 4 台风机佳灵装置开度 1.5° → 全过程温差 5-6°C
  - 风口宽度与开合角度呈线性关系（Table 10 数据）

### [13] Stelmor 冷却线传热 — Campbell, Hawbolt & Brimacombe

- **标题**: Microstructural Engineering Applied to the Controlled Cooling of Steel Wire Rod: Part I. Experimental Design and Heat Transfer
- **作者**: Campbell, P.C., Hawbolt, E.B. & Brimacombe, J.K.
- **期刊**: *Metallurgical Transactions A*, Vol. 22, pp. 2769-2778 (1991)
- **DOI**: [10.1007/BF02851371](https://doi.org/10.1007/BF02851371)
- **内容**: Stelmor 冷却线传热系数实验测定与对流/辐射关联式验证，辐射遮挡修正因子

---

*文档生成日期：2026-05-13*
