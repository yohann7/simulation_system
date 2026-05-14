# Puma（美洲豹）优化算法 —— 架构与原理

## 1. 算法来源

Puma 算法（Puma Optimizer, PO）是一种基于美洲豹捕食行为的元启发式优化算法。美洲豹在捕猎时交替使用两种策略：

- **探索（Exploration）**：大范围巡回搜索猎物，通过随机跳跃和迂回接近目标
- **开发（Exploitation）**：锁定猎物后短距离冲刺、伏击

算法通过**自适应分数机制**动态选择每轮迭代采用哪种策略，模拟捕食者在不同环境下的行为切换。

---

## 2. 算法架构

```
puma_optimize(nSol, MaxIter, lb, ub, dim, CostFunction, BatchCostFunction, patience)
│
├── 种群初始化（均匀随机分布）
│
├── 无经验阶段（Iter 1-3）：每轮同时执行探索+开发，收集统计数据
│   ├── _exploration_phase()  → 差分变异 + 交叉
│   └── _exploitation_phase() → 最优引导 + 随机扰动
│
├── 有经验阶段（Iter 4-MaxIter）：根据分数自适应选择
│   ├── Score_Explore > Score_Exploit → _exploration_phase()
│   ├── Score_Exploit > Score_Explore → _exploitation_phase()
│   ├── 更新分数（F1/F2/F3 + Mega/lmn 自适应权重）
│   └── 早停检测（patience 轮无改进 → 终止）
│
└── 返回 (最优位置, 最优代价, 收敛曲线)
```

### 核心数据结构

```
Solution:
    X:    ndarray[dim]   # 候选解位置向量
    Cost: float          # 适应度（代价），越小越优
```

---

## 3. 探索阶段（Exploration Phase）

**目标**：全局搜索，保持种群多样性，跳出局部最优。

**流程**：

```
对种群按 Cost 升序排序
pCR = 0.2, p = (1-pCR) / nSol

for i = 0 .. nSol-1:
    # 1. 从种群中随机选择 6 个不同于 i 的个体 a,b,c,d,e,f
    A = random_permutation([0..nSol-1] \ {i})[:6]

    # 2. 生成变异向量 y
    if rand() < 0.5:
        y = Uniform(lb, ub)                    # 50% 概率：纯随机探索
    else:
        G = Uniform(-1, 1)                     # 差分缩放因子
        y = x_a + G*(x_a - x_b)
          + G*((x_a-x_b) - (x_c-x_d))
          + G*((x_c-x_d) - (x_e-x_f))         # 三层差分变异

    # 3. 二项式交叉（Binomial Crossover）
    z = x_i
    j0 = random_index(dim)
    for j in 0..dim-1:
        if j == j0 or rand() <= 1-pCR:        # 至少一个维度来自变异
            z[j] = clamp(y[j], lb, ub)
        else:
            z[j] = x_i[j]

    # 4. 贪婪选择（精英保留）
    if Cost(z) < Cost(x_i):
        替换 x_i
    else:
        pCR += p                               # 失败则提高交叉率，增加探索力度
```

**关键参数**：
- `pCR`: 交叉概率（初始 0.2），选择失败时自适应增大，加速探索
- `G`: 差分缩放因子 ∈ [-1, 1]
- 三层嵌套差分：`(a-b)`, `(a-b)-(c-d)`, `(c-d)-(e-f)` 叠加增强扰动幅度

---

## 4. 开发阶段（Exploitation Phase）

**目标**：局部搜索，利用当前最优解引导收敛。

**流程**：

```
Q = 0.67, Beta = 2
mbest = mean(X_population, axis=0)       # 种群质心

for i = 0 .. nSol-1:
    # 1. 计算引导向量
    beta1 = Uniform(0, 2)
    beta2 = Normal(0, 1, dim)
    w, v = Normal(0, 1, dim), Normal(0, 1, dim)
    F1 = Normal(0, 1, dim) * exp(2 - Iter*(2/MaxIter))  # 指数衰减扰动
    F2 = w * v^2 * cos(2*rand()*w)                       # 非线性扰动

    R1 = Uniform(-1, 1)
    S1 = Uniform(-1, 1) + Normal(0, 1, dim)
    S2 = F1 * R1 * x_i + F2 * (1-R1) * Best.X
    VEC = S2 / S1                            # 综合引导方向

    # 2. 生成新解（两种策略随机选择）
    if rand() <= 0.5:
        Xatack = VEC
        if rand() > Q:                       # 33% 概率：随机个体跳跃
            r = random_index(nSol)
            new_x = Best.X + beta1*exp(beta2) * (x_r - x_i)
        else:                                # 67% 概率：攻击向量引导
            new_x = beta1 * Xatack - Best.X
    else:
        r1 = random_index(nSol)              # 种群质心策略
        sign = (-1)^random(0,1)
        new_x = (mbest * x_r1 - sign * x_i) / (1 + Beta*rand())

    # 3. 夹持 + 贪婪选择
    new_x = clamp(new_x, lb, ub)
    if Cost(new_x) < Cost(x_i):
        替换 x_i
```

**关键参数**：
- `F1`: 指数衰减项 `exp(2 - 2*Iter/MaxIter)`，随迭代步数增大而减小 → 搜索半径逐步收缩
- `Beta = 2`: 质心策略的分母缩放因子，控制偏离幅度
- `Q = 0.67`: 攻击模式切换阈值

**设计哲学**：F1 项实现了从大范围局部搜索到精细微调的平滑过渡（类比模拟退火的温度下降）。

---

## 5. 自适应阶段选择机制

这是 Puma 算法区别于传统 DE/PSO 的核心创新。

### 5.1 分数计算（三组分加权）

每轮迭代后更新探索和开发的评分：

```
F1  = PF[0] * (最新改进量 / 时间消耗)        # 瞬时效率（近期表现）
F2  = PF[1] * (累计改进量 / 累计时间消耗)    # 累计效率（长期表现）
F3  = PF[2]                                   # 最小改进量基准

Score = Mega * F1 + Mega * F2 + lmn * min(PF_F3) * F3
```

其中：
- `PF = [0.5, 0.5, 0.3]`：各组分的先验权重
- `Mega ∈ [0.01, 0.99]`：自适应权重，表现更好的一方逐渐增大
- `lmn = 1 - Mega`：互补权重
- `PF_F3`：历史所有非零改进量的集合

### 5.2 Mega 权重自适应更新

```
if Score_Explore < Score_Exploit:      # 探索表现更好
    Mega_Explore = max(Mega_Explore - 0.01, 0.01)  # 降低
    Mega_Exploit = 0.99                             # 提高对方
else:                                  # 开发表现更好
    Mega_Explore = 0.99
    Mega_Exploit = max(Mega_Exploit - 0.01, 0.01)
```

**设计思路**：采用"抑强扶弱"策略——当前表现较好的阶段降低其 Mega 权重（其对立面 lmn 升高），给予 F3 最小改进量项更大的话语权。这避免了某种策略长期垄断选择，维持了探索-开发的动态平衡。

### 5.3 UnSelected 时间计数

```
Count_select = [UnSelected_explore, UnSelected_exploit]

被选中方：
    UnSelected[对方] += 1
    UnSelected[己方] = 1      # 重置
    F3_己方 = PF[2]            # 重置
    F3_对方 += PF[2]           # 累积（未被选中的累积成本）

未被选中过久 → F3 项累积增大 → 提高下次被选中概率
```

### 5.4 选择决策

每轮比较 `Score_Explore` 与 `Score_Exploit`，选择分数更大者执行。

---

## 6. 两阶段迭代流程

### 阶段一：无经验阶段（Iter 1-3）

每次迭代**同时**运行探索和开发，合并结果取 Top-nSol：

```
Sol = sort(Sol + Sol_Explore + Sol_Exploit)[:nSol]
```

目的：收集探索和开发的初始性能数据（Seq_Cost, Seq_Time），为后续自适应选择提供统计基础。

### 阶段二：有经验阶段（Iter 4-MaxIter）

每次迭代**仅运行**分数更高的一方：

```
if Score_Explore > Score_Exploit:
    Sol = _exploration_phase(Sol)
else:
    Sol = _exploitation_phase(Sol)
```

每轮更新分数和 Mega 权重，实现策略的动态切换。

---

## 7. 早停机制

```
no_improve_count = 0

每轮迭代后：
    if TBest.Cost < Best.Cost:
        Best = TBest
        no_improve_count = 0    # 重置
    else:
        no_improve_count += 1   # 累加

    if no_improve_count >= patience:
        终止迭代
```

默认 `patience=30`，即连续 30 轮最优解未更新则提前终止，避免在已收敛后浪费仿真资源。

---

## 8. 参数汇总

| 参数 | 值 | 说明 |
|------|-----|------|
| `nSol` | 15-100 | 种群规模（≥7） |
| `MaxIter` | 12-300 | 最大迭代次数 |
| `pCR` (初始) | 0.2 | 探索阶段交叉概率 |
| `Q` | 0.67 | 开发阶段攻击模式阈值 |
| `Beta` | 2 | 质心策略缩放因子 |
| `PF` | [0.5, 0.5, 0.3] | F1/F2/F3 先验权重 |
| `Mega` (初始) | 0.99 | 自适应权重 |
| `lmn` | 1-Mega | 自适应互补权重 |
| `patience` | 30 | 早停容忍轮数 |

---

## 9. 与标准 DE 的关键差异

| 特征 | 标准 DE | Puma |
|------|---------|------|
| 变异策略 | 单一（如 rand/1, best/1） | 双策略自适应（探索+开发） |
| 阶段选择 | 固定 | 分数驱动动态切换 |
| 步长控制 | 固定 F | 指数衰减 F1（开发阶段） |
| 交叉率 | 固定 CR | 自适应 pCR（探索阶段） |
| 种群淘汰 | 逐对替换 | 合并排序截断 |
| 收敛判定 | 无 | patience 早停 |

---

## 10. 斯太尔摩工艺优化中的集成

```
候选解 x ∈ R^17 (ORT + SPEED1~10 + FAN1~6)
       │
       ▼
stelmor_batch_cost_function(X_batch)
       │
       ├── 1. pre_check_bounds      # 硬约束预筛（无仿真）
       ├── 2. run_all_simulations   # sim_T 温度+相变仿真
       ├── 3. predict_Ts            # ML 力学性能预测
       └── 4. evaluate_constraints  # 专家约束评分
              │
              ▼
         Score = k_MP*MP_cost + k_EK*(EK_penalty - w_bonus*EK_bonus)
```

代价函数评估昂贵（每批 ~10-30s，含 sim_T + ML），因此：
- 预检查提前过滤不可行解，避免无效仿真
- `BatchCostFunction` 批量评估接口利用多进程并行仿真
- 早停机制减少已收敛后的无谓迭代
