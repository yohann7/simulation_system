import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# -------------------------- initialization.m --------------------------
def initialization(N, Dim, UB, LB):
    """种群初始化函数"""
    X = np.zeros((N, Dim))
    
    # 处理单一边界情况
    if np.isscalar(UB) or len(np.array(UB).shape) == 0:
        UB = float(UB)
        LB = float(LB)
        X = np.random.rand(N, Dim) * (UB - LB) + LB
    # 处理多维不同边界情况
    else:
        UB = np.array(UB)
        LB = np.array(LB)
        for i in range(Dim):
            X[:, i] = np.random.rand(N) * (UB[i] - LB[i]) + LB[i]
    return X


# -------------------------- Get_F.m (含Ufun和F1-F23) --------------------------
def Ufun(x, a, k, m):
    """辅助函数Ufun"""
    return k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < -a)


# 测试函数定义
def F1(x): return np.sum(x**2)
def F2(x): return np.sum(np.abs(x)) + np.prod(np.abs(x))
def F3(x):
    dim = len(x)
    return sum(np.sum(x[:i+1])**2 for i in range(dim))
def F4(x): return np.max(np.abs(x))
def F5(x):
    dim = len(x)
    return sum(100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2 for i in range(dim-1))
def F6(x): return np.sum((np.abs(x + 0.5))**2)
def F7(x):
    dim = len(x)
    return sum(np.arange(1, dim+1) * (x**4)) + np.random.rand()
def F8(x): return np.sum(-x * np.sin(np.sqrt(np.abs(x))))
def F9(x):
    dim = len(x)
    return np.sum(x**2 - 10*np.cos(2*np.pi*x)) + 10*dim
def F10(x):
    dim = len(x)
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/dim)) - \
           np.exp(np.sum(np.cos(2*np.pi*x))/dim) + 20 + np.exp(1)
def F11(x):
    dim = len(x)
    return np.sum(x**2)/4000 - np.prod(np.cos(x/np.sqrt(np.arange(1, dim+1)))) + 1
def F12(x):
    dim = len(x)
    term1 = (np.pi/dim) * (10 * (np.sin(np.pi*(1+(x[0]+1)/4)))**2)
    term2 = sum((((x[i]+1)/4)**2) * (1 + 10*(np.sin(np.pi*(1+(x[i+1]+1)/4)))**2) for i in range(dim-1))
    term3 = ((x[-1]+1)/4)**2
    term4 = np.sum(Ufun(x, 10, 100, 4))
    return term1 + term2 + term3 + term4
def F13(x):
    dim = len(x)
    term1 = 0.1 * ((np.sin(3*np.pi*x[0]))**2)
    term2 = sum((x[i]-1)**2 * (1 + (np.sin(3*np.pi*x[i+1]))**2) for i in range(dim-1))
    term3 = ((x[-1]-1)**2) * (1 + (np.sin(2*np.pi*x[-1]))**2)
    term4 = np.sum(Ufun(x, 5, 100, 4))
    return 0.1*(term1 + term2 + term3) + term4
def F14(x):
    aS = np.array([[-32,-16,0,16,32]*5, [-32]*5 + [-16]*5 + [0]*5 + [16]*5 + [32]*5])
    bS = [np.sum((x - aS[:,j])**6) for j in range(25)]
    return (1/500 + sum(1/(np.arange(1,26)+bS)))**(-1)
def F15(x):
    aK = np.array([0.1957,0.1947,0.1735,0.16,0.0844,0.0627,0.0456,0.0342,0.0323,0.0235,0.0246])
    bK = 1/np.array([0.25,0.5,1,2,4,6,8,10,12,14,16])
    term = (x[0]*(bK**2 + x[1]*bK)) / (bK**2 + x[2]*bK + x[3])
    return np.sum((aK - term)**2)
def F16(x): return 4*x[0]**2 - 2.1*x[0]**4 + x[0]**6/3 + x[0]*x[1] - 4*x[1]**2 + 4*x[1]**4
def F17(x):
    term1 = (x[1] - 5.1*x[0]**2/(4*np.pi**2) + 5*x[0]/np.pi -6)**2
    term2 = 10*(1 - 1/(8*np.pi))*np.cos(x[0])
    return term1 + term2 + 10
def F18(x):
    term1 = 1 + (x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2)
    term2 = 30 + (2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2)
    return term1 * term2
def F19(x):
    aH = np.array([[3,10,30],[0.1,10,35],[3,10,30],[0.1,10,35]])
    cH = np.array([1,1.2,3,3.2])
    pH = np.array([[0.3689,0.117,0.2673],[0.4699,0.4387,0.747],[0.1091,0.8732,0.5547],[0.03815,0.5743,0.8828]])
    return -sum(cH[i] * np.exp(-np.sum(aH[i]*(x-pH[i])**2)) for i in range(4))
def F20(x):
    aH = np.array([[10,3,17,3.5,1.7,8],[0.05,10,17,0.1,8,14],[3,3.5,1.7,10,17,8],[17,8,0.05,10,0.1,14]])
    cH = np.array([1,1.2,3,3.2])
    pH = np.array([[0.1312,0.1696,0.5569,0.0124,0.8283,0.5886],[0.2329,0.4135,0.8307,0.3736,0.1004,0.9991],
                   [0.2348,0.1415,0.3522,0.2883,0.3047,0.6650],[0.4047,0.8828,0.8732,0.5743,0.1091,0.0381]])
    return -sum(cH[i] * np.exp(-np.sum(aH[i]*(x-pH[i])**2)) for i in range(4))
def F21(x):
    aSH = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    return -sum(1/(np.dot(x-aSH[i], x-aSH[i]) + cSH[i]) for i in range(5))
def F22(x):
    aSH = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    return -sum(1/(np.dot(x-aSH[i], x-aSH[i]) + cSH[i]) for i in range(7))
def F23(x):
    aSH = np.array([[4,4,4,4],[1,1,1,1],[8,8,8,8],[6,6,6,6],[3,7,3,7],[2,9,2,9],[5,5,3,3],[8,1,8,1],[6,2,6,2],[7,3.6,7,3.6]])
    cSH = np.array([0.1,0.2,0.2,0.4,0.4,0.6,0.3,0.7,0.5,0.5])
    return -sum(1/(np.dot(x-aSH[i], x-aSH[i]) + cSH[i]) for i in range(10))


def Get_F(F_name):
    """获取测试函数的参数和目标函数"""
    func_map = {
        'F1': (F1, -100, 100, 10),
        'F2': (F2, -10, 10, 10),
        'F3': (F3, -100, 100, 10),
        'F4': (F4, -100, 100, 10),
        'F5': (F5, -30, 30, 10),
        'F6': (F6, -100, 100, 10),
        'F7': (F7, -1.28, 1.28, 10),
        'F8': (F8, -500, 500, 10),
        'F9': (F9, -5.12, 5.12, 10),
        'F10': (F10, -32, 32, 10),
        'F11': (F11, -600, 600, 10),
        'F12': (F12, -50, 50, 50),
        'F13': (F13, -50, 50, 10),
        'F14': (F14, -65.536, 65.536, 2),
        'F15': (F15, -5, 5, 4),
        'F16': (F16, -5, 5, 2),
        'F17': (F17, [-5, 0], [10, 15], 2),
        'F18': (F18, -2, 2, 2),
        'F19': (F19, 0, 1, 3),
        'F20': (F20, 0, 1, 6),
        'F21': (F21, 0, 10, 4),
        'F22': (F22, 0, 10, 4),
        'F23': (F23, 0, 10, 4)
    }
    if F_name not in func_map:
        raise ValueError(f"Unsupported function: {F_name}")
    F_obj, LB, UB, Dim = func_map[F_name]
    return np.array(LB) if not np.isscalar(LB) else LB, \
           np.array(UB) if not np.isscalar(UB) else UB, Dim, F_obj


# -------------------------- AOA.m --------------------------
def AOA(N, M_Iter, LB, UB, Dim, F_obj):
    """算术优化算法核心"""
    print("AOA Working")
    eps = np.finfo(float).eps
    
    # 初始化
    Best_P = np.zeros(Dim)
    Best_FF = np.inf
    Conv_curve = np.zeros(M_Iter)
    X = initialization(N, Dim, UB, LB)
    Xnew = X.copy()
    Ffun = np.zeros(N)
    Ffun_new = np.zeros(N)
    
    # 算法参数
    MOP_Max, MOP_Min = 1, 0.2
    Alpha, Mu = 5, 0.499
    
    # 初始适应度计算
    for i in range(N):
        Ffun[i] = F_obj(X[i])
        if Ffun[i] < Best_FF:
            Best_FF, Best_P = Ffun[i], X[i].copy()
    
    # 主迭代循环
    for C_Iter in range(M_Iter):
        current_iter = C_Iter + 1
        # 计算MOP和MOA
        MOP = 1 - (current_iter ** (1/Alpha) / M_Iter ** (1/Alpha))
        MOA = MOP_Min + current_iter * ((MOP_Max - MOP_Min) / M_Iter)
        
        # 更新每个个体
        for i in range(N):
            for j in range(Dim):
                r1 = np.random.rand()
                # 处理边界类型
                if np.isscalar(LB) or len(LB) == 1:
                    ub, lb = float(UB), float(LB)
                    if r1 < MOA:
                        r2 = np.random.rand()
                        if r2 > 0.5:
                            Xnew[i,j] = Best_P[j]/(MOP+eps) * ((ub-lb)*Mu + lb)
                        else:
                            Xnew[i,j] = Best_P[j]*MOP * ((ub-lb)*Mu + lb)
                    else:
                        r3 = np.random.rand()
                        if r3 > 0.5:
                            Xnew[i,j] = Best_P[j] - MOP*((ub-lb)*Mu + lb)
                        else:
                            Xnew[i,j] = Best_P[j] + MOP*((ub-lb)*Mu + lb)
                else:
                    ub_j, lb_j = UB[j], LB[j]
                    if r1 < MOA:
                        r2 = np.random.rand()
                        if r2 > 0.5:
                            Xnew[i,j] = Best_P[j]/(MOP+eps) * ((ub_j-lb_j)*Mu + lb_j)
                        else:
                            Xnew[i,j] = Best_P[j]*MOP * ((ub_j-lb_j)*Mu + lb_j)
                    else:
                        r3 = np.random.rand()
                        if r3 > 0.5:
                            Xnew[i,j] = Best_P[j] - MOP*((ub_j-lb_j)*Mu + lb_j)
                        else:
                            Xnew[i,j] = Best_P[j] + MOP*((ub_j-lb_j)*Mu + lb_j)
            
            # 边界约束
            Flag_UB = Xnew[i] > UB
            Flag_LB = Xnew[i] < LB
            Xnew[i] = Xnew[i] * (~(Flag_UB + Flag_LB)) + UB*Flag_UB + LB*Flag_LB
            
            # 更新适应度和最优解
            Ffun_new[i] = F_obj(Xnew[i])
            if Ffun_new[i] < Ffun[i]:
                X[i], Ffun[i] = Xnew[i].copy(), Ffun_new[i]
            if Ffun[i] < Best_FF:
                Best_FF, Best_P = Ffun[i], X[i].copy()
        
        # 记录收敛曲线
        Conv_curve[C_Iter] = Best_FF
        # 每50次迭代输出
        if current_iter % 50 == 0:
            print(f"At iteration {current_iter}, the best fitness is {Best_FF:.6f}")
    
    return Best_FF, Best_P, Conv_curve


# -------------------------- func_plot.m --------------------------
def func_plot(ax, func_name):
    """绘制测试函数3D曲面"""
    LB, UB, Dim, F_obj = Get_F(func_name)
    # 设置绘图范围
    range_map = {
        'F1': (-100,100), 'F2': (-10,10), 'F3': (-100,100), 'F4': (-100,100),
        'F5': (-200,200), 'F6': (-100,100), 'F7': (-1,1), 'F8': (-500,500),
        'F9': (-5,5), 'F10': (-20,20), 'F11': (-500,500), 'F12': (-10,10),
        'F13': (-5,5), 'F14': (-100,100)
    }
    x_min, x_max = range_map.get(func_name, (-5,5))
    x = np.linspace(x_min, x_max, 50)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # 计算函数值
    for i in range(len(x)):
        for j in range(len(y)):
            if func_name == 'F15':
                Z[i,j] = F_obj(np.array([X[i,j], Y[i,j], 0, 0]))
            elif func_name == 'F19':
                Z[i,j] = F_obj(np.array([X[i,j], Y[i,j], 0]))
            elif func_name == 'F20':
                Z[i,j] = F_obj(np.array([X[i,j], Y[i,j], 0,0,0,0]))
            elif func_name in ['F21','F22','F23']:
                Z[i,j] = F_obj(np.array([X[i,j], Y[i,j], 0,0]))
            else:
                Z[i,j] = F_obj(np.array([X[i,j], Y[i,j]]))
    
    # 绘制曲面
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x_1')
    ax.set_ylabel('x_2')
    ax.set_zlabel(f'{func_name}(x_1, x_2)')
    ax.set_title('Parameter space')


# -------------------------- main.m --------------------------
def main():
    # 参数设置
    Solution_no = 20    # 种群规模
    F_name = 'F1'       # 测试函数
    M_Iter = 1000       # 最大迭代次数
    
    # 获取测试函数
    LB, UB, Dim, F_obj = Get_F(F_name)
    
    # 运行AOA
    Best_FF, Best_P, Conv_curve = AOA(Solution_no, M_Iter, LB, UB, Dim, F_obj)
    
    # 绘图
    fig = plt.figure(figsize=(12, 5))
    # 左图：测试函数曲面
    ax1 = fig.add_subplot(121, projection='3d')
    func_plot(ax1, F_name)
    # 右图：收敛曲线
    ax2 = fig.add_subplot(122)
    ax2.semilogy(Conv_curve, 'r-', linewidth=2)
    ax2.set_title('Convergence curve')
    ax2.set_xlabel('Iteration#')
    ax2.set_ylabel('Best fitness function')
    ax2.legend(['AOA'])
    ax2.axis('tight')
    
    plt.tight_layout()
    plt.show()
    
    # 输出结果
    print(f"\nThe best-obtained solution: {Best_P}")
    print(f"The best optimal value: {Best_FF:.6f}")


if __name__ == "__main__":
    main()