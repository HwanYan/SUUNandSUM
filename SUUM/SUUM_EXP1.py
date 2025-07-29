import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --------------------------
# 1. 参数设置
# --------------------------
beta_b = 0.1    # 目标矿池算力占比
epsilon = 0.02  # 贿赂比例
rho = 0.1       # 攻击者诚实挖矿算力占比
alpha_list = np.linspace(0.01, 0.4, 20)  # 攻击者总算力范围
n_simulations = 100  # 蒙特卡洛模拟次数
converge_threshold = 1e-8  # 收敛阈值：状态概率变化小于此值
converge_window = 10  # 连续10轮稳定则判定收敛
max_steps = 1000000  # 最大步数

# --------------------------
# 2. BSSM状态转换规则
# --------------------------
def get_next_state(current_state, alpha, rho, beta_b):
    """根据当前状态和转换概率返回下一状态（支持无限状态）"""
    # 状态定义：
    # '0'：初始状态；'k=X'：私有链长X（公共链由非攻击者生成）；
    # 'k_=X'：私有链隐藏X（公共链由攻击者诚实池生成）；
    # '0_o'：分叉（攻击者vs其他矿池）；'0_b'：分叉（攻击者vs目标矿池）；'0_a'：分叉（攻击者内部）
    
    if current_state == '0':
        # 状态0的转换（）
        p_selfish = (1 - rho) * alpha  # 转换至状态1
        if np.random.rand() < p_selfish:
            return 'k=1'
        else:
            return '0'
    
    elif current_state.startswith('k='):
        # 状态k（k≥1，私有链长k）的转换（、、）
        k = int(current_state.split('=')[1])
        if k == 1:
            # 状态1的转换
            p_selfish = (1 - rho) * alpha          # 至状态2
            p_o = (1 - alpha - beta_b)             # 至状态0_o
            p_ai = rho * alpha                     # 至状态1_
            p_b = beta_b                           # 至状态0_b
            rand = np.random.rand()
            if rand < p_selfish:
                return 'k=2'
            elif rand < p_selfish + p_o:
                return '0_o'
            elif rand < p_selfish + p_o + p_ai:
                return 'k_=1'
            else:
                return '0_b'
        else:
            # 状态k≥2的转换
            p_selfish = (1 - rho) * alpha  # 至k+1
            p_other = 1 - alpha            # 至状态0
            p_ai = rho * alpha             # 至(k-1)_
            rand = np.random.rand()
            if rand < p_selfish:
                return f'k={k+1}'  # 支持无限增长
            elif rand < p_selfish + p_other:
                return '0'
            else:
                return f'k_={k-1}'
    
    elif current_state.startswith('k_='):
        # 状态k_（k≥1，隐藏私有链k）的转换（、）
        k = int(current_state.split('=')[1])
        p_selfish = (1 - rho) * alpha  # 至(k+1)_
        p_other = 1 - alpha            # 至k-1（k≥2）或0（k=1）
        p_ai = rho * alpha             # 至(k-1)_
        rand = np.random.rand()
        if rand < p_selfish:
            return f'k_={k+1}'  # 支持无限增长
        elif rand < p_selfish + p_other:
            return '0' if k == 1 else f'k={k-1}'
        else:
            return f'k_={k-1}' if k > 1 else '0'
    
    elif current_state in ['0_o', '0_b', '0_a']:
        # 分叉状态的转换（、、）
        # 所有分叉状态最终均转换至状态0
        return '0'
    else:
        return '0'  # 未知状态默认返回初始状态

# --------------------------
# 3. 动态蒙特卡洛模拟（基于收敛性停止）
# --------------------------
def simulate_steady_state(alpha, rho, beta_b, converge_threshold, converge_window, max_steps):
    """模拟获取各状态的稳态概率（当状态概率收敛时停止）"""
    state_counts = defaultdict(int)
    current_state = '0'  # 初始状态
    prev_probs = None
    converge_counter = 0  # 连续收敛计数
    
    for step in range(max_steps):
        current_state = get_next_state(current_state, alpha, rho, beta_b)
        state_counts[current_state] += 1
        
        # 每1000步检查一次收敛性
        if step % 1000 == 0 and step > 0:
            # 计算当前概率分布
            total = sum(state_counts.values())
            current_probs = {s: cnt / total for s, cnt in state_counts.items()}
            # 与上一轮概率比较
            if prev_probs is not None:
                # 计算所有状态的最大概率变化
                all_states = set(prev_probs.keys()).union(current_probs.keys())
                max_diff = max(
                    abs(current_probs.get(s, 0) - prev_probs.get(s, 0)) 
                    for s in all_states
                )
                # 判断是否收敛
                if max_diff < converge_threshold:
                    converge_counter += 1
                    if converge_counter >= converge_window:
                        # 达到收敛条件，停止模拟
                        return current_probs
                else:
                    converge_counter = 0  # 重置计数
            # 更新上一轮概率
            prev_probs = current_probs.copy()
    
    # 若未达到收敛，返回最终概率（超出最大步数）
    total = sum(state_counts.values())
    return {s: cnt / total for s, cnt in state_counts.items()}

# --------------------------
# 4. 计算RER
# --------------------------
def calculate_rer(alpha, rho, beta_b, epsilon, steady_probs):
    """计算各实体的RER"""
    # 提取关键状态概率（默认0）
    p0 = steady_probs.get('0', 0)
    p1 = steady_probs.get('k=1', 0)
    p1_ = steady_probs.get('k_=1', 0)
    p0_o = steady_probs.get('0_o', 0)
    p0_b = steady_probs.get('0_b', 0)
    p0_a = steady_probs.get('0_a', 0)
    # 高阶状态概率（k≥2）
    pk_high = sum(v for k, v in steady_probs.items() if k.startswith('k=') and int(k.split('=')[1]) >=2)
    pk__high = sum(v for k, v in steady_probs.items() if k.startswith('k_=') and int(k.split('=')[1]) >=2)
    
    # 攻击者a的奖励（BSSM vs BSSM'）
    reward_bssm_a = (1 - epsilon) * (
        p0 * rho + p0_b * 2 + p0_o * 2 + p1_ * 1 + pk_high * 1.5 + pk__high * 1.2
    )
    reward_bssm_prime_a = p0 * rho + p0_b * 1 + p0_o * 1 + p1_ * 0.5 + pk_high * 0.8 + pk__high * 0.6
    rer_a_bssm = (reward_bssm_a - reward_bssm_prime_a) / (reward_bssm_prime_a + 1e-10)
    
    # 其他矿池o的奖励（BSSM vs BSSM'）
    reward_bssm_o = p0 * (1 - alpha - beta_b) - (p0_o * beta_b + pk_high * 0.1)
    reward_bssm_prime_o = p0 * (1 - alpha - beta_b)
    rer_o_bssm = (reward_bssm_o - reward_bssm_prime_o) / (reward_bssm_prime_o + 1e-10)
    
    # 目标矿池b的奖励（BSSM vs BSSM'）
    reward_bssm_b = p0 * beta_b + epsilon * reward_bssm_a + p0_b * 0.3
    reward_bssm_prime_b = p0 * beta_b
    rer_b_bssm = (reward_bssm_b - reward_bssm_prime_b) / (reward_bssm_prime_b + 1e-10)
    
    # 攻击者a的奖励（BSSM vs SSM）
    reward_ssm_a = reward_bssm_prime_a  # SSM无贿赂
    rer_a_ssm = (reward_bssm_a - reward_ssm_a) / (reward_ssm_a + 1e-10)
    
    return rer_a_bssm, rer_o_bssm, rer_b_bssm, rer_a_ssm

# --------------------------
# 5. 主程序：模拟与绘图
# --------------------------
# 存储所有alpha对应的RER
rer_a_list = []
rer_o_list = []
rer_b_list = []
rer_a_ssm_list = []

for alpha in alpha_list:
    # 多次模拟取平均
    all_rer_a = []
    all_rer_o = []
    all_rer_b = []
    all_rer_a_ssm = []
    
    for _ in range(n_simulations):
        # 动态模拟稳态概率（基于收敛性停止）
        steady_probs = simulate_steady_state(
            alpha, rho, beta_b, converge_threshold, converge_window, max_steps
        )
        # 计算RER
        rer_a, rer_o, rer_b, rer_a_ssm = calculate_rer(alpha, rho, beta_b, epsilon, steady_probs)
        all_rer_a.append(rer_a)
        all_rer_o.append(rer_o)
        all_rer_b.append(rer_b)
        all_rer_a_ssm.append(rer_a_ssm)
    
    # 取均值
    rer_a_list.append(np.mean(all_rer_a))
    rer_o_list.append(np.mean(all_rer_o))
    rer_b_list.append(np.mean(all_rer_b))
    rer_a_ssm_list.append(np.mean(all_rer_a_ssm))

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(r'RER when $\beta^b=0.1, \varepsilon=0.02, \rho=0.1$ in BSSM', fontsize=14)

# (a) 攻击者a的RER (BSSM vs BSSM')
axes[0, 0].plot(alpha_list, rer_a_list, 'b-', linewidth=2)
axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_title(r'(a) $RER_{a}^{BSSM,BSSM^{\prime}}$', fontsize=12)
axes[0, 0].set_xlabel(r'Adversary Mining Power $\alpha$', fontsize=10)
axes[0, 0].set_ylabel('RER', fontsize=10)

# (b) 其他矿池o的RER (BSSM vs BSSM')
axes[0, 1].plot(alpha_list, rer_o_list, 'r-', linewidth=2)
axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_title(r'(b) $RER_{o}^{BSSM,BSSM^{\prime}}$', fontsize=12)
axes[0, 1].set_xlabel(r'Adversary Mining Power $\alpha$', fontsize=10)
axes[0, 1].set_ylabel('RER', fontsize=10)

# (c) 目标矿池b的RER (BSSM vs BSSM')
axes[1, 0].plot(alpha_list, rer_b_list, 'g-', linewidth=2)
axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_title(r'(c) $RER_{b}^{BSSM,BSSM^{\prime}}$', fontsize=12)
axes[1, 0].set_xlabel(r'Adversary Mining Power $\alpha$', fontsize=10)
axes[1, 0].set_ylabel('RER', fontsize=10)

# (d) 攻击者a的RER (BSSM vs SSM)
axes[1, 1].plot(alpha_list, rer_a_ssm_list, 'm-', linewidth=2)
axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_title(r'(d) $RER_{a}^{BSSM,SSM}$', fontsize=12)
axes[1, 1].set_xlabel(r'Adversary Mining Power $\alpha$', fontsize=10)
axes[1, 1].set_ylabel('RER', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()