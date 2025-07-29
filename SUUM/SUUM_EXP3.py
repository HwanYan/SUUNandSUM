import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# --------------------------
# 1. 参数设置
# --------------------------
beta_b = 0.1    # 目标矿池算力占比
epsilon = 0.02  # 贿赂比例
gamma = 0.5     # 其他矿池加入私有链的概率
rho = 0.1       # 攻击者诚实挖矿算力占比
alpha_list = np.linspace(0.01, 0.4, 20)  # 攻击者总算力范围
n_simulations = 100  # 蒙特卡洛模拟次数
converge_threshold = 1e-8  # 收敛阈值
converge_window = 10  # 连续收敛轮次
max_steps = 1000000  # 最大模拟步数

# --------------------------
# 2. BSSM状态转换规则
# --------------------------
def get_next_state(current_state, alpha, rho, beta_b, gamma):
    """根据当前状态和转换概率返回下一状态（支持无限状态）"""
    # 状态定义：
    # '0'：初始状态；'k=X'：私有链长X（公共链由非攻击者生成）；
    # 'k_=X'：私有链隐藏X（公共链由攻击者诚实池生成）；
    # '0_o'：分叉（攻击者vs其他矿池）；'0_b'：分叉（攻击者vs目标矿池）；'0_a'：分叉（攻击者内部）
    
    if current_state == '0':
        # 状态0的转换
        p_selfish = (1 - rho) * alpha  # 转换至状态1
        if np.random.rand() < p_selfish:
            return 'k=1'
        else:
            return '0'
    
    elif current_state.startswith('k='):
        # 状态k（k≥1，私有链长k）的转换
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
        # 状态k_（k≥1，隐藏私有链k）的转换
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
    
    elif current_state == '0_o':
        # 状态0_o（分叉：攻击者vs其他矿池）的转换，考虑gamma参数
        p_as = (1 - rho) * alpha                  # 私有链胜，至状态0
        p_o_public = (1 - alpha - beta_b) * (1 - gamma)  # 其他矿池公共链胜
        p_o_private = (1 - alpha - beta_b) * gamma       # 其他矿池私有链胜
        p_ai = rho * alpha                         # 诚实池公共链胜
        p_b_public = beta_b * 0.5                  # 目标矿池公共链胜（拒绝贿赂）
        p_b_private = beta_b * 0.5                 # 目标矿池私有链胜（接受贿赂）
        
        rand = np.random.rand()
        cumulative = 0
        
        if rand < cumulative + p_as:
            return '0'
        cumulative += p_as
        
        if rand < cumulative + p_o_public:
            return '0'
        cumulative += p_o_public
        
        if rand < cumulative + p_o_private:
            return '0'
        cumulative += p_o_private
        
        if rand < cumulative + p_ai:
            return '0'
        cumulative += p_ai
        
        if rand < cumulative + p_b_public:
            return '0'
        cumulative += p_b_public
        
        if rand < cumulative + p_b_private:
            return '0'
        return '0'
    
    elif current_state == '0_b':
        # 状态0_b（分叉：攻击者vs目标矿池）的转换
        p_as = (1 - rho) * alpha                  # 私有链胜，至状态0
        p_o_public = (1 - alpha - beta_b) * (1 - gamma)
        p_o_private = (1 - alpha - beta_b) * gamma
        p_ai = rho * alpha                         # 诚实池公共链胜
        p_b_public = beta_b                        # 目标矿池公共链胜（拒绝贿赂）
        
        rand = np.random.rand()
        cumulative = 0
        
        if rand < cumulative + p_as:
            return '0'
        cumulative += p_as
        
        if rand < cumulative + p_o_public:
            return '0'
        cumulative += p_o_public
        
        if rand < cumulative + p_o_private:
            return '0'
        cumulative += p_o_private
        
        if rand < cumulative + p_ai:
            return '0'
        cumulative += p_ai
        
        if rand < cumulative + p_b_public:
            return '0'
        return '0'
    
    elif current_state == '0_a':
        # 状态0_a（分叉：攻击者内部）的转换
        p_as = (1 - rho) * alpha                  # 私有链胜，至状态0
        p_o = (1 - alpha - beta_b)                # 其他矿池公共链胜
        p_ai = rho * alpha                         # 诚实池公共链胜
        p_b_public = beta_b * 0.5                  # 目标矿池公共链胜
        p_b_private = beta_b * 0.5                 # 目标矿池私有链胜
        
        rand = np.random.rand()
        cumulative = 0
        
        if rand < cumulative + p_as:
            return '0'
        cumulative += p_as
        
        if rand < cumulative + p_o:
            return '0'
        cumulative += p_o
        
        if rand < cumulative + p_ai:
            return '0'
        cumulative += p_ai
        
        if rand < cumulative + p_b_public:
            return '0'
        cumulative += p_b_public
        
        if rand < cumulative + p_b_private:
            return '0'
        return '0'
    else:
        return '0'  # 未知状态默认返回初始状态

# --------------------------
# 3. 动态蒙特卡洛模拟（基于收敛性停止）
# --------------------------
def simulate_steady_state(alpha, rho, beta_b, gamma, converge_threshold, converge_window, max_steps):
    """模拟获取各状态的稳态概率（当状态概率收敛时停止）"""
    state_counts = defaultdict(int)
    current_state = '0'  # 初始状态
    prev_probs = None
    converge_counter = 0  # 连续收敛计数
    
    for step in range(max_steps):
        current_state = get_next_state(current_state, alpha, rho, beta_b, gamma)
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
# 4. 计算RER（BSSM与BSSM'的对比）
# --------------------------
def calculate_rer(alpha, rho, beta_b, epsilon, gamma, steady_probs):
    """计算攻击者(a)、其他矿池(o)和目标矿池(b)的RER"""
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
    
    # 攻击者a的奖励计算
    # BSSM策略（有贿赂）
    reward_bssm_a = (1 - epsilon) * (
        p0 * rho + p0_b * 2 + p0_o * 2 + p1_ * 1 + pk_high * 1.5 + pk__high * 1.2
    )
    # BSSM'策略（无贿赂）
    reward_bssm_prime_a = p0 * rho + p0_b * 1 + p0_o * 1 + p1_ * 0.5 + pk_high * 0.8 + pk__high * 0.6
    rer_a = (reward_bssm_a - reward_bssm_prime_a) / (reward_bssm_prime_a + 1e-10)
    
    # 其他矿池o的奖励计算
    reward_bssm_o = p0 * (1 - alpha - beta_b) - (p0_o * beta_b * 0.3 + pk_high * 0.1)
    reward_bssm_prime_o = p0 * (1 - alpha - beta_b)
    rer_o = (reward_bssm_o - reward_bssm_prime_o) / (reward_bssm_prime_o + 1e-10)
    
    # 目标矿池b的奖励计算
    reward_bssm_b = p0 * beta_b + epsilon * reward_bssm_a + p0_b * 0.3
    reward_bssm_prime_b = p0 * beta_b
    rer_b = (reward_bssm_b - reward_bssm_prime_b) / (reward_bssm_prime_b + 1e-10)
    
    return rer_a, rer_o, rer_b

# --------------------------
# 5. 主程序：模拟与绘图
# --------------------------
def main():
    print(f"开始模拟(βb = {beta_b}, ε = {epsilon}, γ = {gamma})...")
    start_time = time.time()
    
    # 初始化存储列表
    rer_a_list = []
    rer_o_list = []
    rer_b_list = []
    
    for i, alpha in enumerate(alpha_list):
        print(f"  进度: {i+1}/{len(alpha_list)} (alpha = {alpha:.3f})")
        
        # 多次模拟取平均
        all_rer_a = []
        all_rer_o = []
        all_rer_b = []
        
        for _ in range(n_simulations):
            # 动态模拟稳态概率
            steady_probs = simulate_steady_state(
                alpha, rho, beta_b, gamma, converge_threshold, converge_window, max_steps
            )
            # 计算RER
            rer_a, rer_o, rer_b = calculate_rer(alpha, rho, beta_b, epsilon, gamma, steady_probs)
            all_rer_a.append(rer_a)
            all_rer_o.append(rer_o)
            all_rer_b.append(rer_b)
        
        # 计算平均值并存储
        rer_a_list.append(np.mean(all_rer_a))
        rer_o_list.append(np.mean(all_rer_o))
        rer_b_list.append(np.mean(all_rer_b))
    
    end_time = time.time()
    print(f"模拟完成，耗时 {end_time - start_time:.2f} 秒")
    
    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle('RER_a,o,b^(BSSM,BSSM\') when β^b=0.1, ε=0.02, γ=0.5 in BSSM', fontsize=16)

    # (a) 攻击者a的RER
    axes[0].plot(alpha_list, rer_a_list, 'b-', linewidth=2)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_title('(a) RER_a^(BSSM,BSSM\')', fontsize=14)
    axes[0].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[0].set_ylabel('RER', fontsize=12)
    axes[0].grid(alpha=0.3)

    # (b) 其他矿池o的RER
    axes[1].plot(alpha_list, rer_o_list, 'r-', linewidth=2)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_title('(b) RER_o^(BSSM,BSSM\')', fontsize=14)
    axes[1].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[1].set_ylabel('RER', fontsize=12)
    axes[1].grid(alpha=0.3)

    # (c) 目标矿池b的RER
    axes[2].plot(alpha_list, rer_b_list, 'g-', linewidth=2)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_title('(c) RER_b^(BSSM,BSSM\')', fontsize=14)
    axes[2].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[2].set_ylabel('RER', fontsize=12)
    axes[2].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为suptitle预留空间
    plt.show()

if __name__ == "__main__":
    main()
    