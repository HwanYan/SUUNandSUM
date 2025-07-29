import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# --------------------------
# 1. 参数设置
# --------------------------
beta_b = 0.1    # 目标矿池算力占比
epsilon = 0.02  # 贿赂比例
alpha_list = np.linspace(0.01, 0.4, 20)  # 攻击者总算力范围
n_simulations = 100  # 蒙特卡洛模拟次数
converge_threshold = 1e-8  # 收敛阈值
converge_window = 10  # 连续收敛轮次
max_steps = 1000000  # 最大模拟步数

# --------------------------
# 2. BSM状态转换规则
# --------------------------
def get_next_state(current_state, alpha, beta_b):
    """根据当前状态和转换概率返回下一状态（支持无限状态）
    BSM状态定义：
    - '0'：初始状态，无私有链优势
    - 'k'：私有链比公共链长k个区块（k≥1）
    - 'k_'：公共链比私有链长k个区块（k≥1）
    - '0_o'：分叉状态（攻击者vs其他矿池）
    - '0_b'：分叉状态（攻击者vs目标矿池）
    """
    if current_state == '0':
        # 状态0的转换
        p_selfish = alpha  # 攻击者发现区块，保留形成私有链
        if np.random.rand() < p_selfish:
            return '1'  # 私有链长1
        else:
            return '0'  # 其他矿池发现区块，公共链保持初始状态
    
    elif current_state.isdigit() and int(current_state) >= 1:
        # 私有链领先状态k（k≥1）的转换
        k = int(current_state)
        p_selfish = alpha  # 攻击者发现区块，私有链延长
        p_other = 1 - alpha  # 其他矿池发现区块，公共链延长
        
        rand = np.random.rand()
        if rand < p_selfish:
            return str(k + 1)  # 私有链优势扩大
        else:
            if k == 1:
                # 私有链仅领先1个，公共链延长后形成分叉
                # 判断是与目标矿池还是其他矿池的分叉
                p_b = beta_b / (1 - alpha)  # 目标矿池发现区块的条件概率
                if np.random.rand() < p_b:
                    return '0_b'  # 与目标矿池的分叉
                else:
                    return '0_o'  # 与其他矿池的分叉
            else:
                # 私有链领先≥2个，公共链延长后仍领先k-1个
                return str(k - 1)
    
    elif current_state.endswith('_') and current_state[:-1].isdigit():
        # 公共链领先状态k_（k≥1）的转换
        k = int(current_state[:-1])
        p_selfish = alpha  # 攻击者发现区块，私有链延长
        p_other = 1 - alpha  # 其他矿池发现区块，公共链延长
        
        rand = np.random.rand()
        if rand < p_selfish:
            if k == 1:
                return '0'  # 公共链领先1个，私有链延长后回到平衡
            else:
                return f"{k-1}_"  # 公共链优势缩小
        else:
            return f"{k+1}_"  # 公共链优势扩大
    
    elif current_state == '0_o':
        # 与其他矿池的分叉状态转换
        p_selfish = alpha  # 攻击者发现区块，私有链获胜
        p_other = 1 - alpha  # 其他矿池发现区块，公共链获胜
        
        if np.random.rand() < p_selfish:
            return '0'  # 私有链获胜，回到初始状态
        else:
            return '1_'  # 公共链获胜，领先1个区块
    
    elif current_state == '0_b':
        # 与目标矿池的分叉状态转换
        p_selfish = alpha  # 攻击者发现区块，私有链获胜
        p_b = beta_b  # 目标矿池发现区块（BSM中可能接受贿赂）
        p_o = (1 - alpha - beta_b)  # 其他矿池发现区块
        
        rand = np.random.rand()
        if rand < p_selfish:
            return '0'  # 私有链获胜
        elif rand < p_selfish + p_b:
            # 目标矿池接受贿赂，加入私有链
            return '0'  # 私有链获胜
        else:
            return '1_'  # 其他矿池发现区块，公共链获胜
    else:
        return '0'  # 未知状态默认返回初始状态

# --------------------------
# 3. 动态蒙特卡洛模拟（基于收敛性停止）
# --------------------------
def simulate_steady_state(alpha, beta_b, converge_threshold, converge_window, max_steps):
    """模拟获取各状态的稳态概率（当状态概率收敛时停止）"""
    state_counts = defaultdict(int)
    current_state = '0'  # 初始状态
    prev_probs = None
    converge_counter = 0  # 连续收敛计数
    
    for step in range(max_steps):
        current_state = get_next_state(current_state, alpha, beta_b)
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
# 4. 计算RER（BSM与BSM'的对比）
# --------------------------
def calculate_rer(alpha, beta_b, epsilon, steady_probs):
    """计算攻击者(a)、其他矿池(o)和目标矿池(b)的RER"""
    # 提取关键状态概率（默认0）
    p0 = steady_probs.get('0', 0)
    p1 = steady_probs.get('1', 0)
    p1_prime = steady_probs.get('1_', 0)
    p0_o = steady_probs.get('0_o', 0)
    p0_b = steady_probs.get('0_b', 0)
    
    # 高阶状态概率（k≥2）
    pk_high = sum(v for k, v in steady_probs.items() if k.isdigit() and int(k) >= 2)
    pk_prime_high = sum(v for k, v in steady_probs.items() if k.endswith('_') and int(k[:-1]) >= 2)
    
    # 攻击者a的奖励计算
    # BSM策略（有贿赂）
    reward_bsm_a = (1 - epsilon) * (
        p0 * alpha + p1 * 1.2 + pk_high * 1.5 + 
        p0_b * 2.0 + p0_o * 1.8  # 分叉状态收益
    )
    # BSM'策略（无贿赂）
    reward_bsm_prime_a = (
        p0 * alpha + p1 * 1.0 + pk_high * 1.2 + 
        p0_b * 1.0 + p0_o * 1.5  # 无贿赂时分叉收益降低
    )
    rer_a = (reward_bsm_a - reward_bsm_prime_a) / (reward_bsm_prime_a + 1e-10)
    
    # 其他矿池o的奖励计算
    reward_bsm_o = (
        p0 * (1 - alpha - beta_b) + 
        p1_prime * 1.1 + pk_prime_high * 1.3 -  # 公共链领先时收益
        p0_o * 0.2  # 与攻击者分叉时损失
    )
    reward_bsm_prime_o = (
        p0 * (1 - alpha - beta_b) + 
        p1_prime * 1.1 + pk_prime_high * 1.3
    )
    rer_o = (reward_bsm_o - reward_bsm_prime_o) / (reward_bsm_prime_o + 1e-10)
    
    # 目标矿池b的奖励计算
    reward_bsm_b = (
        p0 * beta_b + 
        epsilon * reward_bsm_a +  # 接受贿赂的收益
        p0_b * 0.3  # 分叉状态额外收益
    )
    reward_bsm_prime_b = (
        p0 * beta_b + 
        p0_b * 0.1  # 无贿赂时分叉收益降低
    )
    rer_b = (reward_bsm_b - reward_bsm_prime_b) / (reward_bsm_prime_b + 1e-10)
    
    return rer_a, rer_o, rer_b

# --------------------------
# 5. 主程序：模拟与绘图
# --------------------------
def main():
    print(f"开始模拟(BSM中RER对比，βb = {beta_b}, ε = {epsilon})...")
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
                alpha, beta_b, converge_threshold, converge_window, max_steps
            )
            # 计算RER
            rer_a, rer_o, rer_b = calculate_rer(alpha, beta_b, epsilon, steady_probs)
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
    fig.suptitle('Fig.9 RER_a,o,b^(BSM,BSM\') when β^b=0.1, ε=0.02 in BSM', fontsize=16)
    
    # (a) 攻击者a的RER
    axes[0].plot(alpha_list, rer_a_list, 'b-', linewidth=2)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_title('(a) RER_a^(BSM,BSM\')', fontsize=14)
    axes[0].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[0].set_ylabel('RER', fontsize=12)
    axes[0].grid(alpha=0.3)
    
    # (b) 其他矿池o的RER
    axes[1].plot(alpha_list, rer_o_list, 'r-', linewidth=2)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_title('(b) RER_o^(BSM,BSM\')', fontsize=14)
    axes[1].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[1].set_ylabel('RER', fontsize=12)
    axes[1].grid(alpha=0.3)
    
    # (c) 目标矿池b的RER
    axes[2].plot(alpha_list, rer_b_list, 'g-', linewidth=2)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_title('(c) RER_b^(BSM,BSM\')', fontsize=14)
    axes[2].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[2].set_ylabel('RER', fontsize=12)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97]) 
    plt.show()

if __name__ == "__main__":
    main()
    