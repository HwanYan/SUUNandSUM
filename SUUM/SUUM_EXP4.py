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
alpha_list = np.linspace(0.01, 0.5, 25)  # 攻击者总算力范围
n_simulations = 100  # 蒙特卡洛模拟次数
converge_threshold = 1e-8  # 收敛阈值
converge_window = 10  # 连续收敛轮次
max_steps = 1000000  # 最大模拟步数

# --------------------------
# 2. BSSM状态转换规则
# --------------------------
def simulate_and_track_wins(alpha, rho, beta_b, gamma, max_steps):
    """模拟状态转换并跟踪私有链获胜情况"""
    state_counts = defaultdict(int)
    current_state = '0'  # 初始状态
    private_chain_wins = 0  # 私有链获胜次数
    total_battles = 0       # 总分叉竞争次数
    
    for step in range(max_steps):
        next_state = get_next_state(
            current_state, alpha, rho, beta_b, gamma,
            private_chain_wins, total_battles
        )
        
        # 检查是否发生了分叉竞争并记录结果
        if current_state in ['0_o', '0_b', '0_a']:
            total_battles += 1
            # 判断私有链是否获胜（基于转换概率）
            if check_private_win(current_state, alpha, rho, beta_b, gamma):
                private_chain_wins += 1
        
        current_state = next_state
        state_counts[current_state] += 1
    
    # 计算稳态概率
    total = sum(state_counts.values())
    steady_probs = {s: cnt / total for s, cnt in state_counts.items()}
    
    # 计算私有链获胜概率
    win_prob = private_chain_wins / total_battles if total_battles > 0 else 0
    
    return steady_probs, win_prob

def get_next_state(current_state, alpha, rho, beta_b, gamma, private_wins, total_battles):
    """根据当前状态和转换概率返回下一状态"""
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
            p_selfish = (1 - rho) * alpha  # 至k+1
            p_other = 1 - alpha            # 至状态0
            p_ai = rho * alpha             # 至(k-1)_
            rand = np.random.rand()
            if rand < p_selfish:
                return f'k={k+1}'
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
            return f'k_={k+1}'
        elif rand < p_selfish + p_other:
            return '0' if k == 1 else f'k={k-1}'
        else:
            return f'k_={k-1}' if k > 1 else '0'
    
    elif current_state in ['0_o', '0_b', '0_a']:
        # 分叉状态结束后返回初始状态
        return '0'
    else:
        return '0'

def check_private_win(state, alpha, rho, beta_b, gamma):
    """判断在分叉状态中私有链是否获胜"""
    rand = np.random.rand()
    
    if state == '0_o':
        # 分叉：攻击者vs其他矿池
        p_as = (1 - rho) * alpha                  # 私有链胜
        p_o_private = (1 - alpha - beta_b) * gamma # 其他矿池加入私有链
        p_b_private = beta_b * 0.5                # 目标矿池接受贿赂
        
        total_private_prob = p_as + p_o_private + p_b_private
        return rand < total_private_prob
    
    elif state == '0_b':
        # 分叉：攻击者vs目标矿池
        p_as = (1 - rho) * alpha                  # 私有链胜
        p_o_private = (1 - alpha - beta_b) * gamma # 其他矿池加入私有链
        
        total_private_prob = p_as + p_o_private
        return rand < total_private_prob
    
    elif state == '0_a':
        # 分叉：攻击者内部
        p_as = (1 - rho) * alpha                  # 私有链胜
        p_b_private = beta_b * 0.5                # 目标矿池接受贿赂
        
        total_private_prob = p_as + p_b_private
        return rand < total_private_prob
    
    return False

# --------------------------
# 3. 计算RER和获胜条件
# --------------------------
def calculate_metrics(alpha, rho, beta_b, epsilon, gamma, steady_probs, win_prob):
    """计算RER和其他性能指标"""
    # 提取关键状态概率
    p0 = steady_probs.get('0', 0)
    p1 = steady_probs.get('k=1', 0)
    p1_ = steady_probs.get('k_=1', 0)
    p0_o = steady_probs.get('0_o', 0)
    p0_b = steady_probs.get('0_b', 0)
    
    # 高阶状态概率
    pk_high = sum(v for k, v in steady_probs.items() if k.startswith('k=') and int(k.split('=')[1]) >=2)
    pk__high = sum(v for k, v in steady_probs.items() if k.startswith('k_=') and int(k.split('=')[1]) >=2)
    
    # 攻击者a的RER (BSSM vs BSSM')
    reward_bssm_a = (1 - epsilon) * (
        p0 * rho + p0_b * 2 + p0_o * 2 + p1_ * 1 + pk_high * 1.5 + pk__high * 1.2
    )
    reward_bssm_prime_a = p0 * rho + p0_b * 1 + p0_o * 1 + p1_ * 0.5 + pk_high * 0.8 + pk__high * 0.6
    rer_a = (reward_bssm_a - reward_bssm_prime_a) / (reward_bssm_prime_a + 1e-10)
    
    # 目标矿池b的RER (BSSM vs BSSM')
    reward_bssm_b = p0 * beta_b + epsilon * reward_bssm_a + p0_b * 0.3
    reward_bssm_prime_b = p0 * beta_b
    rer_b = (reward_bssm_b - reward_bssm_prime_b) / (reward_bssm_prime_b + 1e-10)
    
    # 其他矿池o的RER (BSSM vs BSSM')
    reward_bssm_o = p0 * (1 - alpha - beta_b) - (p0_o * beta_b * 0.3 + pk_high * 0.1)
    reward_bssm_prime_o = p0 * (1 - alpha - beta_b)
    rer_o = (reward_bssm_o - reward_bssm_prime_o) / (reward_bssm_prime_o + 1e-10)
    
    # 计算分叉率（分叉状态占比）
    fork_rate = p0_o + p0_b + steady_probs.get('0_a', 0)
    
    return rer_a, rer_b, rer_o, win_prob, fork_rate

# --------------------------
# 4. 主程序：模拟与绘图
# --------------------------
def main():
    print(f"开始模拟  (BSSM中的RER和获胜条件)...")
    start_time = time.time()
    
    # 初始化存储列表
    rer_a_list = []
    rer_b_list = []
    rer_o_list = []
    win_prob_list = []
    fork_rate_list = []
    
    for i, alpha in enumerate(alpha_list):
        print(f"  进度: {i+1}/{len(alpha_list)} (alpha = {alpha:.3f})")
        
        # 多次模拟取平均
        metrics = []
        for _ in range(n_simulations):
            steady_probs, win_prob = simulate_and_track_wins(
                alpha, rho, beta_b, gamma, max_steps
            )
            metrics.append(calculate_metrics(
                alpha, rho, beta_b, epsilon, gamma, steady_probs, win_prob
            ))
        
        # 计算平均值并存储
        mean_metrics = np.mean(metrics, axis=0)
        rer_a_list.append(mean_metrics[0])
        rer_b_list.append(mean_metrics[1])
        rer_o_list.append(mean_metrics[2])
        win_prob_list.append(mean_metrics[3])
        fork_rate_list.append(mean_metrics[4])
    
    end_time = time.time()
    print(f"模拟完成，耗时 {end_time - start_time:.2f} 秒")
    
    # 绘图
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle('RER and winning conditions in BSSM', fontsize=16)
    
    # (a) 攻击者a的RER
    axes[0, 0].plot(alpha_list, rer_a_list, 'b-', linewidth=2)
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_title('(a) RER_a^(BSSM,BSSM\')', fontsize=14)
    axes[0, 0].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[0, 0].set_ylabel('RER', fontsize=12)
    axes[0, 0].grid(alpha=0.3)
    
    # (b) 目标矿池b的RER
    axes[0, 1].plot(alpha_list, rer_b_list, 'g-', linewidth=2)
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_title('(b) RER_b^(BSSM,BSSM\')', fontsize=14)
    axes[0, 1].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[0, 1].set_ylabel('RER', fontsize=12)
    axes[0, 1].grid(alpha=0.3)
    
    # (c) 其他矿池o的RER
    axes[1, 0].plot(alpha_list, rer_o_list, 'r-', linewidth=2)
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('(c) RER_o^(BSSM,BSSM\')', fontsize=14)
    axes[1, 0].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[1, 0].set_ylabel('RER', fontsize=12)
    axes[1, 0].grid(alpha=0.3)
    
    # (d) 私有链获胜概率
    axes[1, 1].plot(alpha_list, win_prob_list, 'm-', linewidth=2)
    axes[1, 1].axhline(0.5, color='k', linestyle='--', alpha=0.3)  # 50%基准线
    axes[1, 1].set_title('(d) Private Chain Winning Probability', fontsize=14)
    axes[1, 1].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[1, 1].set_ylabel('Probability', fontsize=12)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(alpha=0.3)
    
    # (e) 分叉率
    axes[2, 0].plot(alpha_list, fork_rate_list, 'c-', linewidth=2)
    axes[2, 0].set_title('(e) Fork Rate', fontsize=14)
    axes[2, 0].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[2, 0].set_ylabel('Rate', fontsize=12)
    axes[2, 0].grid(alpha=0.3)
    
    axes[2, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main()
    