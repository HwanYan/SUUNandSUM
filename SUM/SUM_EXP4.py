import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# --------------------------
# 1. 参数设置
# --------------------------
beta_b = 0.1    # 目标矿池算力占比
epsilon = 0.02  # 贿赂比例
alpha_list = np.linspace(0.01, 0.5, 25)  # 攻击者总算力范围
n_simulations = 100  # 蒙特卡洛模拟次数
converge_threshold = 1e-8  # 收敛阈值
converge_window = 10  # 连续收敛轮次
max_steps = 1000000  # 最大模拟步数

# --------------------------
# 2. BSM状态转换规则与获胜记录
# --------------------------
def simulate_and_track_metrics(alpha, beta_b, max_steps):
    """模拟状态转换并跟踪私有链获胜情况和分叉率"""
    state_counts = defaultdict(int)
    current_state = '0'  # 初始状态
    private_chain_wins = 0  # 私有链获胜次数
    total_battles = 0       # 总分叉竞争次数
    
    for step in range(max_steps):
        # 记录当前状态用于后续统计
        state_counts[current_state] += 1
        
        # 检查是否处于分叉状态并记录竞争结果
        if current_state in ['0_o', '0_b']:
            total_battles += 1
            # 判断私有链是否获胜
            if check_private_win(current_state, alpha, beta_b):
                private_chain_wins += 1
        
        # 转换到下一状态
        current_state = get_next_state(current_state, alpha, beta_b)
    
    # 计算稳态概率
    total = sum(state_counts.values())
    steady_probs = {s: cnt / total for s, cnt in state_counts.items()}
    
    # 计算私有链获胜概率和分叉率
    win_prob = private_chain_wins / total_battles if total_battles > 0 else 0
    fork_rate = (state_counts.get('0_o', 0) + state_counts.get('0_b', 0)) / total if total > 0 else 0
    
    return steady_probs, win_prob, fork_rate

def get_next_state(current_state, alpha, beta_b):
    """根据当前状态和转换概率返回下一状态"""
    if current_state == '0':
        # 状态0的转换
        p_selfish = alpha  # 攻击者发现区块，形成私有链
        if np.random.rand() < p_selfish:
            return '1'  # 私有链长1
        else:
            return '0'  # 其他矿池发现区块，保持初始状态
    
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
                # 私有链仅领先1个，形成分叉
                p_b = beta_b / (1 - alpha) if (1 - alpha) > 0 else 0
                if np.random.rand() < p_b:
                    return '0_b'  # 与目标矿池的分叉
                else:
                    return '0_o'  # 与其他矿池的分叉
            else:
                return str(k - 1)  # 私有链优势缩小
    
    elif current_state.endswith('_') and current_state[:-1].isdigit():
        # 公共链领先状态k_（k≥1）的转换
        k = int(current_state[:-1])
        p_selfish = alpha  # 攻击者发现区块，私有链延长
        p_other = 1 - alpha  # 其他矿池发现区块，公共链延长
        
        rand = np.random.rand()
        if rand < p_selfish:
            if k == 1:
                return '0'  # 回到平衡状态
            else:
                return f"{k-1}_"  # 公共链优势缩小
        else:
            return f"{k+1}_"  # 公共链优势扩大
    
    elif current_state == '0_o':
        # 与其他矿池的分叉状态转换
        p_selfish = alpha  # 攻击者发现区块，私有链获胜
        if np.random.rand() < p_selfish:
            return '0'  # 私有链获胜
        else:
            return '1_'  # 公共链获胜
    
    elif current_state == '0_b':
        # 与目标矿池的分叉状态转换
        p_selfish = alpha  # 攻击者发现区块，私有链获胜
        p_b = beta_b  # 目标矿池发现区块（接受贿赂）
        if np.random.rand() < p_selfish + p_b:
            return '0'  # 私有链获胜（含贿赂情况）
        else:
            return '1_'  # 公共链获胜
    else:
        return '0'  # 未知状态默认返回初始状态

def check_private_win(state, alpha, beta_b):
    """判断在分叉状态中私有链是否获胜"""
    rand = np.random.rand()
    
    if state == '0_o':
        # 与其他矿池的分叉
        p_selfish_win = alpha / (alpha + (1 - alpha - beta_b)) if (1 - beta_b) > 0 else 1
        return rand < p_selfish_win
    
    elif state == '0_b':
        # 与目标矿池的分叉（含贿赂）
        p_total_private = alpha + beta_b  # 攻击者+接受贿赂的目标矿池
        p_private_win = p_total_private / (p_total_private + (1 - alpha - beta_b)) if (1 - beta_b) > 0 else 1
        return rand < p_private_win
    
    return False

# --------------------------
# 3. 计算RER和获胜条件
# --------------------------
def calculate_metrics(alpha, beta_b, epsilon, steady_probs, win_prob, fork_rate):
    """计算RER和其他性能指标"""
    # 提取关键状态概率
    p0 = steady_probs.get('0', 0)
    p1 = steady_probs.get('1', 0)
    p1_prime = steady_probs.get('1_', 0)
    p0_o = steady_probs.get('0_o', 0)
    p0_b = steady_probs.get('0_b', 0)
    
    # 高阶状态概率
    pk_high = sum(v for k, v in steady_probs.items() if k.isdigit() and int(k) >= 2)
    pk_prime_high = sum(v for k, v in steady_probs.items() if k.endswith('_') and int(k[:-1]) >= 2)
    
    # 攻击者a的RER (BSM vs BSM')
    reward_bsm_a = (1 - epsilon) * (
        p0 * alpha + p1 * 1.2 + pk_high * 1.5 + 
        p0_b * 2.0 + p0_o * 1.8
    )
    reward_bsm_prime_a = (
        p0 * alpha + p1 * 1.0 + pk_high * 1.2 + 
        p0_b * 1.0 + p0_o * 1.5
    )
    rer_a = (reward_bsm_a - reward_bsm_prime_a) / (reward_bsm_prime_a + 1e-10)
    
    # 目标矿池b的RER (BSM vs BSM')
    reward_bsm_b = (
        p0 * beta_b + 
        epsilon * reward_bsm_a +  # 贿赂收益
        p0_b * 0.3  # 分叉状态额外收益
    )
    reward_bsm_prime_b = (
        p0 * beta_b + 
        p0_b * 0.1  # 无贿赂时分叉收益
    )
    rer_b = (reward_bsm_b - reward_bsm_prime_b) / (reward_bsm_prime_b + 1e-10)
    
    # 其他矿池o的RER (BSM vs BSM')
    reward_bsm_o = (
        p0 * (1 - alpha - beta_b) + 
        p1_prime * 1.1 + pk_prime_high * 1.3 - 
        p0_o * 0.2  # 分叉损失
    )
    reward_bsm_prime_o = (
        p0 * (1 - alpha - beta_b) + 
        p1_prime * 1.1 + pk_prime_high * 1.3
    )
    rer_o = (reward_bsm_o - reward_bsm_prime_o) / (reward_bsm_prime_o + 1e-10)
    
    return rer_a, rer_b, rer_o, win_prob, fork_rate

# --------------------------
# 4. 主程序：模拟与绘图
# --------------------------
def main():
    print(f"开始模拟 (BSM中的RER和获胜条件)...")
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
            steady_probs, win_prob, fork_rate = simulate_and_track_metrics(
                alpha, beta_b, max_steps
            )
            metrics.append(calculate_metrics(
                alpha, beta_b, epsilon, steady_probs, win_prob, fork_rate
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
    fig.suptitle('RER and winning conditions in BSM', fontsize=16)
    
    # (a) 攻击者a的RER
    axes[0, 0].plot(alpha_list, rer_a_list, 'b-', linewidth=2)
    axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_title('(a) RER_a^(BSM,BSM\')', fontsize=14)
    axes[0, 0].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[0, 0].set_ylabel('RER', fontsize=12)
    axes[0, 0].grid(alpha=0.3)
    
    # (b) 目标矿池b的RER
    axes[0, 1].plot(alpha_list, rer_b_list, 'g-', linewidth=2)
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_title('(b) RER_b^(BSM,BSM\')', fontsize=14)
    axes[0, 1].set_xlabel('Adversary Mining Power α', fontsize=12)
    axes[0, 1].set_ylabel('RER', fontsize=12)
    axes[0, 1].grid(alpha=0.3)
    
    # (c) 其他矿池o的RER
    axes[1, 0].plot(alpha_list, rer_o_list, 'r-', linewidth=2)
    axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('(c) RER_o^(BSM,BSM\')', fontsize=14)
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
    
    # 隐藏最后一个子图
    axes[2, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main()
    