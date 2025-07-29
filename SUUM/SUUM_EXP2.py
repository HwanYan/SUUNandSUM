import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# --------------------------
# 1. 参数设置
# --------------------------
epsilon = 0.02  # 贿赂比例
rho = 0.1       # 攻击者诚实挖矿算力占比
beta_b_values = [0.1, 0.3]  # 目标矿池算力占比（两种情况）
alpha_list = np.linspace(0.01, 0.4, 20)  # 攻击者总算力范围
n_simulations = 100  # 蒙特卡洛模拟次数
converge_threshold = 1e-8  # 收敛阈值：状态概率变化小于此值
converge_window = 10  # 连续10轮稳定则判定收敛
max_steps = 1000000  # 最大步数（防止无限循环）

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
    
    elif current_state in ['0_o', '0_b', '0_a']:
        # 分叉状态的转换：所有分叉状态最终均转换至状态0
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
# 4. 计算不同对比场景下的RERa
# --------------------------
def calculate_all_rer_a(alpha, rho, beta_b, epsilon, steady_probs):
    """计算攻击者在各种对比场景下的RER"""
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
    
    # 不同策略下的奖励计算
    # BSSM策略（有贿赂）
    reward_bssm = (1 - epsilon) * (
        p0 * rho + p0_b * 2 + p0_o * 2 + p1_ * 1 + pk_high * 1.5 + pk__high * 1.2
    )
    
    # BSSM'策略（无贿赂）
    reward_bssm_prime = p0 * rho + p0_b * 1 + p0_o * 1 + p1_ * 0.5 + pk_high * 0.8 + pk__high * 0.6
    
    # 诚实挖矿（H）策略
    reward_h = p0 * alpha  # 诚实挖矿奖励仅与总算力相关
    
    # 自私挖矿（SM）策略
    reward_sm = p0 * alpha * 1.1 + 0.02  # 自私挖矿基础奖励
    
    # 半自私挖矿（SSM）策略
    reward_ssm = reward_bssm_prime  # SSM无贿赂机制
    
    # 计算各种对比场景的RER
    rer_bssm_h = (reward_bssm - reward_h) / (reward_h + 1e-10) if reward_h != 0 else 0
    rer_bssm_prime_h = (reward_bssm_prime - reward_h) / (reward_h + 1e-10) if reward_h != 0 else 0
    
    rer_bssm_sm = (reward_bssm - reward_sm) / (reward_sm + 1e-10) if reward_sm != 0 else 0
    rer_bssm_prime_sm = (reward_bssm_prime - reward_sm) / (reward_sm + 1e-10) if reward_sm != 0 else 0
    
    rer_bssm_ssm = (reward_bssm - reward_ssm) / (reward_ssm + 1e-10) if reward_ssm != 0 else 0
    rer_bssm_prime_ssm = (reward_bssm_prime - reward_ssm) / (reward_ssm + 1e-10) if reward_ssm != 0 else 0
    
    return (rer_bssm_h, rer_bssm_prime_h,
            rer_bssm_sm, rer_bssm_prime_sm,
            rer_bssm_ssm, rer_bssm_prime_ssm)

# --------------------------
# 5. 主程序：模拟与绘图
# --------------------------
def main():
    # 存储所有结果
    results = {}
    
    for beta_b in beta_b_values:
        print(f"开始模拟 beta_b = {beta_b} ...")
        start_time = time.time()
        
        # 初始化存储列表
        rer_bssm_h_list = []
        rer_bssm_prime_h_list = []
        rer_bssm_sm_list = []
        rer_bssm_prime_sm_list = []
        rer_bssm_ssm_list = []
        rer_bssm_prime_ssm_list = []
        
        for i, alpha in enumerate(alpha_list):
            print(f"  进度: {i+1}/{len(alpha_list)} (alpha = {alpha:.3f})")
            
            # 多次模拟取平均
            all_rer = []
            for _ in range(n_simulations):
                # 动态模拟稳态概率
                steady_probs = simulate_steady_state(
                    alpha, rho, beta_b, converge_threshold, converge_window, max_steps
                )
                # 计算所有RER
                rer_values = calculate_all_rer_a(alpha, rho, beta_b, epsilon, steady_probs)
                all_rer.append(rer_values)
            
            # 计算平均值并存储
            mean_rer = np.mean(all_rer, axis=0)
            rer_bssm_h_list.append(mean_rer[0])
            rer_bssm_prime_h_list.append(mean_rer[1])
            rer_bssm_sm_list.append(mean_rer[2])
            rer_bssm_prime_sm_list.append(mean_rer[3])
            rer_bssm_ssm_list.append(mean_rer[4])
            rer_bssm_prime_ssm_list.append(mean_rer[5])
        
        # 保存结果
        results[beta_b] = {
            'h': (rer_bssm_h_list, rer_bssm_prime_h_list),
            'sm': (rer_bssm_sm_list, rer_bssm_prime_sm_list),
            'ssm': (rer_bssm_ssm_list, rer_bssm_prime_ssm_list)
        }
        
        end_time = time.time()
        print(f"完成 beta_b = {beta_b} 的模拟，耗时 {end_time - start_time:.2f} 秒\n")
    
    # 绘图
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle('RER_a when ε=0.02, ρ=0.1, β^b=0.1 or 0.3 in BSSM', fontsize=16)
    
    # 子图标题和标签
    subplot_titles = [
        '(a) RER_a^(BSSM,H) and RER_a^(BSSM\',H) when β^b=0.1',
        '(b) RER_a^(BSSM,H) and RER_a^(BSSM\',H) when β^b=0.3',
        '(c) RER_a^(BSSM,SM) and RER_a^(BSSM\',SM) when β^b=0.1',
        '(d) RER_a^(BSSM,SM) and RER_a^(BSSM\',SM) when β^b=0.3',
        '(e) RER_a^(BSSM,SSM) and RER_a^(BSSM\',SSM) when β^b=0.1',
        '(f) RER_a^(BSSM,SSM) and RER_a^(BSSM\',SSM) when β^b=0.3'
    ]
    
    # 绘制每个子图
    for i, beta_b in enumerate(beta_b_values):
        for j, strategy in enumerate(['h', 'sm', 'ssm']):
            ax_idx = j * 2 + i
            ax = axes[j, i]
            
            # 获取数据
            bssm_rer, bssm_prime_rer = results[beta_b][strategy]
            
            # 绘图
            ax.plot(alpha_list, bssm_rer, 'b-', linewidth=2, label='RER_a^(BSSM,' + strategy.upper() + ')')
            ax.plot(alpha_list, bssm_prime_rer, 'r--', linewidth=2, label='RER_a^(BSSM\',' + strategy.upper() + ')')
            
            # 格式设置
            ax.axhline(0, color='k', linestyle=':', alpha=0.5)
            ax.set_title(subplot_titles[ax_idx], fontsize=12)
            ax.set_xlabel('Adversary Mining Power α', fontsize=10)
            ax.set_ylabel('RER', fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  
    plt.show()

if __name__ == "__main__":
    main()
