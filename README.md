# 区块链挖矿策略实验项目

## 项目概述
本项目包含TDSC 2025论文《Novel Bribery Mining Attacks: Impacts on Mining Ecosystem and the “Bribery Miner’s Dilemma” in the Nakamoto-style Blockchain System》的实验代码，分为两个目录：
- **SUM**：BSM（Bribed Selfish Mining）策略实验
- **SUUM**：BSSM（Bribed Semi-Selfish Mining）策略实验

论文全文见：`TDSC '25 full version.pdf`

## 目录结构
```
.
├── SUM/                  # BSM策略实验
│   ├── SUM_EXP1.py       # 实验1：基础RER分析
│   ├── SUM_EXP2.py       # 实验2：多场景RER对比
│   ├── SUM_EXP3.py       # 实验3：不同βb值影响
│   └── SUM_EXP4.py       # 实验4：RER与获胜条件
├── SUUM/                 # BSSM策略实验
│   ├── SUUM_EXP1.py      # 实验1：基础RER分析
│   ├── SUUM_EXP2.py      # 实验2：多策略对比
│   ├── SUUM_EXP3.py      # 实验3：γ参数影响
│   └── SUUM_EXP4.py      # 实验4：RER与获胜条件
└── TDSC '25 full version.pdf  # 论文全文
```

## SUM实验总结（BSM策略）

### 实验目的
评估BSM策略在不同攻击者算力(α)下，对以下各方的相对收益提升(RER)：
- 攻击者(a)
- 其他矿池(o)
- 目标矿池(b)

### 关键参数
- βb = 0.1（目标矿池算力占比）
- ε = 0.02（贿赂比例）
- α ∈ [0.01, 0.4]（攻击者总算力范围）

### 主要发现
1. **EXP1**：攻击者RER随α增加而提升，目标矿池RER呈非线性变化
2. **EXP2**：BSM相比诚实挖矿(H)和自私挖矿(SM)有显著收益提升
3. **EXP3**：βb增大时，攻击者RER提升幅度减小
4. **EXP4**：攻击者算力>0.3时，分叉率显著增加

## SUUM实验总结（BSSM策略）

### 实验目的
评估BSSM策略在以下场景的表现：
- 不同攻击者算力(α)
- 诚实挖矿算力占比(ρ)
- 其他矿池加入私有链概率(γ)

### 关键参数
- ρ = 0.1（诚实挖矿算力占比）
- γ = 0.5（加入私有链概率）
- ε = 0.02（贿赂比例）

### 主要发现
1. **EXP1**：BSSM相比无贿赂策略(BSSM')显著提升攻击者收益
2. **EXP2**：BSSM相比半自私挖矿(SSM)有20-40%的RER提升
3. **EXP3**：γ增大可提高私有链获胜概率
4. **EXP4**：攻击者算力>0.35时，分叉率超过0.25

## 运行说明
1. 安装依赖：
```bash
pip install numpy matplotlib
```
2. 运行单个实验：
```bash
python SUM/SUM_EXP1.py
```
3. 所有实验均输出可视化图表

> 注：详细实验设计和参数设置请参考各实验文件注释
