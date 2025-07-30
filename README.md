# Blockchain Mining Strategy Experiment Project

## Project Overview

This project contains the experimental code for the paper titled "Novel Bribery Mining Attacks: Impacts on Mining Ecosystem and the “Bribery Miner’s Dilemma” in the Nakamoto-style Blockchain System", organized into two directories:
- **SUM**: Experiments on BSM (Bribed Selfish Mining) strategy
- **SUUM**: Experiments on BSSM (Bribed Semi-Selfish Mining) strategy

## Directory Structure
```
.
├── SUM/                  # BSM Strategy Experiments
│   ├── SUM_EXP1.py       # Experiment 1: Basic RER Analysis
│   ├── SUM_EXP2.py       # Experiment 2: Multi-Scenario RER Comparison
│   ├── SUM_EXP3.py       # Experiment 3: Impact of Different β_b Values
│   └── SUM_EXP4.py       # Experiment 4: RER and Winning Conditions
├── SUUM/                 # BSSM Strategy Experiments
│   ├── SUUM_EXP1.py      # Experiment 1: Basic RER Analysis
│   ├── SUUM_EXP2.py      # Experiment 2: Multi-Strategy Comparison
│   ├── SUUM_EXP3.py      # Experiment 3: Impact of γ Parameter
│   └── SUUM_EXP4.py      # Experiment 4: RER and Winning Conditions
└── TDSC '25 full version.pdf  # Full Paper
```

## SUM Experiment Summary (BSM Strategy)

### Experiment Objective
Evaluate the Relative Earnings Rise (RER) of the BSM strategy under different attacker hashpower ($\alpha$), for the following parties:
- Attacker (a)
- Other mining pools (o)
- Target pool (b)

### Key Parameters
- $\beta_b$ = 0.1 (Target pool hashpower proportion)
- $\epsilon$ = 0.02 (Bribery proportion)
- $\alpha \in [0.01, 0.4]$ (Range of total attacker hashpower)

### Key Findings
1. **EXP1**: Attacker RER increases with $\alpha$, while target pool RER shows nonlinear changes
2. **EXP2**: BSM achieves significantly higher earnings compared to Honest Mining (H) and Selfish Mining (SM)
3. **EXP3**: As $\beta_b$ increases, the attacker’s RER improvement diminishes
4. **EXP4**: When attacker hashpower > 0.3, fork rate significantly increases

## SUUM Experiment Summary (BSSM Strategy)

### Experiment Objective
Evaluate the performance of the BSSM strategy under the following scenarios:
- Different attacker hashpower ($\alpha$)
- Honest mining power proportion ($\rho$)
- Probability of other pools joining the private chain ($\gamma$)

### Key Parameters
- ρ = 0.1 (Honest mining power proportion)
- γ = 0.5 (Probability of joining private chain)
- ε = 0.02 (Bribery proportion)

### Key Findings
1. **EXP1**: BSSM significantly improves attacker earnings compared to non-bribery strategy (BSSM')
2. **EXP2**: BSSM achieves a 20-40% RER improvement over Semi-Selfish Mining (SSM)
3. **EXP3**: Increasing $\gamma$ improves the winning probability of the private chain
4. **EXP4**: When attacker hashpower > 0.35, fork rate exceeds 0.25

## Running Instructions
1. Install dependencies：
```bash
pip install numpy matplotlib
```
2. Run a single experiment：
```bash
python SUM/SUM_EXP1.py
```
3. All experiments output visual charts

> Note: For detailed experimental design and parameter settings, please refer to the comments in each experiment file.
