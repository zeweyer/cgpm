import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
})

df = pd.read_csv("gp_comparison_summary.csv")

# Construct the relative overfitting difference
df['relative_gap'] = (df['MAE test'] - df['MAE train']) / df['MAE train']

# Pareto's frontier judgment function (less is better)
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(costs[is_efficient] < c, axis=1) |
                np.all(costs[is_efficient] == c, axis=1)
            )
            is_efficient[i] = True
    return is_efficient

# complexity vs relative_gap
cost_array = df[['complexity', 'relative_gap']].values
pareto_mask = is_pareto_efficient(cost_array)
df['pareto'] = pareto_mask


color_map = {'r': '#1f77b4', 'q': '#2ca02c', 'r+q': '#ff7f0e'} 

fig, ax = plt.subplots(figsize=(6, 4.5))

for input_type in df['inputs'].unique():
    subset = df[df['inputs'] == input_type]
    ax.scatter(subset['complexity'], subset['relative_gap'],
               label=input_type,
               color=color_map.get(input_type, 'gray'),
               s=60,
               edgecolors='black',
               linewidths=0.5,
               alpha=0.8)

# Pareto Line
pareto_points = df[df['pareto']].sort_values(by='complexity')
ax.plot(pareto_points['complexity'], pareto_points['relative_gap'],
        color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Pareto Front')

for _, row in df.iterrows():
    ax.text(row['complexity'], row['relative_gap'], str(row['ID']),
            fontsize=8, ha='right', va='bottom', color='black')

ax.set_xlabel('Complexity')
ax.set_ylabel('Relative Overfitting')
#ax.set_title('Pareto Analysis of Model Complexity vs Overfitting')
ax.legend(frameon=False)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

plt.tight_layout()
plt.show()

