"""
2D Example - Sanity Check Experiment
Reproduces the 2D_Example.ipynb logic as a standalone script.
Creates outputs/2d_plot.png and outputs/2d_data.csv.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Ensure outputs directory exists
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

np.random.seed(42)

n_per_client = 10

clients = {
    'Client A': {
        'pos_mean': [4.0, 4.0], 'neg_mean': [1.0, 1.0],
        'pos_cov': [[0.5, 0.1], [0.1, 0.5]],
        'neg_cov': [[0.6, -0.1], [-0.1, 0.6]],
        'n_pos': 7,
        'gender_bias_pos': 0.8,
        'gender_bias_neg': 0.2
    },
    'Client B': {
        'pos_mean': [1.5, 4.5], 'neg_mean': [3.5, 1.5],
        'pos_cov': [[0.4, 0.0], [0.0, 0.4]],
        'neg_cov': [[0.7, 0.2], [0.2, 0.8]],
        'n_pos': 3,
        'gender_bias_pos': 0.3,
        'gender_bias_neg': 0.7
    },
    'Client C': {
        'pos_mean': [5.5, 1.5], 'neg_mean': [1.5, 3.5],
        'pos_cov': [[0.5, -0.1], [-0.1, 0.5]],
        'neg_cov': [[0.6, 0.1], [0.1, 0.6]],
        'n_pos': 5,
        'gender_bias_pos': 0.5,
        'gender_bias_neg': 0.5
    }
}

data = []
for client, params in clients.items():
    n_pos = params['n_pos']
    n_neg = n_per_client - n_pos

    x_pos = np.random.multivariate_normal(params['pos_mean'], params['pos_cov'], n_pos)
    y_pos = np.ones(n_pos)
    s_pos = np.random.binomial(1, params['gender_bias_pos'], n_pos)

    # Ensure at least two male positives in Client A
    if client == 'Client A':
        n_male_pos = np.sum(s_pos == 0)
        if n_male_pos < 2:
            female_indices = np.where(s_pos == 1)[0]
            needed = 2 - n_male_pos
            for i in range(min(needed, len(female_indices))):
                s_pos[female_indices[i]] = 0

    x_neg = np.random.multivariate_normal(params['neg_mean'], params['neg_cov'], n_neg)
    y_neg = -np.ones(n_neg)
    s_neg = np.random.binomial(1, params['gender_bias_neg'], n_neg)

    X = np.vstack([x_pos, x_neg])
    y = np.hstack([y_pos, y_neg])
    s = np.hstack([s_pos, s_neg])
    client_col = [client] * n_per_client

    for i in range(n_per_client):
        data.append([X[i, 0], X[i, 1], s[i], y[i], client_col[i]])

df = pd.DataFrame(data, columns=['x1', 'x2', 's', 'y', 'client'])

# Print grouped counts
print("Grouped counts (client, y, s):")
print(df.groupby(["client", "y", "s"]).size())
print()

# Colors per client
client_colors = {'Client A': 'red', 'Client B': 'blue', 'Client C': 'green'}

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

for client, color in client_colors.items():
    client_data = df[df['client'] == client]
    for y_val, marker in [(1, 'o'), (-1, '^')]:
        for s_val, fill in [(0, 'filled'), (1, 'empty')]:
            subset = client_data[(client_data['y'] == y_val) & (client_data['s'] == s_val)]
            if subset.empty:
                continue
            if fill == 'filled':
                ax.scatter(subset['x1'], subset['x2'],
                           marker=marker,
                           facecolor=color,
                           edgecolor='none' if marker == 'o' else color,
                           alpha=0.8, s=80, linewidth=1)
            else:
                ax.scatter(subset['x1'], subset['x2'],
                           marker=marker,
                           facecolor='none',
                           edgecolor=color,
                           alpha=0.8, s=80, linewidth=2)

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Dataset with 10 points per client, label & gender imbalance, separated classes')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)

# Legend: only client colors (upper left)
legend_elements = [Patch(facecolor=color, edgecolor='none', label=client) for client, color in client_colors.items()]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Text box with marker explanations (upper right)
textstr = 'Filled = Male\nEmpty = Female\nCircles = +1\nTriangles = -1'
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='black')
ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()

# Save plot
plot_path = os.path.join(OUTPUTS_DIR, "2d_plot.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to {plot_path}")

# Save dataframe
csv_path = os.path.join(OUTPUTS_DIR, "2d_data.csv")
df.to_csv(csv_path, index=False)
print(f"Data saved to {csv_path}")

plt.close()
