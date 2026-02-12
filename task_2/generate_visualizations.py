import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.use('Agg')

# Save images in the same folder as the script
output_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. ATTENTION MECHANISM VISUALIZATION
# ============================================================

# --- 1a. Attention Heatmap ---
sentence = ["The", "firewall", "blocked", "the", "attack"]
attention_weights = np.array([
    [0.15, 0.05, 0.10, 0.60, 0.10],
    [0.08, 0.30, 0.35, 0.07, 0.20],
    [0.05, 0.25, 0.10, 0.10, 0.50],
    [0.55, 0.10, 0.05, 0.20, 0.10],
    [0.05, 0.20, 0.45, 0.05, 0.25],
])

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(attention_weights, cmap='YlOrRd', vmin=0, vmax=0.65)

ax.set_xticks(range(len(sentence)))
ax.set_yticks(range(len(sentence)))
ax.set_xticklabels(sentence, fontsize=14, fontweight='bold')
ax.set_yticklabels(sentence, fontsize=14, fontweight='bold')
ax.set_xlabel("Keys", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("Queries", fontsize=14, fontweight='bold', labelpad=10)
ax.set_title("Self-Attention Weights\n\"The firewall blocked the attack\"", fontsize=16, fontweight='bold', pad=15)

for i in range(len(sentence)):
    for j in range(len(sentence)):
        color = 'white' if attention_weights[i, j] > 0.35 else 'black'
        ax.text(j, i, f'{attention_weights[i, j]:.2f}', ha='center', va='center',
                fontsize=13, fontweight='bold', color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Attention Weight', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'attention_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("1/4 - attention_heatmap.png saved")

# --- 1b. Multi-Head Attention Visualization ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

head_names = ["Head 1: Syntactic", "Head 2: Semantic", "Head 3: Positional", "Head 4: Entity"]
cmaps = ['Blues', 'Greens', 'Purples', 'Oranges']

np.random.seed(42)
head_weights = [
    np.array([[0.1,0.6,0.1,0.1,0.1],[0.1,0.2,0.5,0.1,0.1],[0.1,0.5,0.1,0.1,0.2],[0.1,0.1,0.1,0.6,0.1],[0.1,0.1,0.5,0.1,0.2]]),
    np.array([[0.1,0.1,0.1,0.6,0.1],[0.1,0.2,0.1,0.1,0.5],[0.1,0.1,0.1,0.1,0.6],[0.5,0.1,0.1,0.2,0.1],[0.1,0.1,0.6,0.1,0.1]]),
    np.array([[0.5,0.3,0.1,0.05,0.05],[0.3,0.4,0.2,0.05,0.05],[0.05,0.2,0.5,0.2,0.05],[0.05,0.05,0.2,0.4,0.3],[0.05,0.05,0.05,0.3,0.55]]),
    np.array([[0.15,0.05,0.1,0.6,0.1],[0.08,0.3,0.35,0.07,0.2],[0.05,0.25,0.1,0.1,0.5],[0.55,0.1,0.05,0.2,0.1],[0.05,0.2,0.45,0.05,0.25]]),
]

for idx, (ax, weights, name, cmap) in enumerate(zip(axes, head_weights, head_names, cmaps)):
    im = ax.imshow(weights, cmap=cmap, vmin=0, vmax=0.65)
    ax.set_xticks(range(len(sentence)))
    ax.set_yticks(range(len(sentence)))
    ax.set_xticklabels(sentence, fontsize=9, rotation=45)
    ax.set_yticklabels(sentence, fontsize=9)
    ax.set_title(name, fontsize=12, fontweight='bold')
    for i in range(len(sentence)):
        for j in range(len(sentence)):
            color = 'white' if weights[i, j] > 0.35 else 'black'
            ax.text(j, i, f'{weights[i, j]:.2f}', ha='center', va='center', fontsize=8, color=color)

fig.suptitle("Multi-Head Attention - Each Head Learns Different Patterns", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'multihead_attention.png'), dpi=200, bbox_inches='tight')
plt.close()
print("2/4 - multihead_attention.png saved")

# ============================================================
# 2. POSITIONAL ENCODING VISUALIZATION
# ============================================================

d_model = 64
max_len = 50

pe = np.zeros((max_len, d_model))
position = np.arange(0, max_len)[:, np.newaxis]
div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

pe[:, 0::2] = np.sin(position * div_term)
pe[:, 1::2] = np.cos(position * div_term)

# --- 2a. Positional Encoding Heatmap ---
fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(pe, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

ax.set_xlabel("Encoding Dimension", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("Position in Sequence", fontsize=14, fontweight='bold', labelpad=10)
ax.set_title("Sinusoidal Positional Encoding\nPE(pos, 2i) = sin(pos / 10000^(2i/d))   |   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))",
             fontsize=14, fontweight='bold', pad=15)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Encoding Value', fontsize=12)

ax.set_xticks(np.arange(0, d_model, 8))
ax.set_yticks(np.arange(0, max_len, 5))

ax.annotate('High frequency\n(changes rapidly)', xy=(2, 45), fontsize=10,
            fontweight='bold', color='black', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
ax.annotate('Low frequency\n(changes slowly)', xy=(58, 45), fontsize=10,
            fontweight='bold', color='black', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'positional_encoding_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("3/4 - positional_encoding_heatmap.png saved")

# --- 2b. Positional Encoding Sinusoidal Waves ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

dims_to_plot = [0, 1, 10, 11]
titles = ["Dimension 0 (sin) - High Freq", "Dimension 1 (cos) - High Freq",
          "Dimension 10 (sin) - Low Freq", "Dimension 11 (cos) - Low Freq"]
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

positions = np.arange(0, max_len)

for idx, (ax, dim, title, color) in enumerate(zip(axes.flat, dims_to_plot, titles, colors)):
    ax.plot(positions, pe[:, dim], color=color, linewidth=2.5, label=f'dim={dim}')
    ax.fill_between(positions, pe[:, dim], alpha=0.15, color=color)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Position", fontsize=11)
    ax.set_ylabel("Encoding Value", fontsize=11)
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

fig.suptitle("Positional Encoding - Sinusoidal Waves at Different Dimensions",
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'positional_encoding_waves.png'), dpi=200, bbox_inches='tight')
plt.close()
print("4/4 - positional_encoding_waves.png saved")

print(f"\nAll visualizations saved in: {output_dir}")
