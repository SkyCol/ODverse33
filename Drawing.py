import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import seaborn as sns

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 27
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["xtick.labelsize"] = 40
plt.rcParams["ytick.labelsize"] = 40

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['xtick.major.pad'] = 10
plt.rcParams['ytick.major.pad'] = 10

fig, axes = plt.subplots(1, 3, figsize=(48, 13))
plt.tight_layout(pad=4)

models1 = ["v5", "v6", "v7", "v8", "v9", "v10", "v11"]
coco_val = [45.4, 50.0, 51.4, 50.2, 51.4, 51.1, 51.5]
ours_val = [61.48, 59.35, 61.34, 61.89, 62.13, 61.95, 62.53]
ours_test = [59.78, 56.55, 58.34, 59.69, 59.53, 59.10, 60.53]

colors_line = sns.color_palette("colorblind", 3)

ax1 = axes[0]
ax1.plot(models1, coco_val, marker='o', markersize=28, label=r"$\mathbf{COCO_{ val}}$",
         linestyle='-', linewidth=5, color=colors_line[0])
ax1.plot(models1, ours_val, marker='o', markersize=28, label=r"$\mathbf{OURS_{ val}}$",
         linestyle='--', linewidth=5, color=colors_line[1])
ax1.plot(models1, ours_test, marker='o', markersize=28, label=r"$\mathbf{OURS_{ test}}$",
         linestyle='-.', linewidth=5, color=colors_line[2])

ax1.set_ylabel(r"$\mathbf{mAP_{50-95} (\%)}$", fontname='Arial', fontsize=42, fontweight='bold')

font_prop = FontProperties()
font_prop.set_weight('bold')

ax1.legend(fontsize=30, loc='lower right', bbox_to_anchor=(0.98, 0.02), frameon=True, 
           title_fontsize=30, prop=font_prop, markerscale=1, borderpad=1.1, labelspacing=1.1)
ax1.set_ylim(40, 65)
ax1.set_yticks(np.arange(40, 66, 5))

ax1.grid(False)
ax1.minorticks_off()

ax1.tick_params(axis='x', length=10, width=4, which='major', direction='in')
ax1.tick_params(axis='y', length=10, width=4, which='major', direction='in')

for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(4)

models2 = ["v5", "v6", "v7", "v8", "v9", "v10 ", "v11"]
ours_small  = np.array([0.368352778, 0.311180556, 0.356002778, 0.368931019, 0.381449074, 0.355515741, 0.37935463])
ours_medium = np.array([0.551184553, 0.500733333, 0.546139837, 0.545868293, 0.556832927, 0.547228049, 0.558750407])
ours_large  = np.array([0.670845635, 0.627299206, 0.66867619, 0.673465079, 0.677045238, 0.668137302, 0.676865873])

ours_small  = ours_small  * 100
ours_medium = ours_medium * 100
ours_large  = ours_large  * 100

x = np.arange(len(models2))

ax2 = axes[1]
ax2.plot(x, ours_small, marker='o', markersize=28, label=r"$\mathbf{OURS_{ small}}$",
         linestyle='-', linewidth=5, color=colors_line[0])
ax2.plot(x, ours_medium, marker='o', markersize=28, label=r"$\mathbf{OURS_{ medium}}$",
         linestyle='--', linewidth=5, color=colors_line[1])
ax2.plot(x, ours_large, marker='o', markersize=28, label=r"$\mathbf{OURS_{ large}}$",
         linestyle='-.', linewidth=5, color=colors_line[2])

ax2.set_ylabel(r"$\mathbf{mAP_{50-95} (\%)}$", fontname='Arial', fontsize=42, fontweight='bold')
ax2.legend(fontsize=30, loc='lower right', bbox_to_anchor=(0.99, 0.01), frameon=True, 
           title_fontsize=30, prop=font_prop, markerscale=1, borderpad=1.05, labelspacing=1.05)
ax2.set_xticks(x)
ax2.set_xticklabels(models2, rotation=0)
ax2.margins(x=0.05)

ax2.grid(False)
ax2.minorticks_off()

ax2.tick_params(axis='x', length=10, width=4, which='major', direction='in')
ax2.tick_params(axis='y', length=10, width=4, which='major', direction='in')

ax2.set_ylim(10, 80)
for spine in ax2.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(4)

models3 = ["YOLOv11", "YOLOv10", "YOLOv9", "YOLOv8", "YOLOv7", "YOLOv6", "YOLOv5"]
params = np.array([20.1, 35.1, 20.1, 25.9, 36.9, 28.1, 25.1])
latency = np.array([8.80, 9.97, 10.85, 7.58, 9.94, 8.72, 7.80])
bubble_values = params * latency
scale_factor = 0.2
bubble_sizes = (bubble_values ** 2) * scale_factor

custom_colors = ['#f1c232','#1f77b4', '#ff7f0e', '#e377c2','#2ca02c', '#d62728', '#9467bd']

ax3 = axes[2]

for i in range(len(models3)):
    ax3.text(params[i] + 0.5, latency[i], models3[i], fontsize=32, ha='center', va='center', fontweight='bold', color='#333333')
    ax3.scatter(params[i], latency[i], s=bubble_sizes[i], color=custom_colors[i], 
                edgecolors="#ffffff", linewidth=3, zorder=5, alpha=0.95)

    ax3.scatter(params[i], latency[i], s=bubble_sizes[i] * 0.65, color="white", 
                alpha=0.05, zorder=6)

ax3.set_facecolor('#ffffff')

ax3.set_xlabel(r"$\mathbf{Params (M)}$", fontname='Arial', fontsize=40, fontweight='bold', color='#333333')
ax3.set_ylabel(r"$\mathbf{A100 FP16 (ms/image)}$", fontname='Arial', fontsize=40, fontweight='bold', color='#333333')

ax3.tick_params(axis='x', labelsize=35)
ax3.tick_params(axis='y', labelsize=35)

ax3.grid(False)
ax3.minorticks_off()

ax3.tick_params(axis='x', length=12, width=4, which='major', direction='in')
ax3.tick_params(axis='y', length=12, width=4, which='major', direction='in')

ax3.set_xlim(14, 40)
ax3.set_ylim(6, 12)

for spine in ax3.spines.values():
    spine.set_edgecolor('#333333')
    spine.set_linewidth(4)

plt.tight_layout()
plt.savefig("1.svg", format="svg")
plt.show()
