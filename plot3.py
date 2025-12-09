import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

# --- DONNÉES BRUTES ADHD ---
# (GCN_Correct, GCN_AUC, Lin_Correct, Lin_AUC, Total)

# Baseline (Sans Age) - "hyperparametre Camille sans age"
data_no_age = [
    (40, 0.779, 28, 0.520, 58), (41, 0.826, 31, 0.559, 58), (46, 0.828, 34, 0.576, 58), 
    (36, 0.604, 35, 0.687, 58), (38, 0.628, 30, 0.473, 58), (35, 0.707, 36, 0.630, 58), 
    (44, 0.766, 28, 0.416, 58), (41, 0.759, 30, 0.499, 58), (44, 0.738, 39, 0.657, 59), 
    (38, 0.757, 39, 0.636, 59)
]

# Sigma = 2
data_sigma_2 = [
    (40, 0.760, 28, 0.520, 58), (41, 0.830, 31, 0.559, 58), (42, 0.739, 34, 0.576, 58), 
    (31, 0.663, 35, 0.687, 58), (37, 0.641, 30, 0.473, 58), (37, 0.699, 36, 0.630, 58), 
    (39, 0.689, 28, 0.416, 58), (42, 0.719, 30, 0.499, 58), (44, 0.711, 39, 0.657, 59), 
    (41, 0.775, 39, 0.636, 59)
]

# Sigma = 4
data_sigma_4 = [
    (39, 0.747, 28, 0.520, 58), (42, 0.855, 31, 0.559, 58), (43, 0.773, 34, 0.576, 58), 
    (37, 0.717, 35, 0.687, 58), (39, 0.638, 30, 0.473, 58), (39, 0.723, 36, 0.630, 58), 
    (39, 0.722, 28, 0.416, 58), (42, 0.717, 30, 0.499, 58), (43, 0.663, 39, 0.657, 59), 
    (40, 0.799, 39, 0.636, 59)
]

# Sigma = 6
data_sigma_6 = [
    (40, 0.751, 28, 0.520, 58), (42, 0.850, 31, 0.559, 58), (45, 0.809, 34, 0.576, 58), 
    (36, 0.650, 35, 0.687, 58), (37, 0.631, 30, 0.473, 58), (39, 0.727, 36, 0.630, 58), 
    (41, 0.710, 28, 0.416, 58), (41, 0.716, 30, 0.499, 58), (44, 0.671, 39, 0.657, 59), 
    (40, 0.789, 39, 0.636, 59)
]

# Sigma = 10
data_sigma_10 = [
    (40, 0.763, 28, 0.520, 58), (42, 0.855, 31, 0.559, 58), (46, 0.816, 34, 0.576, 58), 
    (34, 0.620, 35, 0.687, 58), (37, 0.635, 30, 0.473, 58), (40, 0.710, 36, 0.630, 58), 
    (42, 0.755, 28, 0.416, 58), (41, 0.730, 30, 0.499, 58), (44, 0.668, 39, 0.657, 59), 
    (40, 0.790, 39, 0.636, 59)
]

def get_scores(data):
    return [run[0]/run[4] for run in data], [run[1] for run in data]

# Extraction
acc_base, auc_base = get_scores(data_no_age)
acc_2, auc_2 = get_scores(data_sigma_2)
acc_4, auc_4 = get_scores(data_sigma_4)
acc_6, auc_6 = get_scores(data_sigma_6)
acc_10, auc_10 = get_scores(data_sigma_10)

data_acc = [acc_base, acc_2, acc_4, acc_6, acc_10]
data_auc = [auc_base, auc_2, auc_4, auc_6, auc_10]
labels = ['Baseline\n(No Age)', '$\sigma=2$', '$\sigma=4$', '$\sigma=6$', '$\sigma=10$']

# --- PLOT ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_box(ax, data, title, ylabel):
    bplot = ax.boxplot(data, patch_artist=True, labels=labels, showmeans=True)
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Couleur différente pour la Baseline
    colors = ['lightgray', 'lightblue', 'lightblue', 'lightblue', 'lightblue']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Légende
    green_triangle = mlines.Line2D([], [], color='green', marker='^', linestyle='None', 
                                   markersize=8, label='Mean')
    orange_line = mlines.Line2D([], [], color='orange', linewidth=1, label='Median')
    ax.legend(handles=[green_triangle, orange_line], loc='lower right')

plot_box(ax1, data_acc, 'ADHD Accuracy (10 Folds)', 'Accuracy')
plot_box(ax2, data_auc, 'ADHD AUC Score', 'Area Under Curve (AUC)')

plt.tight_layout()
plt.show()