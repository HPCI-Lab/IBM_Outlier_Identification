import pandas as pd
import matplotlib.pyplot as plt

# Load your CSVs
same_df = pd.read_csv('./MIA/results/pretrain_rotten_tomatoes_eval_rotten_tomatoes.csv')
diff_df = pd.read_csv('./MIA/results/pretrain_rotten_tomatoes_eval_imdb.csv')

same_df = same_df.iloc[3:]
diff_df = diff_df.iloc[3:]

# Common metrics
metrics = same_df['metric'].values

# Extract the values
auc_same = same_df['AUC'].values
auc_diff = diff_df['AUC'].values

tpr_same = same_df['TPR@0.1%%FPR'].values
tpr_diff = diff_df['TPR@0.1%%FPR'].values

acc_same = same_df['accuracy'].values
acc_diff = diff_df['accuracy'].values

# Set up plots
fig, axs = plt.subplots(3, 1, figsize=(18, 8), sharex=True)

# --- AUC Comparison ---
axs[0].plot(metrics, auc_same, marker='o', label='Same Dataset', color='blue')
axs[0].plot(metrics, auc_diff, marker='x', label='Different Dataset', color='red')
# axs[0].axhline(0.5, linestyle='--', color='gray', label='Random Guessing')
axs[0].set_ylabel('AUC')
axs[0].set_title('AUC Comparison by Metric')
axs[0].legend()
axs[0].grid(True)

# --- TPR@0.1%FPR Comparison ---
axs[1].plot(metrics, tpr_same, marker='o', label='Same Dataset', color='blue')
axs[1].plot(metrics, tpr_diff, marker='x', label='Different Dataset', color='red')
axs[1].set_ylabel('TPR @ 0.1% FPR')
axs[1].set_title('True Positive Rate @ 0.1% False Positive Rate')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(metrics, acc_same, marker='o', label='Same Dataset', color='blue')
axs[2].plot(metrics, acc_diff, marker='x', label='Different Dataset', color='red')
axs[2].set_ylabel('Accuracy by Metric')
axs[2].set_title('Accuracy')
axs[2].legend()
axs[2].grid(True)

# Rotate x labels for clarity
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("./MIA/results/pretrain_plot.png")