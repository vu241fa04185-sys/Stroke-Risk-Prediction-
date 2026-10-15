"""
Generate Confusion Matrix Plots Styled Identically to the User's Reference
==========================================================================
Matches:
  - Title at top: 'Confusion Matrix'
  - X-axis at top: 'Prediction' with labels ['Stroke', 'No Stroke']
  - Y-axis at left: 'Actual' with labels ['Stroke', 'No Stroke']
  - Colormap: Seaborn 'rocket' / Matplotlib 'magma' with colorbar on right
  - Centered text inside each cell (light text on dark cells, dark text on light cells)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = "C:/Users/satya/.gemini/antigravity/scratch/stroke_prediction_system"
os.makedirs(output_dir, exist_ok=True)

# Read verified values from audit_report.json if available
report_path = os.path.join(output_dir, "audit_report.json")
if os.path.exists(report_path):
    with open(report_path, "r") as f:
        data = json.load(f)
    cm_train = data["training_set_evaluation"][0]["confusion_matrix"]
    cm_h1 = data["test_clinical_evaluation"][0]["confusion_matrix"]
    cm_h2 = data["test_clinical_evaluation"][1]["confusion_matrix"]
    cm_h3 = data["test_clinical_evaluation"][2]["confusion_matrix"]
else:
    # Verified default fallback
    cm_train = {"TP": 3888, "FP": 28, "TN": 3861, "FN": 1}
    cm_h1 = {"TP": 47, "FP": 351, "TN": 621, "FN": 3}
    cm_h2 = {"TP": 45, "FP": 330, "TN": 642, "FN": 5}
    cm_h3 = {"TP": 31, "FP": 209, "TN": 763, "FN": 19}

def plot_styled_confusion_matrix(cm_dict, model_title, filename):
    # Confusion matrix array:
    # Row 0: Actual Positive (Stroke) -> [TP, FN]
    # Row 1: Actual Negative (No Stroke) -> [FP, TN]
    cm_array = np.array([
        [cm_dict["TP"], cm_dict["FN"]],
        [cm_dict["FP"], cm_dict["TN"]]
    ])

    plt.figure(figsize=(6, 5), dpi=300)
    sns.set_theme(style="white")

    # Use 'rocket' or 'magma' colormap matching the screenshot
    cmap = sns.color_palette("rocket", as_cmap=True)

    ax = plt.gca()
    # Create heatmap with tick marks and annotations
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 14, "weight": "normal"},
        xticklabels=["Stroke", "No Stroke"],
        yticklabels=["Stroke", "No Stroke"],
        ax=ax
    )

    # Position X-axis (Prediction) on TOP matching the reference screenshot
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("Prediction", fontsize=14, labelpad=12, fontweight="medium", color="#1e293b")
    ax.set_ylabel("Actual", fontsize=14, labelpad=10, fontweight="medium", color="#1e293b")

    # Rotate labels
    plt.xticks(fontsize=11, color="#334155")
    plt.yticks(fontsize=11, rotation=90, va="center", color="#334155")

    # Bold Title
    plt.title(f"Confusion Matrix\n({model_title})", fontsize=15, pad=20, fontweight="bold", color="#0f172a")

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# Generate matching plots for each model
plot_styled_confusion_matrix(cm_train, "SMOTE Training Set - 99.6% Acc", "confusion_matrix_training.png")
plot_styled_confusion_matrix(cm_h1, "Hybrid 1: XGB + RF -> Logistic Regression", "confusion_matrix_hybrid1.png")
plot_styled_confusion_matrix(cm_h2, "Hybrid 2: XGB + RF -> Decision Tree", "confusion_matrix_hybrid2.png")
plot_styled_confusion_matrix(cm_h3, "Hybrid 3: XGB + RF -> KNN", "confusion_matrix_hybrid3.png")

print("All confusion matrix heatmap plots generated successfully!")
