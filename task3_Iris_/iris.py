"""
=============================================================
  IRIS FLOWER 
  
=============================================================
"""

# ── Standard Library ──────────────────────────────────────
print("start")
import warnings
import os
warnings.filterwarnings("ignore")

# ── Data & Numerics ────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualization ──────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Scikit-learn ───────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, roc_auc_score
)
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm             import SVC
from sklearn.neighbors       import KNeighborsClassifier

# ══════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════
DATA_PATH    = "IRIS.csv"
OUTPUT_DIR   = "outputs"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

PALETTE = {
    "Iris-setosa":     "#2ECC71",
    "Iris-versicolor": "#3498DB",
    "Iris-virginica":  "#E74C3C",
}
sns.set_theme(style="whitegrid", font_scale=1.05)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════
# 1. DATA LOADING & EXPLORATION
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  IRIS FLOWER CLASSIFICATION — ML PIPELINE")
print("=" * 60)

df = pd.read_csv("IRIS.csv")

print("\n── Dataset Overview ──")
print(f"  Rows × Cols  : {df.shape}")
print(f"  Missing vals : {df.isnull().sum().sum()}")
print(f"\n{df.head(5).to_string(index=False)}")

print("\n── Descriptive Statistics ──")
print(df.describe().round(3).to_string())

print("\n── Class Distribution ──")
print(df["species"].value_counts().to_string())

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET   = "species"

X = df[FEATURES].values
y = df[TARGET].values

# Label-encode for AUC / some classifiers
le = LabelEncoder()
y_enc = le.fit_transform(y)          # 0 / 1 / 2

# ══════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════
print("\n[INFO] Generating EDA figures …")

# ── 2a. Pair Plot ─────────────────────────────────────────
pairfig = sns.pairplot(
    df, hue="species", palette=PALETTE,
    diag_kind="kde", plot_kws={"alpha": 0.7, "s": 50},
    diag_kws={"fill": True, "alpha": 0.4}
)
pairfig.fig.suptitle("Pair Plot — Iris Features by Species",
                     y=1.02, fontsize=14, fontweight="bold")
pairfig.savefig(f"{OUTPUT_DIR}/01_pair_plot.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 2b. Correlation Heat-map ──────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
corr = df[FEATURES].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="coolwarm",
    linewidths=0.5, square=True, ax=ax,
    vmin=-1, vmax=1,
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 2c. Box Plots (4 features) ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Feature Distribution by Species", fontsize=14,
             fontweight="bold", y=1.01)
for ax, feat in zip(axes.flat, FEATURES):
    sns.boxplot(data=df, x="species", y=feat, palette=PALETTE,
                width=0.5, flierprops={"marker": "o", "markersize": 4}, ax=ax)
    ax.set_title(feat.replace("_", " ").title(), fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=9)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/03_boxplots.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 2d. Violin Plots ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Feature Density by Species (Violin)", fontsize=14,
             fontweight="bold", y=1.01)
for ax, feat in zip(axes.flat, FEATURES):
    sns.violinplot(data=df, x="species", y=feat, palette=PALETTE,
                   inner="quartile", cut=0, ax=ax)
    ax.set_title(feat.replace("_", " ").title(), fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=9)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/04_violin_plots.png", dpi=150, bbox_inches="tight")
plt.close("all")

print("  ✔️  EDA figures saved.")

# ══════════════════════════════════════════════════════════
# 3. DATA SPLITTING
# ══════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE,
    stratify=y_enc, random_state=RANDOM_STATE
)
print(f"\n── Train / Test Split ({int((1-TEST_SIZE)*100)} / {int(TEST_SIZE*100)}) ──")
print(f"  Train : {X_train.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]} samples")

# ══════════════════════════════════════════════════════════
# 4. MODEL DEFINITIONS (with pipelines that include scaling)
# ══════════════════════════════════════════════════════════
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    KNeighborsClassifier(n_neighbors=5))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(n_estimators=200,
                                               learning_rate=0.05,
                                               random_state=RANDOM_STATE))
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
    ]),
}

# ══════════════════════════════════════════════════════════
# 5. CROSS-VALIDATED MODEL COMPARISON
# ══════════════════════════════════════════════════════════
print("\n── Cross-Validation Results ──")
print(f"  Strategy : {CV_FOLDS}-fold Stratified K-Fold\n")

cv_results = {}
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

header = f"  {'Model':<25} {'CV Mean':>9} {'CV Std':>8}"
print(header)
print("  " + "-" * 45)

for name, pipeline in models.items():
    scores = cross_val_score(pipeline, X_train, y_train,
                             cv=skf, scoring="accuracy")
    cv_results[name] = scores
    print(f"  {name:<25} {scores.mean():.4f}    ±{scores.std():.4f}")

# ══════════════════════════════════════════════════════════
# 6. HYPER-PARAMETER TUNING — Best two models
# ══════════════════════════════════════════════════════════
print("\n── Hyper-parameter Tuning ──")

# Random Forest grid
rf_param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth":    [None, 5, 10],
    "clf__min_samples_split": [2, 5],
}
rf_gs = GridSearchCV(models["Random Forest"], rf_param_grid,
                     cv=skf, scoring="accuracy", n_jobs=-1)
rf_gs.fit(X_train, y_train)
print(f"  Random Forest  best params : {rf_gs.best_params_}")
print(f"  Random Forest  best CV acc : {rf_gs.best_score_:.4f}")

# SVM grid
svm_param_grid = {
    "clf__C":     [0.1, 1, 10, 100],
    "clf__gamma": ["scale", "auto", 0.1, 0.01],
}
svm_gs = GridSearchCV(models["Support Vector Machine"], svm_param_grid,
                      cv=skf, scoring="accuracy", n_jobs=-1)
svm_gs.fit(X_train, y_train)
print(f"  SVM            best params : {svm_gs.best_params_}")
print(f"  SVM            best CV acc : {svm_gs.best_score_:.4f}")

# Replace with tuned versions
models["Random Forest"]       = rf_gs.best_estimator_
models["Support Vector Machine"] = svm_gs.best_estimator_

# ══════════════════════════════════════════════════════════
# 7. EVALUATE ALL MODELS ON HELD-OUT TEST SET
# ══════════════════════════════════════════════════════════
print("\n── Test-Set Evaluation ──\n")

results_records = []

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred      = pipeline.predict(X_test)
    y_prob      = pipeline.predict_proba(X_test)
    test_acc    = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    results_records.append({
        "Model":       name,
        "Test Acc":    round(test_acc, 4),
        "ROC-AUC":     round(auc, 4),
        "CV Mean":     round(cv_results[name].mean(), 4),
        "CV Std":      round(cv_results[name].std(), 4),
    })

results_df = pd.DataFrame(results_records).sort_values("Test Acc", ascending=False)
print(results_df.to_string(index=False))

# ══════════════════════════════════════════════════════════
# 8. BEST MODEL — DETAILED REPORT
# ══════════════════════════════════════════════════════════
best_name = results_df.iloc[0]["Model"]
best_model = models[best_name]
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

print(f"\n── Best Model : {best_name} ──")
print(f"\n{classification_report(y_test, y_pred_best, target_names=le.classes_)}")

# ══════════════════════════════════════════════════════════
# 9. VISUALIZATION — RESULTS
# ══════════════════════════════════════════════════════════
print("[INFO] Generating result figures …")

# ── 9a. Model comparison bar chart ───────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
bar_colors = ["#2ECC71" if i == 0 else "#3498DB"
              for i in range(len(results_df))]
bars = ax.barh(results_df["Model"], results_df["Test Acc"],
               color=bar_colors, edgecolor="white", height=0.55)
ax.errorbar(
    results_df["CV Mean"], results_df["Model"],
    xerr=results_df["CV Std"],
    fmt="D", color="#E74C3C", capsize=4, markersize=6,
    label="CV Mean ± Std", zorder=5
)
for bar, acc in zip(bars, results_df["Test Acc"]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{acc:.4f}", va="center", fontsize=9)
ax.set_xlim(0.85, 1.05)
ax.set_xlabel("Accuracy", fontsize=11)
ax.set_title("Model Comparison — Test Accuracy vs CV Mean", fontsize=13,
             fontweight="bold")
ax.legend(fontsize=9)
ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/05_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 9b. Confusion matrix — best model ────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/06_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 9c. CV score distribution (box plot) ─────────────────
fig, ax = plt.subplots(figsize=(10, 5))
cv_data  = [cv_results[m] for m in results_df["Model"]]
bp = ax.boxplot(cv_data, vert=True, patch_artist=True,
                labels=results_df["Model"],
                medianprops={"color": "#E74C3C", "linewidth": 2})
colors = sns.color_palette("Blues", len(cv_data))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
ax.set_ylabel("CV Accuracy", fontsize=11)
ax.set_title(f"Cross-Validation Score Distribution ({CV_FOLDS}-Fold)",
             fontsize=13, fontweight="bold")
ax.tick_params(axis="x", labelrotation=15)
ax.set_ylim(0.85, 1.05)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/07_cv_distribution.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 9d. Feature Importances (Random Forest) ──────────────
rf_pipeline = models["Random Forest"]
importances = rf_pipeline.named_steps["clf"].feature_importances_
feat_df = pd.DataFrame({
    "Feature":    FEATURES,
    "Importance": importances
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(feat_df["Feature"], feat_df["Importance"],
               color=["#2ECC71" if v > 0.2 else "#3498DB"
                      for v in feat_df["Importance"]],
               edgecolor="white")
for bar, val in zip(bars, feat_df["Importance"]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_xlabel("Importance Score", fontsize=11)
ax.set_title("Feature Importances — Random Forest", fontsize=13,
             fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/08_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ── 9e. Dashboard summary ─────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Iris Classification — Results Dashboard",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1 — Model Accuracy
ax1 = fig.add_subplot(gs[0, :2])
bar_colors = ["#2ECC71" if i == 0 else "#95A5A6" for i in range(len(results_df))]
ax1.barh(results_df["Model"], results_df["Test Acc"],
         color=bar_colors, edgecolor="white", height=0.55)
ax1.set_xlim(0.85, 1.02)
ax1.set_xlabel("Test Accuracy")
ax1.set_title("Test Accuracy by Model", fontweight="bold")
ax1.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)

# Panel 2 — Confusion Matrix
ax2 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Set", "Ver", "Vir"],
            yticklabels=["Set", "Ver", "Vir"],
            ax=ax2, cbar=False, linewidths=0.5)
ax2.set_title(f"Confusion Matrix\n({best_name})", fontweight="bold")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

# Panel 3 — Feature Importance
ax3 = fig.add_subplot(gs[1, 0])
ax3.barh(feat_df["Feature"], feat_df["Importance"],
         color="#3498DB", edgecolor="white")
ax3.set_xlabel("Importance")
ax3.set_title("Feature Importances\n(Random Forest)", fontweight="bold")

# Panel 4 — Petal scatter
ax4 = fig.add_subplot(gs[1, 1])
for sp, grp in df.groupby("species"):
    ax4.scatter(grp["petal_length"], grp["petal_width"],
                label=sp, color=PALETTE[sp], alpha=0.75, s=40, edgecolors="white")
ax4.set_xlabel("Petal Length")
ax4.set_ylabel("Petal Width")
ax4.set_title("Petal Dimensions\nby Species", fontweight="bold")
ax4.legend(fontsize=7)

# Panel 5 — Sepal scatter
ax5 = fig.add_subplot(gs[1, 2])
for sp, grp in df.groupby("species"):
    ax5.scatter(grp["sepal_length"], grp["sepal_width"],
                label=sp, color=PALETTE[sp], alpha=0.75, s=40, edgecolors="white")
ax5.set_xlabel("Sepal Length")
ax5.set_ylabel("Sepal Width")
ax5.set_title("Sepal Dimensions\nby Species", fontweight="bold")
ax5.legend(fontsize=7)

fig.savefig(f"{OUTPUT_DIR}/09_dashboard.png", dpi=150, bbox_inches="tight")
plt.close("all")

print("  ✔️  Result figures saved.")

# ══════════════════════════════════════════════════════════
# 10. FINAL SUMMARY
# ══════════════════════════════════════════════════════════
best_row = results_df.iloc[0]
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Best Model    : {best_row['Model']}")
print(f"  Test Accuracy : {best_row['Test Acc'] * 100:.2f}%")
print(f"  ROC-AUC Score : {best_row['ROC-AUC']:.4f}")
print(f"  CV Mean Acc   : {best_row['CV Mean'] * 100:.2f}% ± {best_row['CV Std'] * 100:.2f}%")
print("\n  Output files:")
output_files = [
    "01_pair_plot.png", "02_correlation_heatmap.png",
    "03_boxplots.png",  "04_violin_plots.png",
    "05_model_comparison.png", "06_confusion_matrix.png",
    "07_cv_distribution.png",  "08_feature_importance.png",
    "09_dashboard.png",
]
for f in output_files:
    print(f"    • {f}")
print("=" * 60)
print("\n  Pipeline complete.\n")

print("end")