# Group 13 - AI3013 Machine Learning Course Project
# Telecom Customer Churn Prediction Using Machine Learning Models
#
# This submission implements the machine learning workflow from scratch.
# Models included:
#   1. Logistic Regression
#   2. Cost-Sensitive Logistic Regression
#   3. Gaussian Naive Bayes
#   4. Hybrid Naive Bayes
#   5. Decision Tree
#
# Important: This code does NOT use scikit-learn, TensorFlow, Keras, or any
# prebuilt machine learning model library. It uses only NumPy, Pandas, and
# Matplotlib for numerical computation, data handling, and visualization.
#
# To run:
#   cd Code
#   python Group13_Telco_Churn_From_Scratch.py
#
# Required dataset file in the same folder:
#   WA_Fn-UseC_-Telco-Customer-Churn.csv
# %%
# %% [markdown]
# # Telco Customer Churn Prediction
#
# This notebook studies a real-world telecom churn prediction problem using
# machine learning models implemented from scratch.
#
# Models compared:
# - Logistic Regression
# - Cost-Sensitive Logistic Regression
# - Gaussian Naive Bayes
# - Hybrid Naive Bayes
# - Decision Tree
#
# Included:
# - exploratory data analysis
# - preprocessing
# - stratified k-fold cross validation
# - model comparison
# - threshold analysis
# - ROC / PR visualization
# - interpretability analysis
# - hyperparameter sensitivity analysis
# - learning curve analysis
# - final conclusion

# %%
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython.display import display, Markdown
except ImportError:
    display = None
    Markdown = None

# -----------------------------
# Global plotting configuration
# -----------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})

pd.set_option("display.max_columns", 200)
np.set_printoptions(suppress=True, precision=4)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path.cwd()
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42

print("Working directory:", BASE_DIR)
print("Dataset path:", DATA_PATH)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        f"Please put WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder as this notebook."
    )

# -----------------------------
# Consistent model ordering/colors
# -----------------------------
MODEL_ORDER = [
    "Logistic Regression",
    "Cost-Sensitive Logistic Regression",
    "Gaussian Naive Bayes",
    "Hybrid Naive Bayes",
    "Decision Tree",
]

MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Cost-Sensitive Logistic Regression": "#ff7f0e",
    "Gaussian Naive Bayes": "#2ca02c",
    "Hybrid Naive Bayes": "#9467bd",
    "Decision Tree": "#d62728",
}

def existing_model_order(names):
    return [m for m in MODEL_ORDER if m in list(names)]

def safe_name(s):
    return s.lower().replace(" ", "_").replace("/", "_").replace("-", "_")

def save_plot(filename):
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")

# %% [markdown]
# ## 1. Load Raw Dataset

# %%
def load_raw_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

raw_df = load_raw_data()
print("Raw shape:", raw_df.shape)
raw_df.head()

# %% [markdown]
# ## 2. Exploratory Data Analysis

# %%
print("Data types:")
print(raw_df.dtypes)

print("\nMissing values:")
print(raw_df.isna().sum().sort_values(ascending=False).head(15))

print("\nChurn distribution:")
print(raw_df["Churn"].value_counts())

print("\nChurn ratio:")
print(raw_df["Churn"].value_counts(normalize=True).round(4))

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

raw_df["Churn"].value_counts().plot(
    kind="bar", ax=axes[0, 0], title="Churn distribution", color=["#4c72b0", "#dd8452"]
)
axes[0, 0].set_ylabel("Count")
axes[0, 0].tick_params(axis="x", rotation=0)

raw_df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values().plot(
    kind="bar", ax=axes[0, 1], title="Churn rate by contract", color="#4c72b0"
)
axes[0, 1].set_ylabel("Churn rate")
axes[0, 1].tick_params(axis="x", rotation=15)

raw_df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values().plot(
    kind="bar", ax=axes[1, 0], title="Churn rate by internet service", color="#55a868"
)
axes[1, 0].set_ylabel("Churn rate")
axes[1, 0].tick_params(axis="x", rotation=15)

raw_df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values().plot(
    kind="bar", ax=axes[1, 1], title="Churn rate by payment method", color="#c44e52"
)
axes[1, 1].set_ylabel("Churn rate")
axes[1, 1].tick_params(axis="x", rotation=25)

plt.tight_layout()
save_plot("eda_overview.png")
plt.show()

# %%
numeric_cols_for_eda = ["tenure", "MonthlyCharges", "TotalCharges"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col in zip(axes, numeric_cols_for_eda):
    ax.hist(raw_df[col].dropna(), bins=30, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

plt.tight_layout()
save_plot("numeric_distributions.png")
plt.show()

# %% [markdown]
# ## 3. Data Preprocessing
#
# Important note:
# We do not standardize the full dataset before cross validation, because that would introduce
# data leakage. Standardization is done inside each fold using training data only.

# %%
def build_design_matrix(df):
    df = df.copy()
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    y = (df["Churn"] == "Yes").astype(int).to_numpy()

    X_df = df.drop(columns=["customerID", "Churn"])

    # Continuous features
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    # Treat everything else as categorical / binary-capable
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    X_df = pd.get_dummies(X_df, columns=categorical_cols, drop_first=False)

    feature_names = X_df.columns.tolist()
    numeric_idx = [feature_names.index(col) for col in numeric_cols]
    binary_idx = [i for i, c in enumerate(feature_names) if c not in numeric_cols]

    X = X_df.astype(float).to_numpy()
    return X, y, feature_names, numeric_cols, numeric_idx, binary_idx, df, X_df

X_all, y_all, feature_names, numeric_cols, numeric_idx, binary_idx, cleaned_df, X_df = build_design_matrix(raw_df)

print(f"Samples after cleaning: {len(y_all)}")
print(f"Features after encoding: {X_all.shape[1]}")
print(f"Churn rate: {y_all.mean():.4f}")
X_df.head()

# %%
def standardize_train_val(X_train, X_val, numeric_idx):
    X_train = X_train.copy()
    X_val = X_val.copy()

    means = X_train[:, numeric_idx].mean(axis=0)
    stds = X_train[:, numeric_idx].std(axis=0)
    stds[stds == 0] = 1.0

    X_train[:, numeric_idx] = (X_train[:, numeric_idx] - means) / stds
    X_val[:, numeric_idx] = (X_val[:, numeric_idx] - means) / stds

    return X_train, X_val, means, stds

# %% [markdown]
# ## 4. Evaluation Utilities

# %%
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def stratified_kfold_indices(y, k=5, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)

    folds = []
    all_idx = np.arange(len(y))
    for i in range(k):
        val_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        rng.shuffle(val_idx)
        train_idx = np.setdiff1d(all_idx, val_idx)
        folds.append((train_idx, val_idx))
    return folds

def confusion_counts(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn

def roc_curve_points(y_true, y_prob):
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    P = y_true.sum()
    N = len(y_true) - P
    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tps, fps = 0, 0
    fprs, tprs = [0.0], [0.0]

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tps += 1
        else:
            fps += 1

        if i == len(y_true) - 1 or y_prob[i] != y_prob[i + 1]:
            fprs.append(fps / N)
            tprs.append(tps / P)

    return np.array(fprs), np.array(tprs)

def precision_recall_curve_points(y_true, y_prob):
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    P = y_true.sum()
    if P == 0:
        return np.array([0.0]), np.array([0.0])

    tp, fp = 0, 0
    recalls, precisions = [0.0], [1.0]

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

        if i == len(y_true) - 1 or y_prob[i] != y_prob[i + 1]:
            recall = tp / P
            precision = tp / (tp + fp) if (tp + fp) else 1.0
            recalls.append(recall)
            precisions.append(precision)

    return np.array(recalls), np.array(precisions)

def auc_trapezoid(x, y):
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    fprs, tprs = roc_curve_points(y_true, y_prob)
    roc_auc = auc_trapezoid(fprs, tprs)

    recalls, precisions = precision_recall_curve_points(y_true, y_prob)
    pr_auc = auc_trapezoid(recalls, precisions)

    eps = 1e-12
    clipped = np.clip(y_prob, eps, 1 - eps)
    logloss = -np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped))
    brier = np.mean((y_prob - y_true) ** 2)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "logloss": logloss,
        "brier": brier,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def plot_confusion_matrix_normalized(y_true, y_prob, title, threshold=0.5, filename=None):
    y_pred = (y_prob >= threshold).astype(int)
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                f"{int(cm[i, j])}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center", color="black", fontsize=11
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted No", "Predicted Yes"])
    ax.set_yticklabels(["Actual No", "Actual Yes"])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if filename is not None:
        save_plot(filename)
    plt.show()

def plot_cv_bar_with_error(result_df, metrics, title, ylabel, filename, ylim=None):
    order = existing_model_order(result_df["model"].unique())
    means = result_df.groupby("model")[metrics].mean().loc[order]
    stds = result_df.groupby("model")[metrics].std().loc[order]

    x = np.arange(len(order))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, metric in enumerate(metrics):
        ax.bar(
            x + i * width,
            means[metric],
            width=width,
            yerr=stds[metric],
            capsize=4,
            label=metric,
            alpha=0.9
        )

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(order, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    save_plot(filename)
    plt.show()

def plot_single_metric_bar(series, title, ylabel, filename, color="#4c72b0"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series.plot(kind="bar", color=color, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    save_plot(filename)
    plt.show()

# %% [markdown]
# ## 5. Theoretical Background
#
# Logistic Regression:
#   P(y=1|x) = sigmoid(w^T x + b)
#   Optimized with regularized binary cross entropy.
#
# Cost-Sensitive Logistic Regression:
#   Same probabilistic model as Logistic Regression, but with weighted loss
#   to better handle class imbalance.
#
# Gaussian Naive Bayes:
#   Assumes conditional independence and Gaussian feature likelihood.
#
# Hybrid Naive Bayes:
#   Gaussian for continuous features + Bernoulli for one-hot binary features.
#
# Decision Tree:
#   Recursive binary splitting using Gini impurity reduction.

# %% [markdown]
# ## 6. Machine Learning Models Implemented From Scratch

# %%
class LogisticRegressionScratch:
    def __init__(self, lr=0.05, epochs=2500, l2=0.02):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _loss(self, X, y):
        p = np.clip(sigmoid(X @ self.w + self.b), 1e-12, 1 - 1e-12)
        ce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = 0.5 * self.l2 * np.sum(self.w ** 2)
        return ce + reg

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []

        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            error = p - y

            grad_w = (X.T @ error) / n + self.l2 * self.w
            grad_b = error.mean()

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            self.loss_history.append(self._loss(X, y))
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def num_parameters(self):
        return 0 if self.w is None else len(self.w) + 1


class CostSensitiveLogisticRegressionScratch:
    def __init__(self, lr=0.05, epochs=2500, l2=0.02, pos_weight=2.0, neg_weight=1.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.w = None
        self.b = 0.0
        self.loss_history = []

    def _sample_weights(self, y):
        return np.where(y == 1, self.pos_weight, self.neg_weight)

    def _loss(self, X, y):
        p = np.clip(sigmoid(X @ self.w + self.b), 1e-12, 1 - 1e-12)
        weights = self._sample_weights(y)
        ce = -np.sum(weights * (y * np.log(p) + (1 - y) * np.log(1 - p))) / weights.sum()
        reg = 0.5 * self.l2 * np.sum(self.w ** 2)
        return ce + reg

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []

        weights = self._sample_weights(y)

        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            error = (p - y) * weights

            grad_w = (X.T @ error) / weights.sum() + self.l2 * self.w
            grad_b = error.sum() / weights.sum()

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            self.loss_history.append(self._loss(X, y))

        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def num_parameters(self):
        return 0 if self.w is None else len(self.w) + 1


class GaussianNaiveBayesScratch:
    def __init__(self, var_smoothing=1e-6):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.array([0, 1])
        self.priors = np.array([(y == c).mean() for c in self.classes])
        self.means = np.vstack([X[y == c].mean(axis=0) for c in self.classes])
        self.vars = np.vstack([X[y == c].var(axis=0) + self.var_smoothing for c in self.classes])
        return self

    def predict_proba(self, X):
        log_probs = []
        for i, c in enumerate(self.classes):
            log_prior = np.log(self.priors[i] + 1e-12)
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.vars[i]) + ((X - self.means[i]) ** 2) / self.vars[i],
                axis=1
            )
            log_probs.append(log_prior + log_likelihood)

        log_probs = np.vstack(log_probs).T
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1]

    def num_parameters(self):
        if not hasattr(self, "means"):
            return 0
        return self.means.size + self.vars.size + len(self.priors)


class HybridNaiveBayesScratch:
    """
    Gaussian for continuous features, Bernoulli for binary one-hot features.
    """
    def __init__(self, numeric_idx, binary_idx, var_smoothing=1e-6, alpha=1.0):
        self.numeric_idx = np.array(numeric_idx, dtype=int)
        self.binary_idx = np.array(binary_idx, dtype=int)
        self.var_smoothing = var_smoothing
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.array([0, 1])
        self.priors = np.array([(y == c).mean() for c in self.classes])

        # Gaussian parameters
        self.means_num = np.vstack([
            X[y == c][:, self.numeric_idx].mean(axis=0)
            for c in self.classes
        ])
        self.vars_num = np.vstack([
            X[y == c][:, self.numeric_idx].var(axis=0) + self.var_smoothing
            for c in self.classes
        ])

        # Bernoulli parameters with Laplace smoothing
        self.probs_bin = np.vstack([
            (X[y == c][:, self.binary_idx].sum(axis=0) + self.alpha) /
            (len(X[y == c]) + 2 * self.alpha)
            for c in self.classes
        ])

        return self

    def predict_proba(self, X):
        X_num = X[:, self.numeric_idx]
        X_bin = X[:, self.binary_idx]

        log_probs = []
        for i, c in enumerate(self.classes):
            log_prior = np.log(self.priors[i] + 1e-12)

            mu = self.means_num[i]
            var = self.vars_num[i]
            log_gauss = -0.5 * np.sum(
                np.log(2.0 * np.pi * var) + ((X_num - mu) ** 2) / var,
                axis=1
            )

            p = np.clip(self.probs_bin[i], 1e-12, 1 - 1e-12)
            log_bern = np.sum(
                X_bin * np.log(p) + (1 - X_bin) * np.log(1 - p),
                axis=1
            )

            log_probs.append(log_prior + log_gauss + log_bern)

        log_probs = np.vstack(log_probs).T
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1]

    def num_parameters(self):
        if not hasattr(self, "means_num"):
            return 0
        return (
            self.means_num.size +
            self.vars_num.size +
            self.probs_bin.size +
            len(self.priors)
        )


class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=30, min_samples_leaf=10, max_thresholds=25):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.root = None
        self.feature_importances_ = None
        self.n_nodes_ = 0

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 1.0 - p * p - (1.0 - p) * (1.0 - p)

    def _best_split(self, X, y):
        n, d = X.shape
        parent_impurity = self._gini(y)

        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for j in range(d):
            vals = np.unique(X[:, j])
            if len(vals) <= 1:
                continue

            if len(vals) > self.max_thresholds:
                qs = np.linspace(0.05, 0.95, self.max_thresholds)
                thresholds = np.unique(np.quantile(vals, qs))
            else:
                thresholds = (vals[:-1] + vals[1:]) / 2.0

            for t in thresholds:
                left = X[:, j] <= t
                right = ~left

                if left.sum() < self.min_samples_leaf or right.sum() < self.min_samples_leaf:
                    continue

                weighted_impurity = (
                    left.mean() * self._gini(y[left]) +
                    right.mean() * self._gini(y[right])
                )
                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = j
                    best_threshold = t

        return best_feature, best_threshold, best_gain

    def _build(self, X, y, depth):
        self.n_nodes_ += 1
        node = {
            "prob": float(y.mean()),
            "feature": None,
            "threshold": None,
            "left": None,
            "right": None
        }

        if (
            depth >= self.max_depth or
            len(y) < self.min_samples_split or
            y.mean() == 0.0 or
            y.mean() == 1.0
        ):
            return node

        feature, threshold, gain = self._best_split(X, y)
        if feature is None:
            return node

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        self.feature_importances_[feature] += gain * len(y)

        node["feature"] = feature
        node["threshold"] = threshold
        node["left"] = self._build(X[left_mask], y[left_mask], depth + 1)
        node["right"] = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.n_nodes_ = 0
        self.root = self._build(X, y, 0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def _predict_one(self, x):
        node = self.root
        while node["feature"] is not None:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prob"]

    def predict_proba(self, X):
        return np.array([self._predict_one(x) for x in X])

    def num_parameters(self):
        return self.n_nodes_

# %% [markdown]
# ## 7. Cross Validation and Model Comparison

# %%
POS_WEIGHT_BALANCED = len(y_all[y_all == 0]) / len(y_all[y_all == 1])

model_specs = {
    "Logistic Regression": {
        "factory": lambda: LogisticRegressionScratch(lr=0.05, epochs=2500, l2=0.02),
        "needs_scaling": True,
    },
    "Cost-Sensitive Logistic Regression": {
        "factory": lambda: CostSensitiveLogisticRegressionScratch(
            lr=0.05,
            epochs=2500,
            l2=0.02,
            pos_weight=POS_WEIGHT_BALANCED,
            neg_weight=1.0,
        ),
        "needs_scaling": True,
    },
    "Gaussian Naive Bayes": {
        "factory": lambda: GaussianNaiveBayesScratch(var_smoothing=1e-6),
        "needs_scaling": True,
    },
    "Hybrid Naive Bayes": {
        "factory": lambda: HybridNaiveBayesScratch(
            numeric_idx=numeric_idx,
            binary_idx=binary_idx,
            var_smoothing=1e-6,
            alpha=1.0
        ),
        "needs_scaling": True,
    },
    "Decision Tree": {
        "factory": lambda: DecisionTreeScratch(
            max_depth=5,
            min_samples_split=30,
            min_samples_leaf=10,
            max_thresholds=25
        ),
        "needs_scaling": False,
    },
}

# %%
def cross_validate_models(model_specs, X, y, numeric_idx, k=5, threshold=0.5):
    folds = stratified_kfold_indices(y, k=k, seed=RANDOM_SEED)
    rows = []
    oof_probs = {name: np.zeros(len(y), dtype=float) for name in model_specs}

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        for name, spec in model_specs.items():
            model = spec["factory"]()

            if spec["needs_scaling"]:
                X_train, X_val, _, _ = standardize_train_val(X_train_raw, X_val_raw, numeric_idx)
            else:
                X_train, X_val = X_train_raw.copy(), X_val_raw.copy()

            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start

            train_prob = model.predict_proba(X_train)
            val_prob = model.predict_proba(X_val)

            train_metrics = classification_metrics(y_train, train_prob, threshold=threshold)
            val_metrics = classification_metrics(y_val, val_prob, threshold=threshold)

            rows.append({
                "model": name,
                "fold": fold_id,
                "train_time_sec": train_time,
                "num_parameters": model.num_parameters(),
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_specificity": val_metrics["specificity"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_logloss": val_metrics["logloss"],
                "val_brier": val_metrics["brier"],
                "generalization_gap_f1": train_metrics["f1"] - val_metrics["f1"],
            })

            oof_probs[name][val_idx] = val_prob

    return pd.DataFrame(rows), oof_probs

# %%
result_df, oof_probs = cross_validate_models(model_specs, X_all, y_all, numeric_idx, k=5, threshold=0.5)
result_df.to_csv(OUTPUT_DIR / "cv_results_detailed.csv", index=False)
result_df.head()

# %%
summary_mean = result_df.groupby("model")[[
    "train_time_sec", "num_parameters",
    "train_f1", "val_accuracy", "val_precision", "val_recall",
    "val_specificity", "val_balanced_accuracy", "val_f1",
    "val_roc_auc", "val_pr_auc", "val_logloss", "val_brier",
    "generalization_gap_f1"
]].mean().loc[existing_model_order(result_df["model"].unique())]

summary_std = result_df.groupby("model")[[
    "val_accuracy", "val_precision", "val_recall",
    "val_f1", "val_roc_auc", "val_pr_auc", "val_logloss", "val_brier"
]].std().loc[existing_model_order(result_df["model"].unique())]

print("Cross-validation mean summary:")
display(summary_mean.round(4) if display else summary_mean.round(4))

print("\nCross-validation std summary:")
display(summary_std.round(4) if display else summary_std.round(4))

# %%
oof_rows = []
for name, probs in oof_probs.items():
    metrics = classification_metrics(y_all, probs, threshold=0.5)
    oof_rows.append({"model": name, **metrics})

oof_summary = pd.DataFrame(oof_rows).set_index("model").loc[existing_model_order(model_specs.keys())]
oof_summary.to_csv(OUTPUT_DIR / "oof_summary.csv")
print("OOF summary:")
display(oof_summary.round(4) if display else oof_summary.round(4))

# %% [markdown]
# ## 8. Visualization of Main Results

# %%
plot_cv_bar_with_error(
    result_df,
    metrics=["val_f1", "val_recall", "val_precision", "val_balanced_accuracy"],
    title="Cross-validation classification performance",
    ylabel="Score",
    filename="cv_classification_metrics.png",
    ylim=(0, 1)
)

# %%
plot_cv_bar_with_error(
    result_df,
    metrics=["val_roc_auc", "val_pr_auc"],
    title="Cross-validation ranking performance",
    ylabel="AUC",
    filename="cv_auc_metrics.png",
    ylim=(0, 1)
)

# %%
fig, ax = plt.subplots(figsize=(10, 4.5))
order = existing_model_order(oof_summary.index)
oof_summary.loc[order, ["accuracy", "f1", "roc_auc", "pr_auc"]].plot(kind="bar", ax=ax)
ax.set_title("Out-of-fold performance summary")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
save_plot("oof_performance_summary.png")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

summary_mean["train_time_sec"].loc[order].plot(kind="bar", ax=axes[0], color="#4c72b0")
axes[0].set_title("Average training time by model")
axes[0].set_ylabel("Seconds")
axes[0].tick_params(axis="x", rotation=15)

summary_mean["num_parameters"].loc[order].plot(kind="bar", ax=axes[1], color="#55a868")
axes[1].set_title("Model complexity proxy")
axes[1].set_ylabel("Count")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
save_plot("complexity_comparison.png")
plt.show()

# %% [markdown]
# ## 9. Confusion Matrices

# %%
for model_name, probs in oof_probs.items():
    plot_confusion_matrix_normalized(
        y_all, probs,
        title=f"{model_name} confusion matrix",
        threshold=0.5,
        filename=f"confusion_matrix_{safe_name(model_name)}.png"
    )

# %% [markdown]
# ## 10. ROC and Precision-Recall Curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for model_name in existing_model_order(oof_probs.keys()):
    probs = oof_probs[model_name]
    fpr, tpr = roc_curve_points(y_all, probs)
    roc_auc = auc_trapezoid(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})", color=MODEL_COLORS[model_name])

axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
axes[0].set_title("Receiver operating characteristic curves")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].legend(fontsize=8)

for model_name in existing_model_order(oof_probs.keys()):
    probs = oof_probs[model_name]
    recall, precision = precision_recall_curve_points(y_all, probs)
    pr_auc = auc_trapezoid(recall, precision)
    axes[1].plot(recall, precision, label=f"{model_name} (AUC={pr_auc:.3f})", color=MODEL_COLORS[model_name])

axes[1].set_title("Precision–recall curves")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend(fontsize=8)

plt.tight_layout()
save_plot("roc_pr_curves.png")
plt.show()

# %% [markdown]
# ## 11. Threshold Analysis

# %%
def threshold_sweep(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.20, 0.81, 0.05)

    rows = []
    for t in thresholds:
        m = classification_metrics(y_true, y_prob, threshold=t)
        rows.append({
            "threshold": t,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "specificity": m["specificity"],
            "balanced_accuracy": m["balanced_accuracy"],
        })
    return pd.DataFrame(rows)

# %%
logit_threshold_df = threshold_sweep(y_all, oof_probs["Logistic Regression"])
cost_logit_threshold_df = threshold_sweep(y_all, oof_probs["Cost-Sensitive Logistic Regression"])

logit_threshold_df.to_csv(OUTPUT_DIR / "threshold_analysis_logit_oof.csv", index=False)
cost_logit_threshold_df.to_csv(OUTPUT_DIR / "threshold_analysis_cost_logit_oof.csv", index=False)

print("Logistic Regression threshold sweep:")
display(logit_threshold_df.round(4) if display else logit_threshold_df.round(4))

print("\nCost-Sensitive Logistic Regression threshold sweep:")
display(cost_logit_threshold_df.round(4) if display else cost_logit_threshold_df.round(4))

# %%
best_threshold_row = logit_threshold_df.sort_values("f1", ascending=False).iloc[0]
best_cost_threshold_row = cost_logit_threshold_df.sort_values("f1", ascending=False).iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for col in ["precision", "recall", "f1", "balanced_accuracy"]:
    axes[0].plot(logit_threshold_df["threshold"], logit_threshold_df[col], marker="o", label=col)
axes[0].axvline(best_threshold_row["threshold"], color="black", linestyle="--",
                alpha=0.7, label=f"Best = {best_threshold_row['threshold']:.2f}")
axes[0].set_title("Threshold sensitivity: Logistic Regression")
axes[0].set_xlabel("Threshold")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=8)

for col in ["precision", "recall", "f1", "balanced_accuracy"]:
    axes[1].plot(cost_logit_threshold_df["threshold"], cost_logit_threshold_df[col], marker="o", label=col)
axes[1].axvline(best_cost_threshold_row["threshold"], color="black", linestyle="--",
                alpha=0.7, label=f"Best = {best_cost_threshold_row['threshold']:.2f}")
axes[1].set_title("Threshold sensitivity: Cost-Sensitive Logistic Regression")
axes[1].set_xlabel("Threshold")
axes[1].legend(fontsize=8)

plt.tight_layout()
save_plot("threshold_analysis_comparison.png")
plt.show()

print("Best threshold by F1 (Logistic Regression):")
print(best_threshold_row.round(4))

print("\nBest threshold by F1 (Cost-Sensitive Logistic Regression):")
print(best_cost_threshold_row.round(4))

# %% [markdown]
# ## 12. Interpretability Analysis

# %%
X_std_full, _, _, _ = standardize_train_val(X_all, X_all, numeric_idx)
logit_full = LogisticRegressionScratch(lr=0.05, epochs=2500, l2=0.02)
logit_full.fit(X_std_full, y_all)

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": logit_full.w
}).sort_values("coefficient")

top_negative = coef_df.head(10)
top_positive = coef_df.tail(10).sort_values("coefficient", ascending=False)

print("Top features decreasing churn:")
display(top_negative if display else top_negative)

print("Top features increasing churn:")
display(top_positive if display else top_positive)

# %%
top_features = pd.concat([top_negative, top_positive])

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["feature"],
    top_features["coefficient"],
    color=["#4c72b0" if x < 0 else "#c44e52" for x in top_features["coefficient"]]
)
plt.title("Top logistic regression coefficients")
plt.xlabel("Coefficient")
plt.tight_layout()
save_plot("logistic_top_coefficients.png")
plt.show()

# %%
tree_full = DecisionTreeScratch(max_depth=5, min_samples_split=30, min_samples_leaf=10, max_thresholds=25)
tree_full.fit(X_all, y_all)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": tree_full.feature_importances_
}).sort_values("importance", ascending=False).head(15)

print("Top Decision Tree Feature Importances:")
display(importance_df if display else importance_df)

# %%
plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#55a868")
plt.title("Top decision tree feature importances")
plt.xlabel("Importance")
plt.tight_layout()
save_plot("tree_feature_importance.png")
plt.show()

# %% [markdown]
# ## 13. Generalization Gap Analysis

# %%
gap_summary = result_df.groupby("model")[["train_f1", "val_f1", "generalization_gap_f1"]].mean()
gap_summary = gap_summary.loc[existing_model_order(gap_summary.index)]
display(gap_summary.round(4) if display else gap_summary.round(4))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

gap_summary[["train_f1", "val_f1"]].plot(kind="bar", figsize=(8, 5), ax=axes[0])
axes[0].set_title("Train vs validation F1")
axes[0].set_ylabel("F1 score")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=15)

gap_summary["generalization_gap_f1"].plot(kind="bar", ax=axes[1], color="orange")
axes[1].set_title("Generalization gap (train F1 - validation F1)")
axes[1].set_ylabel("Gap")
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
save_plot("generalization_gap_analysis.png")
plt.show()

# %% [markdown]
# ## 14. Complexity Discussion Table

# %%
complexity_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Cost-Sensitive Logistic Regression",
        "Gaussian Naive Bayes",
        "Hybrid Naive Bayes",
        "Decision Tree"
    ],
    "Training Complexity (rough)": [
        "O(n * d * epochs)",
        "O(n * d * epochs)",
        "O(n * d)",
        "O(n * d)",
        "O(d * n * T)  (T = tested thresholds)"
    ],
    "Inference Complexity per sample": [
        "O(d)",
        "O(d)",
        "O(d)",
        "O(d)",
        "O(tree depth)"
    ],
    "Main Strength": [
        "Interpretable linear baseline",
        "Class-imbalance aware baseline",
        "Very fast probabilistic baseline",
        "Mixed-type probabilistic baseline",
        "Captures nonlinear relations"
    ],
    "Main Weakness": [
        "Limited nonlinear expressiveness",
        "Requires weight tuning",
        "Strong independence + Gaussian assumption",
        "Still assumes independence",
        "May overfit if too deep"
    ]
})
display(complexity_df if display else complexity_df)


# %% [markdown]
# ## 15. Hyperparameter Sensitivity Analysis

# %%
def evaluate_single_model_cv(factory, needs_scaling, X, y, numeric_idx, k=5, threshold=0.5):
    folds = stratified_kfold_indices(y, k=k, seed=RANDOM_SEED)
    rows = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if needs_scaling:
            X_train, X_val, _, _ = standardize_train_val(X_train_raw, X_val_raw, numeric_idx)
        else:
            X_train, X_val = X_train_raw.copy(), X_val_raw.copy()

        model = factory()

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        train_prob = model.predict_proba(X_train)
        val_prob = model.predict_proba(X_val)

        train_metrics = classification_metrics(y_train, train_prob, threshold=threshold)
        val_metrics = classification_metrics(y_val, val_prob, threshold=threshold)

        rows.append({
            "fold": fold_id,
            "train_time_sec": train_time,

            "train_f1": train_metrics["f1"],
            "train_accuracy": train_metrics["accuracy"],

            "val_f1": val_metrics["f1"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_accuracy": val_metrics["accuracy"],
        })

    return pd.DataFrame(rows)


# %%
def hyperparameter_sensitivity(param_name, param_values, factory_builder, needs_scaling, X, y, numeric_idx, k=5, threshold=0.5):
    rows = []

    for value in param_values:
        cv_df = evaluate_single_model_cv(
            factory=lambda v=value: factory_builder(v),
            needs_scaling=needs_scaling,
            X=X,
            y=y,
            numeric_idx=numeric_idx,
            k=k,
            threshold=threshold
        )

        rows.append({
            param_name: value,

            "train_time_sec": cv_df["train_time_sec"].mean(),

            "train_f1": cv_df["train_f1"].mean(),
            "train_accuracy": cv_df["train_accuracy"].mean(),

            "val_f1": cv_df["val_f1"].mean(),
            "val_recall": cv_df["val_recall"].mean(),
            "val_precision": cv_df["val_precision"].mean(),
            "val_balanced_accuracy": cv_df["val_balanced_accuracy"].mean(),
            "val_roc_auc": cv_df["val_roc_auc"].mean(),
            "val_pr_auc": cv_df["val_pr_auc"].mean(),
            "val_accuracy": cv_df["val_accuracy"].mean(),

            "generalization_gap_f1": (cv_df["train_f1"] - cv_df["val_f1"]).mean()
        })

    return pd.DataFrame(rows)


# %% [markdown]
# ### 15.1 Logistic Regression sensitivity: L2 regularization

# %%
logit_l2_values = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1]

logit_sensitivity_df = hyperparameter_sensitivity(
    param_name="l2",
    param_values=logit_l2_values,
    factory_builder=lambda l2: LogisticRegressionScratch(lr=0.05, epochs=2500, l2=l2),
    needs_scaling=True,
    X=X_all,
    y=y_all,
    numeric_idx=numeric_idx,
    k=5,
    threshold=0.5
)

logit_sensitivity_df.to_csv(OUTPUT_DIR / "logit_hyperparameter_sensitivity.csv", index=False)

print("Logistic Regression sensitivity columns:")
print(logit_sensitivity_df.columns.tolist())
display(logit_sensitivity_df.round(4) if display else logit_sensitivity_df.round(4))


# %%
plt.figure(figsize=(10, 5))
plt.plot(logit_sensitivity_df["l2"], logit_sensitivity_df["train_f1"], marker="o", label="Train F1")
plt.plot(logit_sensitivity_df["l2"], logit_sensitivity_df["val_f1"], marker="o", label="Validation F1")
plt.plot(logit_sensitivity_df["l2"], logit_sensitivity_df["val_roc_auc"], marker="o", label="Validation ROC-AUC")
plt.title("Logistic regression hyperparameter sensitivity (L2)")
plt.xlabel("L2 regularization strength")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
save_plot("logit_hyperparameter_sensitivity.png")
plt.show()


# %% [markdown]
# ### 15.2 Cost-Sensitive Logistic Regression sensitivity: positive-class weight

# %%
cost_weight_values = [1.0, 1.5, 2.0, 2.5, 3.0, POS_WEIGHT_BALANCED]

cost_logit_sensitivity_df = hyperparameter_sensitivity(
    param_name="pos_weight",
    param_values=cost_weight_values,
    factory_builder=lambda w: CostSensitiveLogisticRegressionScratch(
        lr=0.05,
        epochs=2500,
        l2=0.02,
        pos_weight=w,
        neg_weight=1.0
    ),
    needs_scaling=True,
    X=X_all,
    y=y_all,
    numeric_idx=numeric_idx,
    k=5,
    threshold=0.5
)

cost_logit_sensitivity_df.to_csv(OUTPUT_DIR / "cost_logit_hyperparameter_sensitivity.csv", index=False)

print("Cost-Sensitive Logistic Regression sensitivity columns:")
print(cost_logit_sensitivity_df.columns.tolist())
display(cost_logit_sensitivity_df.round(4) if display else cost_logit_sensitivity_df.round(4))


# %%
required_cols = ["pos_weight", "train_f1", "val_f1", "val_recall", "val_roc_auc"]
missing = [c for c in required_cols if c not in cost_logit_sensitivity_df.columns]

if missing:
    print("Missing columns:", missing)
    print("Current columns:", cost_logit_sensitivity_df.columns.tolist())
else:
    plt.figure(figsize=(10, 5))
    plt.plot(cost_logit_sensitivity_df["pos_weight"], cost_logit_sensitivity_df["train_f1"], marker="o", label="Train F1")
    plt.plot(cost_logit_sensitivity_df["pos_weight"], cost_logit_sensitivity_df["val_f1"], marker="o", label="Validation F1")
    plt.plot(cost_logit_sensitivity_df["pos_weight"], cost_logit_sensitivity_df["val_recall"], marker="o", label="Validation Recall")
    plt.plot(cost_logit_sensitivity_df["pos_weight"], cost_logit_sensitivity_df["val_roc_auc"], marker="o", label="Validation ROC-AUC")
    plt.title("Cost-sensitive logistic regression sensitivity (positive weight)")
    plt.xlabel("Positive class weight")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    save_plot("cost_logit_hyperparameter_sensitivity.png")
    plt.show()


# %% [markdown]
# ### 15.3 Decision Tree sensitivity: max_depth

# %%
tree_depth_values = [2, 3, 4, 5, 6, 7, 8]

tree_sensitivity_df = hyperparameter_sensitivity(
    param_name="max_depth",
    param_values=tree_depth_values,
    factory_builder=lambda d: DecisionTreeScratch(
        max_depth=d,
        min_samples_split=30,
        min_samples_leaf=10,
        max_thresholds=25
    ),
    needs_scaling=False,
    X=X_all,
    y=y_all,
    numeric_idx=numeric_idx,
    k=5,
    threshold=0.5
)

tree_sensitivity_df.to_csv(OUTPUT_DIR / "tree_hyperparameter_sensitivity.csv", index=False)

print("Decision Tree sensitivity columns:")
print(tree_sensitivity_df.columns.tolist())
display(tree_sensitivity_df.round(4) if display else tree_sensitivity_df.round(4))


# %%
plt.figure(figsize=(10, 5))
plt.plot(tree_sensitivity_df["max_depth"], tree_sensitivity_df["train_f1"], marker="o", label="Train F1")
plt.plot(tree_sensitivity_df["max_depth"], tree_sensitivity_df["val_f1"], marker="o", label="Validation F1")
plt.plot(tree_sensitivity_df["max_depth"], tree_sensitivity_df["val_roc_auc"], marker="o", label="Validation ROC-AUC")
plt.title("Decision tree hyperparameter sensitivity (max_depth)")
plt.xlabel("Max depth")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
save_plot("tree_hyperparameter_sensitivity.png")
plt.show()

# %%
# Decision Tree sensitivity: max_depth
tree_depth_values = [2, 3, 4, 5, 6, 7, 8]
tree_sensitivity_df = hyperparameter_sensitivity(
    param_name="max_depth",
    param_values=tree_depth_values,
    factory_builder=lambda d: DecisionTreeScratch(
        max_depth=d, min_samples_split=30, min_samples_leaf=10, max_thresholds=25
    ),
    needs_scaling=False,
    X=X_all, y=y_all, numeric_idx=numeric_idx, k=5
)
tree_sensitivity_df.to_csv(OUTPUT_DIR / "tree_hyperparameter_sensitivity.csv", index=False)
display(tree_sensitivity_df.round(4) if display else tree_sensitivity_df.round(4))

# %%
plt.figure(figsize=(10, 5))
plt.plot(tree_sensitivity_df["max_depth"], tree_sensitivity_df["train_f1"], marker="o", label="Train F1")
plt.plot(tree_sensitivity_df["max_depth"], tree_sensitivity_df["val_f1"], marker="o", label="Validation F1")
plt.plot(tree_sensitivity_df["max_depth"], tree_sensitivity_df["val_roc_auc"], marker="o", label="Validation ROC-AUC")
plt.title("Decision tree hyperparameter sensitivity (max_depth)")
plt.xlabel("Max depth")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
save_plot("tree_hyperparameter_sensitivity.png")
plt.show()

# %% [markdown]
# ## 16. Learning Curve Analysis

# %%
def stratified_subsample_indices(y, frac, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos = max(1, int(np.ceil(len(pos) * frac)))
    n_neg = max(1, int(np.ceil(len(neg) * frac)))

    return np.concatenate([pos[:n_pos], neg[:n_neg]])

def learning_curve_analysis(model_name, spec, X, y, numeric_idx, train_fracs=None, k=5, threshold=0.5):
    if train_fracs is None:
        train_fracs = [0.2, 0.4, 0.6, 0.8, 1.0]

    folds = stratified_kfold_indices(y, k=k, seed=RANDOM_SEED)
    raw_rows = []

    for frac in train_fracs:
        for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
            X_train_raw, X_val_raw = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            sub_idx = stratified_subsample_indices(
                y_train, frac, seed=RANDOM_SEED + fold_id * 100 + int(frac * 100)
            )

            X_sub_raw = X_train_raw[sub_idx]
            y_sub = y_train[sub_idx]

            if spec["needs_scaling"]:
                X_sub, X_val, _, _ = standardize_train_val(X_sub_raw, X_val_raw, numeric_idx)
            else:
                X_sub, X_val = X_sub_raw.copy(), X_val_raw.copy()

            model = spec["factory"]()
            model.fit(X_sub, y_sub)

            train_prob = model.predict_proba(X_sub)
            val_prob = model.predict_proba(X_val)

            train_metrics = classification_metrics(y_sub, train_prob, threshold=threshold)
            val_metrics = classification_metrics(y_val, val_prob, threshold=threshold)

            raw_rows.append({
                "model": model_name,
                "frac": frac,
                "fold": fold_id,
                "train_size": len(y_sub),
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
            })

    raw_df = pd.DataFrame(raw_rows)
    summary_df = raw_df.groupby(["model", "frac"], as_index=False).agg({
        "train_size": "mean",
        "train_f1": ["mean", "std"],
        "val_f1": ["mean", "std"],
        "train_accuracy": ["mean", "std"],
        "val_accuracy": ["mean", "std"],
    })
    summary_df.columns = [
        "model", "frac", "avg_train_size",
        "train_f1_mean", "train_f1_std",
        "val_f1_mean", "val_f1_std",
        "train_acc_mean", "train_acc_std",
        "val_acc_mean", "val_acc_std",
    ]
    return raw_df, summary_df

# %%
learning_curve_raw_frames = []
learning_curve_summary_frames = []

for model_name, spec in model_specs.items():
    raw_lc_df, summary_lc_df = learning_curve_analysis(
        model_name=model_name,
        spec=spec,
        X=X_all,
        y=y_all,
        numeric_idx=numeric_idx,
        train_fracs=[0.2, 0.4, 0.6, 0.8, 1.0],
        k=5,
        threshold=0.5
    )
    learning_curve_raw_frames.append(raw_lc_df)
    learning_curve_summary_frames.append(summary_lc_df)

learning_curve_raw_df = pd.concat(learning_curve_raw_frames, ignore_index=True)
learning_curve_df = pd.concat(learning_curve_summary_frames, ignore_index=True)

learning_curve_raw_df.to_csv(OUTPUT_DIR / "learning_curves_raw.csv", index=False)
learning_curve_df.to_csv(OUTPUT_DIR / "learning_curves_summary.csv", index=False)

display(learning_curve_df.round(4) if display else learning_curve_df.round(4))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for model_name in existing_model_order(learning_curve_df["model"].unique()):
    sub = learning_curve_df[learning_curve_df["model"] == model_name]
    color = MODEL_COLORS[model_name]

    axes[0].plot(sub["avg_train_size"], sub["train_f1_mean"], marker="o", color=color, label=f"{model_name} Train")
    axes[0].plot(sub["avg_train_size"], sub["val_f1_mean"], marker="o", linestyle="--", color=color, label=f"{model_name} Val")
    axes[0].fill_between(
        sub["avg_train_size"],
        sub["val_f1_mean"] - sub["val_f1_std"].fillna(0),
        sub["val_f1_mean"] + sub["val_f1_std"].fillna(0),
        color=color, alpha=0.15
    )

    axes[1].plot(sub["avg_train_size"], sub["train_acc_mean"], marker="o", color=color, label=f"{model_name} Train")
    axes[1].plot(sub["avg_train_size"], sub["val_acc_mean"], marker="o", linestyle="--", color=color, label=f"{model_name} Val")
    axes[1].fill_between(
        sub["avg_train_size"],
        sub["val_acc_mean"] - sub["val_acc_std"].fillna(0),
        sub["val_acc_mean"] + sub["val_acc_std"].fillna(0),
        color=color, alpha=0.15
    )

axes[0].set_title("Learning curve (F1)")
axes[0].set_xlabel("Training samples")
axes[0].set_ylabel("F1 score")
axes[0].legend(fontsize=8)

axes[1].set_title("Learning curve (accuracy)")
axes[1].set_xlabel("Training samples")
axes[1].set_ylabel("Accuracy")
axes[1].legend(fontsize=8)

plt.tight_layout()
save_plot("learning_curves.png")
plt.show()

# %% [markdown]
# ## 17. Final Conclusion Markdown

# %%
best_f1_model = oof_summary["f1"].idxmax()
best_auc_model = oof_summary["roc_auc"].idxmax()
best_recall_model = oof_summary["recall"].idxmax()
fastest_model = summary_mean["train_time_sec"].idxmin()
smallest_gap_model = summary_mean["generalization_gap_f1"].abs().idxmin()

best_threshold = best_threshold_row["threshold"]
best_threshold_f1 = best_threshold_row["f1"]
best_cost_threshold = best_cost_threshold_row["threshold"]
best_cost_threshold_f1 = best_cost_threshold_row["f1"]

final_conclusion_md = f"""
## Final Conclusion

This project addressed a real-world **telecom customer churn prediction** problem using five machine learning models implemented **from scratch**:
- Logistic Regression
- Cost-Sensitive Logistic Regression
- Gaussian Naive Bayes
- Hybrid Naive Bayes
- Decision Tree

### Main findings

1. **Model performance**
   - Based on out-of-fold evaluation, **{best_f1_model}** achieved the highest **F1-score**.
   - In terms of ranking quality, **{best_auc_model}** achieved the highest **ROC-AUC**.
   - For identifying churn customers, **{best_recall_model}** achieved the highest **recall**.

2. **Innovation beyond the baseline**
   - We introduced a **Cost-Sensitive Logistic Regression** to better handle class imbalance by assigning more weight to churn samples.
   - We introduced a **Hybrid Naive Bayes** model that uses **Gaussian likelihoods for continuous variables** and **Bernoulli likelihoods for one-hot binary variables**, which better matches the mixed structure of the churn dataset.

3. **Computational efficiency**
   - The fastest model in average training time was **{fastest_model}**.
   - Simpler probabilistic models remained computationally efficient, while the decision tree required more training time due to threshold search.

4. **Generalization**
   - The model with the smallest train-validation F1 gap was **{smallest_gap_model}**, suggesting relatively stable generalization.
   - Learning-curve and generalization-gap analyses showed that higher flexibility does not always produce better validation performance.

5. **Threshold tuning**
   - For standard Logistic Regression, the best threshold in the sweep was **{best_threshold:.2f}**, with **F1 = {best_threshold_f1:.4f}**.
   - For Cost-Sensitive Logistic Regression, the best threshold in the sweep was **{best_cost_threshold:.2f}**, with **F1 = {best_cost_threshold_f1:.4f}**.
   - This confirms that the default threshold of 0.50 is not always optimal for churn prediction under class imbalance.

6. **Interpretability**
   - Logistic Regression provided direct coefficient-based interpretability.
   - Decision Tree provided feature-importance-based interpretability.
   - These analyses help explain which customer characteristics are associated with higher churn risk.

### Overall conclusion

Overall, the experiments show that **classical machine learning models implemented from scratch can effectively solve the telecom churn prediction task**.
The extended models also demonstrate that incorporating **distribution-aware modeling** and **imbalance-aware loss design** can provide meaningful methodological improvements beyond standard baselines.

### Future work

Possible future improvements include:
- adding advanced tree ensembles such as Random Forest or Gradient Boosting (if allowed),
- evaluating the models on multiple churn datasets,
- introducing pruning strategies for the decision tree,
- calibrating predicted probabilities,
- incorporating explicit business-cost optimization.
"""

(Path(OUTPUT_DIR) / "final_conclusion.md").write_text(final_conclusion_md, encoding="utf-8")

if display and Markdown:
    display(Markdown(final_conclusion_md))
else:
    print(final_conclusion_md)

# %% [markdown]
# ## 18. Optional Report Notes
#
# You can emphasize these points in your report:
# - churn prediction is a class-imbalanced binary classification task
# - F1 / recall / PR-AUC are important alongside accuracy
# - Hybrid Naive Bayes is a dataset-aware extension for mixed feature types
# - Cost-Sensitive Logistic Regression is a task-aware extension for imbalance
# - learning curves and generalization gaps explain bias–variance behavior
# - coefficients and importances support interpretability

# %%




