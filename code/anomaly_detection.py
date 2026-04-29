import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
from tfidf_features import build_tfidf

plt.rcParams["figure.dpi"] = 150


# 1. Isolation Forest (RCF) - Trained on REAL only

def isolation_forest_detection(X_train_vec, y_train, X_test_vec, contamination=0.1):
    """Run Isolation Forest on real news only. Returns (anomalies, scores, model)."""
    print("\n--- Isolation Forest (RCF) - Trained on REAL only ---")
    y_train = np.array(y_train)
    X_train_real = X_train_vec[y_train == 0]
    print(f"Training on real news only: {X_train_real.shape[0]:,} samples")

    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=2)
    model.fit(X_train_real)

    scores = model.score_samples(X_test_vec)
    anomalies = (model.predict(X_test_vec) == -1)
    print(f"Anomalies: {anomalies.sum():,} / {X_test_vec.shape[0]:,} ({anomalies.mean():.2%})")

    return anomalies, scores, model


# 2. One-Class SVM (SVD reduction) - Trained on REAL only

def one_class_svm_detection(X_train_vec, y_train, X_test_vec, contamination=0.1, n_components=100):
    """Run One-Class SVM on real news only with SVD reduction. Returns (anomalies, scores, model)."""
    print("\n--- One-Class SVM (SVD 100d) - Trained on REAL only ---")
    y_train = np.array(y_train)
    X_train_real = X_train_vec[y_train == 0]
    print(f"Training on real news only: {X_train_real.shape[0]:,} samples")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_tr_svd = svd.fit_transform(X_train_real)
    X_te_svd = svd.transform(X_test_vec)
    print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.2%}")

    model = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
    model.fit(X_tr_svd)

    scores = model.score_samples(X_te_svd)
    anomalies = (model.predict(X_te_svd) == -1)
    print(f"Anomalies: {anomalies.sum():,} / {X_test_vec.shape[0]:,} ({anomalies.mean():.2%})")

    return anomalies, scores, model


# 3. Analysis

def run_analysis(y, anom_if, scores_if, anom_svm, scores_svm):
    """Print enrichment, percentile, and statistical significance analysis."""
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    baseline = y.mean()
    for name, anom, sc in [("Isolation Forest (REAL trained)", anom_if, scores_if),
                            ("One-Class SVM (REAL trained)", anom_svm, scores_svm)]:
        print(f"\n[{name}]")
        if anom.sum() > 0:
            enrich = y[anom].mean() / baseline
            f1 = f1_score(y, anom.astype(int), zero_division=0)
            print(f"  Fake in anomalies: {y[anom].mean():.2%} (enrichment {enrich:.2f}x)")
            print(f"  Fake in normals:   {y[~anom].mean():.2%}")
            print(f"  As fake detector F1: {f1:.3f}")

        for pct in [5, 10, 25]:
            mask = sc <= np.percentile(sc, pct)
            print(f"  Bottom {pct:2d}% score → {y[mask].mean():.1%} fake ({mask.sum():,} samples)")

        stat, pval = stats.mannwhitneyu(sc[y == 1], sc[y == 0], alternative="two-sided")
        sig = "***" if pval < 0.001 else "ns"
        print(f"  Mann-Whitney p={pval:.2e} {sig}")

    agree = (anom_if == anom_svm).mean()
    both = (anom_if & anom_svm).sum()
    print(f"\nAgreement: {agree:.2%}, Both anomaly: {both:,}")
    if both > 0:
        print(f"Fake ratio (both agree): {y[anom_if & anom_svm].mean():.2%}")

    return agree, both


# 4. Visualization

def plot_all(y, anom_if, scores_if, anom_svm, scores_svm, save_dir="results/figures"):
    """Save 6 plots: score distributions, agreement pie, fake ratio bars, and PR curves."""
    os.makedirs(save_dir, exist_ok=True)
    baseline = y.mean()
    both = (anom_if & anom_svm).sum()

    
    plt.figure(figsize=(8, 5))
    plt.hist(scores_if[y == 0], bins=50, alpha=0.6, label="Real", color="steelblue", density=True)
    plt.hist(scores_if[y == 1], bins=50, alpha=0.6, label="Fake", color="tomato", density=True)
    plt.title("Isolation Forest Score Distribution (Trained on REAL)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anomaly_if_score_distribution.png", dpi=150)
    plt.close()
    print("[SAVED] anomaly_if_score_distribution.png")

    
    plt.figure(figsize=(8, 5))
    plt.hist(scores_svm[y == 0], bins=50, alpha=0.6, label="Real", color="steelblue", density=True)
    plt.hist(scores_svm[y == 1], bins=50, alpha=0.6, label="Fake", color="tomato", density=True)
    plt.title("One-Class SVM Score Distribution (Trained on REAL)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anomaly_svm_score_distribution.png", dpi=150)
    plt.close()
    print("[SAVED] anomaly_svm_score_distribution.png")

   
    plt.figure(figsize=(7, 7))
    only_if = (anom_if & ~anom_svm).sum()
    only_svm = (~anom_if & anom_svm).sum()
    labels = ["Both Anomaly", "IF Only", "SVM Only", "Both Normal"]
    sizes = [both, only_if, only_svm, (~anom_if & ~anom_svm).sum()]
    colors = ["#FFE66D", "#4ECDC4", "#FF6B6B", "#95E1D3"]
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title("Method Agreement (Both Trained on REAL)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anomaly_agreement_pie.png", dpi=150)
    plt.close()
    print("[SAVED] anomaly_agreement_pie.png")

    
    plt.figure(figsize=(8, 5))
    categories = ["Overall", "IF\nAnomaly", "IF\nNormal", "SVM\nAnomaly", "SVM\nNormal"]
    ratios = [
        baseline,
        y[anom_if].mean() if anom_if.sum() else 0,
        y[~anom_if].mean() if (~anom_if).sum() else 0,
        y[anom_svm].mean() if anom_svm.sum() else 0,
        y[~anom_svm].mean() if (~anom_svm).sum() else 0,
    ]
    bar_colors = ["gray", "tomato", "steelblue", "tomato", "steelblue"]
    bars = plt.bar(categories, ratios, color=bar_colors, edgecolor="gray", alpha=0.8)
    for b, r in zip(bars, ratios):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                 f"{r:.1%}", ha="center", va="bottom", fontsize=10)
    plt.title("Fake Ratio: Anomaly vs Normal (Trained on REAL)")
    plt.ylabel("Fake Ratio")
    plt.axhline(baseline, color="gray", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anomaly_fake_ratio_comparison.png", dpi=150)
    plt.close()
    print("[SAVED] anomaly_fake_ratio_comparison.png")

   
    plt.figure(figsize=(7, 6))
    prec_if, rec_if, _ = precision_recall_curve(y, -scores_if)
    ap_if = average_precision_score(y, -scores_if)
    plt.plot(rec_if, prec_if, color="darkorange", lw=2, label=f"IF (AP={ap_if:.3f})")
    plt.axhline(baseline, color="gray", ls="--", label=f"Baseline {baseline:.1%}")
    plt.title("PR Curve - Isolation Forest (Trained on REAL)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anomaly_if_pr_curve.png", dpi=150)
    plt.close()
    print("[SAVED] anomaly_if_pr_curve.png")

    
    plt.figure(figsize=(7, 6))
    prec_svm, rec_svm, _ = precision_recall_curve(y, -scores_svm)
    ap_svm = average_precision_score(y, -scores_svm)
    plt.plot(rec_svm, prec_svm, color="purple", lw=2, label=f"SVM (AP={ap_svm:.3f})")
    plt.axhline(baseline, color="gray", ls="--", label=f"Baseline {baseline:.1%}")
    plt.title("PR Curve - One-Class SVM (Trained on REAL)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/anomaly_svm_pr_curve.png", dpi=150)
    plt.close()
    print("[SAVED] anomaly_svm_pr_curve.png")

    return ap_if, ap_svm


# 5. Save Results

def save_results(y, anom_if, anom_svm, agree, ap_if, ap_svm, save_dir="results/csv"):
    """Save anomaly detection summary to CSV."""
    os.makedirs(save_dir, exist_ok=True)
    baseline = y.mean()

    df = pd.DataFrame([
        {
            "method": "Isolation Forest",
            "trained_on": "real_only",
            "n_anomalies": int(anom_if.sum()),
            "fake_in_anomalies": round(float(y[anom_if].mean()), 4) if anom_if.sum() else None,
            "enrichment": round(float(y[anom_if].mean() / baseline), 2) if anom_if.sum() else None,
            "average_precision": round(float(ap_if), 4),
        },
        {
            "method": "One-Class SVM",
            "trained_on": "real_only",
            "n_anomalies": int(anom_svm.sum()),
            "fake_in_anomalies": round(float(y[anom_svm].mean()), 4) if anom_svm.sum() else None,
            "enrichment": round(float(y[anom_svm].mean() / baseline), 2) if anom_svm.sum() else None,
            "average_precision": round(float(ap_svm), 4),
        },
    ])
    df.to_csv(f"{save_dir}/anomaly_detection_results.csv", index=False)
    print("[SAVED] anomaly_detection_results.csv")



# 6. Run Full Anomaly Detection Pipeline

def run_anomaly_detection():
    """Run full anomaly detection pipeline: TF-IDF -> IF + SVM -> analysis -> plots."""
    X_train_vec, y_train, X_test_vec, y_test = build_tfidf()
    y = y_test.values if hasattr(y_test, "values") else np.array(y_test)
    print(f"Train: {X_train_vec.shape}, Test: {X_test_vec.shape}, Fake ratio: {y.mean():.2%}")

    anom_if, scores_if, _ = isolation_forest_detection(X_train_vec, y_train, X_test_vec)
    anom_svm, scores_svm, _ = one_class_svm_detection(X_train_vec, y_train, X_test_vec)

    agree, both = run_analysis(y, anom_if, scores_if, anom_svm, scores_svm)
    ap_if, ap_svm = plot_all(y, anom_if, scores_if, anom_svm, scores_svm)
    save_results(y, anom_if, anom_svm, agree, ap_if, ap_svm)

    print("\nDone!")


# 7. Main 

if __name__ == "__main__":
    run_anomaly_detection()