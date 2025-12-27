# 1-feature_selection.py

# 実務的なクレンジング: 定数（値が変わらない）カラムや、欠損率が極端に高いカラムを統計的根拠に基づき削除。
# 不均衡対策の比較: 通常時とSMOTE適用時の「Recall」と「PR-AUC」を比較算出。
# 特徴量重要度の抽出: シミュレーションの「スライダー」候補となる上位20個を特定。

# ターミナルで以下を実行して必要なライブラリをインストールしておく。:
# pip install pandas numpy scikit-learn imbalanced-learn matplotlib
# pip install pandas


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

# 0-pandasのバージョン確認
print(f"Pandas version: {pd.__version__}")

# 1. パス設定
base_dir = r"C:\Users\sr582\Downloads\キカガク\test3"
features_csv = os.path.join(base_dir, "data_processed", "secom_features_sorted.csv")
labels_csv = os.path.join(base_dir, "data_processed", "secom_labels_sorted.csv")
output_dir = os.path.join(base_dir, "feature_selection")

# 保存先フォルダの作成
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory created: {output_dir}")

# 2. データのロード
X = pd.read_csv(features_csv)
y = pd.read_csv(labels_csv).iloc[:, 0].replace(-1, 0) # 0: Pass, 1: Fail

# 3. データクレンジング（統計的根拠に基づく除外）
# 欠損率50%以上を除外
missing_ratio = X.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.5].index
X_cleaned = X.drop(columns=cols_to_drop)

# 定数カラムを除外
const_cols = [col for col in X_cleaned.columns if X_cleaned[col].nunique() <= 1]
X_cleaned = X_cleaned.drop(columns=const_cols)

# 中央値補完
X_final = X_cleaned.fillna(X_cleaned.median())

# 4. モデル検証とサンプリング比較
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y, random_state=42)

# Baseline
model_base = RandomForestClassifier(random_state=42, n_estimators=100)
model_base.fit(X_train, y_train)
y_prob_base = model_base.predict_proba(X_test)[:, 1]

# SMOTE適用
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
model_smote = RandomForestClassifier(random_state=42, n_estimators=100)
model_smote.fit(X_res, y_res)
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

# 5. レポート用グラフの作成と保存
plt.figure(figsize=(14, 6))

# グラフ1: クラス分布（不均衡の可視化）
plt.subplot(1, 2, 1)
sns.countplot(x=y, palette=['skyblue', 'salmon'])
plt.title('Class Distribution (Pass: 0, Fail: 1)')
plt.xlabel('Status')
plt.ylabel('Count')

# グラフ2: PR曲線（検知能力の比較）
plt.subplot(1, 2, 2)
precision_b, recall_b, _ = precision_recall_curve(y_test, y_prob_base)
precision_s, recall_s, _ = precision_recall_curve(y_test, y_prob_smote)
plt.plot(recall_b, precision_b, label=f'Baseline (PR-AUC: {average_precision_score(y_test, y_prob_base):.3f})')
plt.plot(recall_s, precision_s, label=f'With SMOTE (PR-AUC: {average_precision_score(y_test, y_prob_smote):.3f})')
plt.title('PR Curve Comparison')
plt.xlabel('Recall (Detection Rate)')
plt.ylabel('Precision')
plt.legend()

plt.tight_layout()
report_img_path = os.path.join(output_dir, "eda_pr_comparison.png")
plt.savefig(report_img_path)
print(f"Graph saved: {report_img_path}")

# 6. 特徴量重要度と解析レポートの保存
importances = pd.Series(model_smote.feature_importances_, index=X_final.columns).sort_values(ascending=False)
top_20 = importances.head(20)
top_20.to_csv(os.path.join(output_dir, "top_20_features.csv"))

with open(os.path.join(output_dir, "analysis_summary.txt"), "w") as f:
    f.write("=== SECOM Yield Simulation: 1st Step Report ===\n")
    f.write(f"Initial Features: {X.shape[1]}\n")
    f.write(f"Dropped (Missing > 50%): {len(cols_to_drop)}\n")
    f.write(f"Dropped (Constant): {len(const_cols)}\n")
    f.write(f"Final Features for Simulation: {X_final.shape[1]}\n\n")
    f.write("=== Model Performance (SMOTE Effect) ===\n")
    f.write(f"Baseline PR-AUC: {average_precision_score(y_test, y_prob_base):.4f}\n")
    f.write(f"SMOTE Applied PR-AUC: {average_precision_score(y_test, y_prob_smote):.4f}\n")

print(f"Report files saved in: {output_dir}")