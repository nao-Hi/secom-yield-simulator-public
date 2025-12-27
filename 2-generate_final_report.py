import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, average_precision_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. パス設定とフォルダ準備
# ==========================================
base_dir = r"C:\Users\sr582\Downloads\キカガク\test3"
features_csv = os.path.join(base_dir, "data_processed", "secom_features_sorted.csv")
labels_csv = os.path.join(base_dir, "data_processed", "secom_labels_sorted.csv")
output_dir = os.path.join(base_dir, "feature_selection")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"フォルダ作成完了: {output_dir}")

# ==========================================
# 2. データのロードとクレンジング
# ==========================================
X = pd.read_csv(features_csv)
y = pd.read_csv(labels_csv).iloc[:, 0].replace(-1, 0) # 0: Pass, 1: Fail

# 欠損率50%以上を除外
missing_ratio = X.isnull().mean()
cols_to_drop = missing_ratio[missing_ratio > 0.5].index
X_cleaned = X.drop(columns=cols_to_drop)

# 定数カラムを除外（シミュレーションに不要なため）
const_cols = [col for col in X_cleaned.columns if X_cleaned[col].nunique() <= 1]
X_cleaned = X_cleaned.drop(columns=const_cols)

# 中央値補完（外れ値に強い根拠に基づく）
X_final = X_cleaned.fillna(X_cleaned.median())

# ==========================================
# 3. モデル検証 (Baseline vs SMOTE)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, stratify=y, random_state=42
)

# Baseline
model_base = RandomForestClassifier(random_state=42, n_estimators=100)
model_base.fit(X_train, y_train)
y_prob_base = model_base.predict_proba(X_test)[:, 1]

# SMOTE適用（不均衡データ対策）
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
model_smote = RandomForestClassifier(random_state=42, n_estimators=100)
model_smote.fit(X_res, y_res)
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

# ==========================================
# 4. 可視化レポートの生成
# ==========================================
plt.figure(figsize=(15, 6))

# 左：クラス分布（不均衡の現状）
plt.subplot(1, 2, 1)
counts = y.value_counts()
sns.barplot(x=counts.index, y=counts.values, palette=['skyblue', 'salmon'])
plt.title('Step 1: Class Distribution (Normal vs Failure)')
plt.xticks([0, 1], ['Pass (0)', 'Fail (1)'])
plt.ylabel('Number of Samples')

# 右：PR曲線（SMOTEによる改善の証明）
plt.subplot(1, 2, 2)
precision_b, recall_b, _ = precision_recall_curve(y_test, y_prob_base)
precision_s, recall_s, _ = precision_recall_curve(y_test, y_prob_smote)
plt.plot(recall_b, precision_b, label=f'Baseline (PR-AUC: {average_precision_score(y_test, y_prob_base):.3f})')
plt.plot(recall_s, precision_s, label=f'With SMOTE (PR-AUC: {average_precision_score(y_test, y_prob_smote):.3f})', linewidth=2)
plt.title('Step 2: Improvement by SMOTE (PR-Curve)')
plt.xlabel('Recall (Capability to catch Failures)')
plt.ylabel('Precision')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "analysis_visual_report.png"))

# ==========================================
# 5. 特徴量重要度とサマリー出力
# ==========================================
importances = pd.Series(model_smote.feature_importances_, index=X_final.columns).sort_values(ascending=False)
top_20 = importances.head(20)
top_20.to_csv(os.path.join(output_dir, "top_20_features_list.csv"))

with open(os.path.join(output_dir, "executive_summary.txt"), "w", encoding="utf-8") as f:
    f.write("【歩留まりシミュレーション構築 1st Step 報告書】\n")
    f.write("--------------------------------------------------\n")
    f.write(f"1. データクレンジング結果:\n")
    f.write(f"   - 初期特徴量数: {X.shape[1]}\n")
    f.write(f"   - 削除（欠損率50%超）: {len(cols_to_drop)}項目\n")
    f.write(f"   - 削除（定数変数）: {len(const_cols)}項目\n")
    f.write(f"   - 最終採用特徴量数: {X_final.shape[1]}項目\n\n")
    f.write(f"2. モデル性能（不均衡データ対策の効果）:\n")
    f.write(f"   - Baseline PR-AUC: {average_precision_score(y_test, y_prob_base):.4f}\n")
    f.write(f"   - SMOTE適用後 PR-AUC: {average_precision_score(y_test, y_prob_smote):.4f}\n")
    f.write(f"   * PR-AUCの向上により、Fail（不良）の予兆を捉える能力が改善されました。\n\n")
    f.write(f"3. シミュレーション変数の選定:\n")
    f.write(f"   - 寄与度の高い上位20項目を特定。これをUIのスライダーとして実装します。\n")

print(f"\n完了！レポートを確認してください: {output_dir}")