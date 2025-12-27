import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# --- 1. 基本設定 ---
st.set_page_config(page_title="SECOM Yield Simulator", layout="wide")

# パス設定
base_dir = r"C:\Users\sr582\Downloads\キカガク\test3"
features_csv = os.path.join(base_dir, "data_processed", "secom_features_sorted.csv")
labels_csv = os.path.join(base_dir, "data_processed", "secom_labels_sorted.csv")
top_20_path = os.path.join(base_dir, "feature_selection", "top_20_features_list.csv")

# --- 2. リソースのロードと学習 (キャッシュ利用) ---
@st.cache_resource
def load_and_train_model():
    # データ読み込み
    X_raw = pd.read_csv(features_csv)
    y_raw = pd.read_csv(labels_csv).iloc[:, 0].replace(-1, 0) # -1 -> 0(Pass), 1 -> 1(Fail)
    
    # --- 前処理 (エラー修正箇所) ---
    # 1. 欠損率50%以上削除
    missing_threshold = 0.5
    cols_to_drop_missing = X_raw.columns[X_raw.isnull().mean() > missing_threshold]
    X_cleaned = X_raw.drop(columns=cols_to_drop_missing)
    
    # 2. 定数カラム削除 (修正済み：Seriesの曖昧さを回避)
    # 各列のユニークな値の数を数え、1以下の列名を特定
    nunique = X_cleaned.nunique()
    const_cols = nunique[nunique <= 1].index
    X_cleaned = X_cleaned.drop(columns=const_cols)
    
    # 3. 中央値補完
    X_final = X_cleaned.fillna(X_cleaned.median())
    
    # SMOTE適用
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_final, y_raw)
    
    # モデル学習
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_res, y_res)
    
    # 上位20項目のリスト取得 (CSVから読み込み)
    top_20_df = pd.read_csv(top_20_path, index_col=0)
    top_20_names = top_20_df.index.tolist()
    
    return model, X_final, top_20_names

# リソース読み込みの実行
try:
    model, X_template, top_20_features = load_and_train_model()
except Exception as e:
    st.error(f"リソースの読み込み中にエラーが発生しました: {e}")
    st.stop()

# --- 3. UI 構成 ---
st.title("🛡️ SECOM 歩留まりシミュレーション & 意思決定支援システム")
st.markdown("1st Stepで特定された**重要変数20項目**を操作し、品質と生産のバランスをシミュレーションします。")

# サイドバー: パラメータ調整
st.sidebar.header("📊 プロセスパラメータ (Top 20)")
input_values = {}
for feat in top_20_features:
    # テンプレートデータの最小・最大・中央値を取得
    min_v = float(X_template[feat].min())
    max_v = float(X_template[feat].max())
    mid_v = float(X_template[feat].median())
    
    # スライダーの作成
    input_values[feat] = st.sidebar.slider(f"{feat}", min_v, max_v, mid_v)

# シミュレーション用データの作成 (全カラムのベースラインを中央値で作成し、スライダー値を上書き)
sim_row = X_template.median().to_frame().T
for feat, val in input_values.items():
    sim_row[feat] = val

# 予測実行 (Class 0: Pass, Class 1: Fail)
probs = model.predict_proba(sim_row)[0]
prob_pass = probs[0]
prob_fail = probs[1]

# --- 4. タブ別表示 ---
tab_qa, tab_prod = st.tabs(["🔍 QA責任者モード (リスク検知)", "⚙️ 生産・開発責任者モード (歩留まり最適化)"])

# --- QA責任者タブ ---
with tab_qa:
    st.header("品質保証 (QA) 視点：見逃しリスクの最小化")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        threshold = st.slider("検知しきい値", 0.0, 1.0, 0.3)
        is_fail = prob_fail > threshold
        if is_fail:
            st.error(f"🚨 【異常警告】\n\n予測Fail確率: {prob_fail:.1%}")
        else:
            st.success(f"✅ 【正常判定】\n\n予測Fail確率: {prob_fail:.1%}")

    with col2:
        st.info("**QAの意思決定ポイント:** SMOTEによりモデルはFailの予兆に敏感です。現場の許容度に合わせて『検知しきい値』を調整してください。")

# --- 生産・開発責任者タブ ---
with tab_prod:
    st.header("生産・開発視点：歩留まり最大化の探索")
    col3, col4 = st.columns([2, 1])
    
    with col3:
        yield_rate = prob_pass * 100
        st.metric("予測歩留まり (Yield Rate)", f"{yield_rate:.2f}%", delta=f"{(yield_rate - 93.4):.2f}% (ベースライン比)")
        
        chart_df = pd.DataFrame({
            "判定項目": ["良品 (Pass)", "不良 (Fail)"],
            "確率": [prob_pass, prob_fail]
        })
        
        c = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('判定項目', sort=None),
            y='確率',
            color=alt.Color('判定項目', scale=alt.Scale(domain=['良品 (Pass)', '不良 (Fail)'], range=['#2ecc71', '#e74c3c']))
        ).properties(height=400)
        st.altair_chart(c, use_container_width=True)

    with col4:
        st.write("**生産性のヒント:**")
        st.write("スライダーを動かして緑のバー（Pass）が最大になる設定を探してください。")
        st.warning("※絶対値ではなく『ベースラインからの変化幅』を評価基準にしてください。")

# --- 5. 統計的根拠の表示 ---
with st.expander("📈 このシミュレーションを支える統計的根拠"):
    st.write("1st Stepでの分析結果に基づき、以下のパイプラインで予測を行っています。")
    st.image(os.path.join(base_dir, "feature_selection", "analysis_visual_report.png"))