import streamlit as st
import pandas as pd
import numpy as np
import os
import altair as alt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# --- 1. 基本設定 ---
st.set_page_config(page_title="SECOM 意思決定支援シミュレーター", layout="wide")

# パス設定
base_dir = r"C:\Users\sr582\Downloads\キカガク\test3"
features_csv = os.path.join(base_dir, "data_processed", "secom_features_sorted.csv")
labels_csv = os.path.join(base_dir, "data_processed", "secom_labels_sorted.csv")
top_20_path = os.path.join(base_dir, "feature_selection", "top_20_features_list.csv")

@st.cache_resource
def load_and_train_model():
    X_raw = pd.read_csv(features_csv)
    y_raw = pd.read_csv(labels_csv).iloc[:, 0].replace(-1, 0)
    
    # 前処理
    X_filled = X_raw.fillna(X_raw.median())
    X_filled = X_filled.loc[:, X_filled.nunique() > 1]
    
    # SMOTE適用
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_filled, y_raw)
    
    # 学習
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_res, y_res)
    
    stats = {'median': X_filled.median(), 'min': X_filled.min(), 'max': X_filled.max()}
    
    # 【修正箇所】変数名を入れず、値のみを順番に返すように修正
    return model, X_filled.columns.tolist(), stats, y_raw

# 関数の戻り値を受け取る側
model, feature_names, stats, y_true = load_and_train_model()

# 全データ予測用データの準備
# 欠損値を埋めた状態で、モデルが学習した時と同じ列順に揃える
X_clean_df = pd.read_csv(features_csv).fillna(pd.read_csv(features_csv).median())
X_clean_df = X_clean_df[feature_names]

# --- 重要度データの読み込みとソート ---
try:
    # 寄与度データの読み込み (1列目:特徴量名, 2列目:寄与度スコア)
    top_20_df = pd.read_csv(top_20_path, header=None)
    top_20_df.columns = ['feature', 'importance']
    
    # 寄与度が高い順にソート
    top_20_df = top_20_df.sort_values(by='importance', ascending=False)
    
    # スライダー用のリストと辞書を作成
    sorted_features = top_20_df['feature'].tolist()
    importance_dict = dict(zip(top_20_df['feature'], top_20_df['importance']))
except Exception as e:
    st.error(f"重要度ファイルの読み込み失敗: {e}")
    sorted_features = feature_names[:20]
    importance_dict = {f: 0.0 for f in sorted_features}

# --- サイドバー：ビジネスコスト設定 (単位：百万円) ---
st.sidebar.header("💰 コストシミュレーション設定")
cost_miss = st.sidebar.number_input("見逃し1件の損失 (百万円)", value=10.0, step=1.0)
cost_false = st.sidebar.number_input("空振り1件の検査コスト (百万円)", value=0.5, step=0.1)

# --- メイン画面 ---
st.title("🏭 SECOM 意思決定支援シミュレーター")

tab_qa, tab_prod = st.tabs(["🔍 QA責任者モード (リスク検知)", "📈 生産・開発責任者モード (歩留まり最適化)"])

# --- 1. QA責任者モード ---
with tab_qa:
    st.header("品質保証(QA)：センサー異常による不合格リスク判定")
    col_input, col_res = st.columns([2, 1])
    
    with col_input:
        st.subheader("主要センサー値の調整（寄与度順）")
        st.caption("※寄与度が高い順に並んでいます。上位の値を動かすと不合格確率が大きく変動します。")
        input_values = {}
        input_cols = st.columns(2)
        
        for i, feat in enumerate(sorted_features):
            if feat not in stats['median']: continue 
            
            imp_val = importance_dict.get(feat, 0)
            prefix = "🔥 " if i < 5 else ""
            label_text = f"{prefix}{feat} (寄与度: {imp_val:.1%})"
            
            with input_cols[i % 2]:
                m_val = float(stats['median'][feat])
                min_v, max_v = float(stats['min'][feat]), float(stats['max'][feat])
                if min_v >= max_v: min_v, max_v = m_val - 1.0, m_val + 1.0
                
                input_values[feat] = st.slider(label_text, min_v, max_v, m_val, key=f"slider_{feat}")

    with col_res:
        st.subheader("リスク判定(アラートになる不合格確率)")
        current_input = pd.DataFrame([stats['median'].to_dict()])
        for k, v in input_values.items():
            current_input[k] = v
        current_input = current_input[feature_names]
        
        prob_fail = model.predict_proba(current_input)[0][1]
        
        qa_threshold = st.slider("⚖️ 検知しきい値（感度 = 低いほどFailを検知しやすくなる）", 0.0, 1.0, 0.3)
        
        if prob_fail > qa_threshold:
            st.error(f"### 🚨 判定は異常と予測\n不合格確率: **{prob_fail:.1%}**")
            st.warning("アクション: 直ちに再検査またはライン停止を検討")
        else:
            st.success(f"### ✅ 判定は正常と予測\n不合格確率: **{prob_fail:.1%}**")
            st.info("アクション: 次工程へパス可能")

# --- 2. 生産・開発責任者モード ---
with tab_prod:
    st.header("戦略決定：コスト最小化としきい値の最適化")
    
    # 1枚1枚全データの確率を計算
    all_probs = model.predict_proba(X_clean_df)[:, 1]
    opt_threshold = st.select_slider("戦略的しきい値の選択", options=np.round(np.arange(0.0, 1.01, 0.05), 2), value=0.3)
    
    preds = (all_probs > opt_threshold).astype(int)
    fn = np.sum((preds == 0) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    yield_rate = (np.sum(preds == 0) / len(y_true)) * 100
    
    total_miss_cost = fn * cost_miss
    total_false_cost = fp * cost_false
    total_cost = total_miss_cost + total_false_cost
    
    m1, m2, m3 = st.columns(3)
    m1.metric("推定歩留まり (Yield)", f"{yield_rate:.2f}%")
    m2.metric("見逃し / 空振り 件数", f"{fn} / {fp} 件")
    m3.metric("合計損失コスト", f"{total_cost:.1f} 百万円", delta=f"見逃し損: {total_miss_cost:.1f}M")

    st.subheader("コスト内訳の比較")
    cost_data = pd.DataFrame({
        'Category': ['見逃しコスト', '空振りコスト'],
        'Amount': [total_miss_cost, total_false_cost]
    })

    chart = alt.Chart(cost_data).mark_bar().encode(
        x=alt.X('Category:N', axis=alt.Axis(labelAngle=0), title='コスト項目'),
        y=alt.Y('Amount:Q', title='金額（百万円）'),
        color='Category:N'
    ).properties(height=350)
    
    st.altair_chart(chart, use_container_width=True)
    st.info(f"💡 現在のしきい値 **{opt_threshold}** における経営インパクトを表示しています。")