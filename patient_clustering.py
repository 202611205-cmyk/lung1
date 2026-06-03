import platform
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
import os

# ── [프로젝트 내부 폰트 파일 로드 방식] ───────────────────────────────────
def set_local_font():
    font_path = "NanumGothic-Regular.ttf" 
    
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "NanumGothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
        
    plt.rcParams["axes.unicode_minus"] = False

# 폰트 설정 적용
set_local_font()

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="환자 군집 분석", page_icon="🏥", layout="centered")

st.title("📋 환자 정보 입력")

# ── 입력 위젯 ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("나이", min_value=0.0, max_value=120.0, value=50.0, step=1.0, format="%.2f")
with col2:
    smoking = st.number_input("흡연량", min_value=0.0, max_value=100.0, value=10.0, step=1.0, format="%.2f")
with col3:
    alcohol = st.number_input("음주량", min_value=0.0, max_value=100.0, value=5.0, step=1.0, format="%.2f")

st.divider()

# ── 군집 분석 버튼 ───────────────────────────────────────────────────────────
if st.button("🔍 군집 분석하기", use_container_width=True):

    # ── 샘플 데이터 생성 (재현 가능) ─────────────────────────────────────────
    rng = np.random.default_rng(42)

    c0 = rng.multivariate_normal([4, 1.5], [[2, 0.3], [0.3, 0.5]], 30)
    c1 = rng.multivariate_normal([18, 5], [[8, 1], [1, 1]], 30)
    c2 = rng.multivariate_normal([28, 6.5], [[10, 1], [1, 1]], 30)

    X = np.vstack([c0, c1, c2])
    labels_true = np.array([0]*30 + [1]*30 + [2]*30)

    # ── KMeans 학습 ──────────────────────────────────────────────────────────
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km.fit(X)

    order = np.argsort(km.cluster_centers_[:, 0])
    remap = {old: new for new, old in enumerate(order)}
    labels_mapped = np.array([remap[l] for l in km.labels_])

    # 입력 환자 예측
    patient = np.array([[smoking, alcohol]])
    raw_pred = km.predict(patient)[0]
    patient_cluster = remap[raw_pred]

    cluster_names = {0: "매우 건강군", 1: "위험군", 2: "건강군"}
    cluster_colors = {0: "#F4A72A", 1: "#4C7FBF", 2: "#3BAA5C"}
    cluster_name = cluster_names[patient_cluster]
    cluster_color = cluster_colors[patient_cluster]

    # ── 결과 배너 ────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:#d4edda;border:1px solid #c3e6cb;border-radius:6px;
                    padding:12px 18px;font-size:16px;font-weight:600;color:#155724;">
            이 환자는 {patient_cluster}번 군집에 속합니다. ({cluster_name})
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("0번은 매우 건강군, 1번은 위험군, 2번은 건강군입니다.")

    # ── 산점도 그리기 직전 안전하게 폰트 다시 세팅 ──────────────────────────────
    set_local_font() 
    
    fig, ax = plt.subplots(figsize=(7, 5))

    color_map = ["#F4A72A", "#4C7FBF", "#3BAA5C"]
    label_map = ["0: 매우 건강군", "1: 위험군", "2: 건강군"]

    for cid in range(3):
        mask = labels_mapped == cid
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=color_map[cid], label=label_map[cid],
            s=60, alpha=0.85, edgecolors="white", linewidths=0.4,
        )

    # 현재 환자 표시
    ax.scatter(
        smoking, alcohol,
        marker="*", s=300, c="#1A237E",
        label="현재 환자", zorder=5, edgecolors="white", linewidths=0.5,
    )

    ax.set_title("군집 분석 결과", fontsize=14, fontweight="bold")
    ax.set_xlabel("흡연량")
    ax.set_ylabel("음주량")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    
    st.pyplot(fig)
