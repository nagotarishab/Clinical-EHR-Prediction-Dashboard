import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(_SCRIPT_DIR, "synthea-mimic", "csv")
sys.path.append(os.path.join(_SCRIPT_DIR, "src"))

from data_processing import ClinicalDataProcessor
from models import ModelPipeline

logging.basicConfig(level=logging.INFO)

# High-end aesthetic configuration
st.set_page_config(
    page_title="EH Drift & Continual Learning",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "## Team-32\n"
            "### Enterprise Clinical Prediction Dashboard\n"
            "---\n"
            "| Member | ID |\n"
            "|---|---|\n"
            "| Advik Kashi Vishwanath | 2022B4A70973H |\n"
            "| Rishab Nagota | 2022B1A71229H |\n"
            "| Rajas Bamb | 2022B2A31579H |\n"
            "| Sohan | 2023A3PS0389H |\n"
        ),
    },
)

# Set base seaborn theme for all matplotlib plots safely blending with dark/light themes
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#4A5568'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['text.color'] = '#A0AEC0'
plt.rcParams['axes.labelcolor'] = '#A0AEC0'
plt.rcParams['xtick.color'] = '#A0AEC0'
plt.rcParams['ytick.color'] = '#A0AEC0'
plt.rcParams['grid.color'] = '#4A5568'
plt.rcParams['grid.alpha'] = 0.3


# Custom CSS for modern, premium appearance
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sleek typography */
    h1, h2, h3 { 
        color: #1A365D; 
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Metrics styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2B6CB0;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #4A5568;
    }
    
    /* Subtle container borders and spacing */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    /* Cards */
    div.row-widget.stRadio > div{flex-direction:row;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    hr {
        margin: 1.5em 0;
        border: 0;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


st.title("Enterprise Clinical Prediction Dashboard")
st.markdown("Automated monitoring for clinical predictive performance under temporal drift & continual learning.")

if "processed" not in st.session_state:
    st.session_state.processed = False

# Helper functions (UI Improvements added)
def _metrics_df(eval_dict):
    return pd.DataFrame({k: v["metrics"] for k, v in eval_dict.items()}).T

def _plot_metric_grouped_bars(metric, ev_d1_test, ev_d2_pre, ev_d2_post=None):
    names = list(ev_d1_test.keys())
    x = np.arange(len(names))
    w = 0.22
    fig, ax = plt.subplots(figsize=(10, 4.5))
    v1 = [ev_d1_test[n]["metrics"][metric] for n in names]
    v2 = [ev_d2_pre[n]["metrics"][metric] for n in names]
    
    ax.bar(x - w, v1, w, label="Historical Data", color="#4299E1", edgecolor="none", alpha=0.9)
    ax.bar(x, v2, w, label="Current Data (No CL)", color="#F6E05E", edgecolor="none", alpha=0.9)
    
    if ev_d2_post is not None:
        v3 = [ev_d2_post[n]["metrics"][metric] for n in names]
        ax.bar(x + w, v3, w, label="Current Data (After CL)", color="#48BB78", edgecolor="none", alpha=0.9)
        
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight='bold')
    ax.set_ylabel(metric, fontweight='bold')
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    allv = list(v1) + list(v2)
    if ev_d2_post is not None:
        allv += [ev_d2_post[n]["metrics"][metric] for n in names]
    ymax = max(allv) if allv else 1.0
    ax.set_ylim(0, min(1.05, ymax * 1.15 + 0.02))
    plt.tight_layout()
    return fig

def _plot_rocs_before_after(ev_before, ev_after, y_true, title="Future Evaluation"):
    names = list(ev_before.keys())
    n = max(1, len(names))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    axes = np.atleast_1d(axes[0])
    for ax, name in zip(axes, names):
        fpr1, tpr1, auc1 = ModelPipeline.roc_curve_data(y_true, ev_before[name]["y_score"])
        fpr2, tpr2, auc2 = ModelPipeline.roc_curve_data(y_true, ev_after[name]["y_score"])
        if fpr1 is not None:
            ax.plot(fpr1, tpr1, label=f"Before CL (AUC={auc1:.3f})", color="#E53E3E", linewidth=2.5)
        if fpr2 is not None:
            ax.plot(fpr2, tpr2, label=f"After CL (AUC={auc2:.3f})", color="#38A169", linewidth=2.5)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.15)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{name}", fontweight='bold')
        ax.legend(loc="lower right", frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

def _plot_confusions_dual(ev_d1, ev_d2, row_titles=("Historical", "Current")):
    names = list(ev_d1.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
    for j, name in enumerate(names):
        ConfusionMatrixDisplay(ev_d1[name]["confusion"]).plot(ax=axes[0, j], colorbar=False, cmap="Blues")
        axes[0, j].set_title(f"{name} ({row_titles[0]})", fontweight='bold', pad=15)
        axes[0, j].grid(False)
        ConfusionMatrixDisplay(ev_d2[name]["confusion"]).plot(ax=axes[1, j], colorbar=False, cmap="Blues")
        axes[1, j].set_title(f"{name} ({row_titles[1]})", fontweight='bold', pad=15)
        axes[1, j].grid(False)
    plt.tight_layout()
    return fig

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("### ⚙️ Pipeline Configuration")
data_dir = st.sidebar.text_input("Data Source Directory", _DEFAULT_DATA)
temporal_split = st.sidebar.date_input("Temporal Cutoff", value=pd.to_datetime("2018-01-01"))
target_cond = st.sidebar.selectbox(
    "Clinical Target",
    ["Hypertension", "Diabetes", "Bronchitis", "Sinusitis"],
)
test_frac = st.sidebar.slider("Test Set Fraction", 0.1, 0.4, 0.2, 0.05)
dt_depth = st.sidebar.slider("Decision Tree Depth", 3, 25, 10)

st.sidebar.markdown("---")

if st.sidebar.button("🚀 Initialize Data Pipeline", type="primary", use_container_width=True):
    with st.spinner("Processing clinical encounters and generating temporal cohorts..."):
        try:
            processor = ClinicalDataProcessor(
                data_dir=data_dir,
                temporal_split_date=temporal_split.strftime("%Y-%m-%d"),
                target_condition=target_cond,
                test_size=test_frac,
            )
            df = processor.load_and_merge()
            (
                X_h_train, y_h_train, X_h_test, y_h_test,
                X_c_train, y_c_train, X_c_test, y_c_test, fnames
            ) = processor.split_and_preprocess(df)

            st.session_state.processor = processor
            st.session_state.df = df
            st.session_state.X_h_train, st.session_state.y_h_train = X_h_train, y_h_train
            st.session_state.X_h_test, st.session_state.y_h_test = X_h_test, y_h_test
            st.session_state.X_c_train, st.session_state.y_c_train = X_c_train, y_c_train
            st.session_state.X_c_test, st.session_state.y_c_test = X_c_test, y_c_test
            st.session_state.fnames = fnames
            st.session_state.processed = True
            st.session_state.models_trained = False
            
            # Clear old evaluation metrics to prevent shape mismatches
            for k in ["model_pipeline", "ev_d1_train", "ev_d1_test", "ev_d2_test", "ev_d2_after", "ev_num_d1", "ev_num_d2", "dt_sweep_df"]:
                st.session_state.pop(k, None)
                
            st.sidebar.success("Pipeline deployed successfully.")
            pos_tr = int(np.sum(y_h_train))
            if pos_tr < 40:
                st.sidebar.warning(f"Low positive class prevalence ({pos_tr} cases). Models may exhibit high variance.")
        except Exception as e:
            st.sidebar.error("Execution Failed.")
            logging.exception("pipeline")

if not st.session_state.processed:
    st.info("👋 Welcome! Use the sidebar configuration to establish temporal cohorts and initialize the pipeline.")
    st.stop()

proc = st.session_state.processor
fn = st.session_state.fnames
ts = pd.Timestamp(proc.temporal_split_date)
df_all = st.session_state.df
d1_full = df_all[df_all["START"] < ts].copy()
d2_full = df_all[df_all["START"] >= ts].copy()

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3 = st.tabs([
    "📊 Dataset Overview", 
    "📈 Baseline Model Performance & Drift", 
    "🔄 Continual Learning"
])

# --- TAB 1: DATASET OVERVIEW ---
with tab1:
    st.markdown("### Clinical Dataset Cohort Overview")
    
    # Top KPI Metrics row
    col1, col2, col3, col4 = st.columns(4)
    mi = getattr(proc, "merge_inspection", {}) or {}
    total_unified = mi.get("unified_encounters", 0)
    unique_patients = mi.get("unique_patients", 0)
    
    with col1:
        st.metric("Total Encounters", f"{total_unified:,}")
    with col2:
        st.metric("Unique Patients", f"{unique_patients:,}")
    with col3:
        st.metric("Historical Set (Train)", f"{len(st.session_state.y_h_train):,}")
    with col4:
        st.metric("Current Set (Train)", f"{len(st.session_state.y_c_train):,}")

    st.markdown("---")
    
    # Drift & Distribution
    st.markdown("#### Temporal Drift & Prevalence")
    colA, colB = st.columns([1, 1])
    
    with colA:
        cls_rows = []
        for name, part in ("Historical (Regime 1)", d1_full), ("Current (Regime 2)", d2_full):
            n = len(part)
            pos = int(part["TARGET"].sum()) if n else 0
            cls_rows.append({"Regime": name, "n": n, "positives": pos, "P(target)": (pos / n) if n else 0.0})
        cls_df = pd.DataFrame(cls_rows)
        
        figc, axc = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(cls_df))
        bars_neg = axc.bar(x - 0.2, cls_df["n"] - cls_df["positives"], 0.4, label="Negative (0)", color="#CBD5E0")
        bars_pos = axc.bar(x + 0.2, cls_df["positives"], 0.4, label="Positive (1)", color="#4299E1")
        
        axc.set_yscale("log")
        axc.set_ylabel("Count (Log Scale)")
        
        for bars in [bars_neg, bars_pos]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    axc.annotate(f'{int(height):,}',
                                 xy=(bar.get_x() + bar.get_width() / 2, height),
                                 xytext=(0, 3),
                                 textcoords="offset points",
                                 ha='center', va='bottom', fontsize=8)

        axc.set_xticks(x)
        axc.set_xticklabels(cls_df["Regime"], fontweight='bold')
        axc.legend(frameon=False)
        axc.spines['top'].set_visible(False)
        axc.spines['right'].set_visible(False)
        
        # Expand y-limit slightly to fit annotations
        axc.set_ylim(bottom=1, top=axc.get_ylim()[1] * 5)
        st.pyplot(figc)
        plt.close(figc)

    with colB:
        st.markdown(f"**Target Condition:** {target_cond}")
        prev_d1 = cls_df.iloc[0]['P(target)']
        prev_d2 = cls_df.iloc[1]['P(target)']
        
        c1, c2 = st.columns(2)
        c1.metric("Historical Prevalence", f"{prev_d1:.2%}")
        c2.metric("Current Prevalence", f"{prev_d2:.2%}", delta=f"{(prev_d2 - prev_d1)*100:.2f}%", delta_color="inverse")
        
        if "AGE_AT_ENCOUNTER" in d1_full.columns:
            age_d1 = d1_full["AGE_AT_ENCOUNTER"].mean()
            age_d2 = d2_full["AGE_AT_ENCOUNTER"].mean()
            st.metric("Avg Patient Age (Current)", f"{age_d2:.1f} yrs", delta=f"{age_d2 - age_d1:.1f} yrs from Historical")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Advanced Data & Density Expander
    with st.expander("Advanced Exploratory Data Analysis & Descriptive Stats"):
        c_density, c_tables = st.columns(2)
        with c_density:
            st.markdown("**Density Drift: Numerical Variables**")
            n_num = len(proc.num_cols)
            numeric_names = fn[:n_num]
            
            # Only keep smooth continuous variables that make sense for a density plot
            continuous_vars = [
                "AGE_AT_ENCOUNTER",
                "mean_Body_Height",
                "mean_Body_Weight",
                "mean_Body_mass_index_BMI_Ratio",
                "mean_Diastolic_Blood_Pressure",
                "mean_Heart_rate",
                "mean_Respiratory_rate",
                "mean_Systolic_Blood_Pressure"
            ]
            plot_options = [n for n in numeric_names if n in continuous_vars]
            if not plot_options:
                plot_options = numeric_names[:min(5, len(numeric_names))]
                
            pick = st.selectbox("Select Feature Visualization", plot_options, label_visibility="collapsed")
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.kdeplot(proc.eda_num_h_test[pick], fill=True, label="Historical", color="#4299E1", ax=ax, alpha=0.4)
            sns.kdeplot(proc.eda_num_c_test[pick], fill=True, label="Current", color="#F6E05E", ax=ax, alpha=0.4)
            ax.set_xlabel(f"{pick} (Raw Values)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)
            
        with c_tables:
            num_eda = [c for c in proc.num_cols if c in d1_full.columns and c in d2_full.columns][:35]
            if num_eda:
                st.markdown("**Descriptive Statistics (Current Cohort)**")
                st.dataframe(d2_full[num_eda].describe().T.round(3), use_container_width=True, height=350)
                
            miss_top = mi.get("missing_cells_top")
            if miss_top is not None and len(miss_top) > 0:
                st.markdown("**Top Missing Variables Profiling**")
                st.dataframe(miss_top, use_container_width=True)
                
        st.markdown("---")
        st.markdown("**Feature Correlation Matrix (Historical Cohort)**")
        # Select a clean, clinically relevant subset of variables for correlation
        corr_vars = [
            "AGE_AT_ENCOUNTER",
            "mean_Body_Height",
            "mean_Body_Weight",
            "mean_Diastolic_Blood_Pressure",
            "mean_Systolic_Blood_Pressure",
            "mean_Heart_rate",
            "mean_Respiratory_rate",
            "med_n",
            "proc_n"
        ]
        num_eda = [c for c in corr_vars if c in d1_full.columns]
        if num_eda:
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            corr = d1_full[num_eda].corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax_corr, cbar_kws={"shrink": 0.8}, annot_kws={"size": 10})
            ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=10)
            ax_corr.set_yticklabels(ax_corr.get_yticklabels(), fontsize=10)
            st.pyplot(fig_corr)
            plt.close(fig_corr)

# --- TAB 2: MODEL PERFORMANCE ---
with tab2:
    st.markdown("### Evaluation & Temporal Degradation")
    if st.button("🚀 Train Baseline Models (Historical Cohort)", type="primary"):
        with st.spinner("Fitting Decision Tree, SVM, and MLP Ensembles..."):
            pipe = ModelPipeline(dt_max_depth=dt_depth)
            pipe.train_base_models(st.session_state.X_h_train, st.session_state.y_h_train)
            st.session_state.model_pipeline = pipe
            st.session_state.ev_d1_train = pipe.evaluate_models(st.session_state.X_h_train, st.session_state.y_h_train, "D1 train")
            st.session_state.ev_d1_test = pipe.evaluate_models(st.session_state.X_h_test, st.session_state.y_h_test, "D1 test")
            st.session_state.ev_d2_test = pipe.evaluate_models(st.session_state.X_c_test, st.session_state.y_c_test, "D2 test")
            st.session_state.models_trained = True

    if st.session_state.get("models_trained"):
        # Model KPI Cards
        st.markdown("<br>", unsafe_allow_html=True)
        d1m = _metrics_df(st.session_state.ev_d1_test)
        d2m = _metrics_df(st.session_state.ev_d2_test)
        
        metrics_to_show = ["ROC-AUC", "F1-Score", "Accuracy", "Precision", "Recall"]
        models_available = list(d1m.index)
        
        for idx, model_name in enumerate(models_available):
            st.markdown(f"#### {model_name} Performance")
            m_cols = st.columns(5)
            for m_idx, metric in enumerate(metrics_to_show):
                val_d1 = d1m.loc[model_name, metric]
                val_d2 = d2m.loc[model_name, metric]
                delta = val_d2 - val_d1
                m_cols[m_idx].metric(
                    f"{metric} (Current)", 
                    f"{val_d2:.3f}", 
                    delta=f"{delta:+.3f} vs Baseline", 
                    delta_color="normal"
                )
            if idx < len(models_available) - 1:
                st.markdown("<hr style='margin: 1em 0; border-top: 1px dashed #E2E8F0'>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Overfitting Analysis (Train vs Test Accuracy)")
        d1m_train = _metrics_df(st.session_state.ev_d1_train)
        
        fig_over, ax_over = plt.subplots(figsize=(8, 4))
        x = np.arange(len(models_available))
        w = 0.25
        train_acc = [d1m_train.loc[m, "Accuracy"] for m in models_available]
        test1_acc = [d1m.loc[m, "Accuracy"] for m in models_available]
        test2_acc = [d2m.loc[m, "Accuracy"] for m in models_available]
        
        ax_over.bar(x - w, train_acc, w, label="Train (Historical)", color="#4299E1", edgecolor="none", alpha=0.9)
        ax_over.bar(x, test1_acc, w, label="Test (Historical)", color="#48BB78", edgecolor="none", alpha=0.9)
        ax_over.bar(x + w, test2_acc, w, label="Test (Current)", color="#F6E05E", edgecolor="none", alpha=0.9)
        
        ax_over.set_xticks(x)
        ax_over.set_xticklabels(models_available, fontweight='bold')
        ax_over.set_ylabel("Accuracy", fontweight='bold')
        ax_over.legend(frameon=False, loc="upper right")
        ax_over.grid(axis='y', linestyle='--', alpha=0.5)
        ax_over.spines['top'].set_visible(False)
        ax_over.spines['right'].set_visible(False)
        ymax_over = max(max(train_acc), max(test1_acc), max(test2_acc))
        ax_over.set_ylim(0, min(1.05, ymax_over * 1.15 + 0.02))
        st.pyplot(fig_over)
        plt.close(fig_over)

        st.markdown("---")
        st.markdown("#### Diagnostic Validation: Historical vs Current Cohorts")
        fig_cm = _plot_confusions_dual(st.session_state.ev_d1_test, st.session_state.ev_d2_test)
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        
        st.markdown("---")
        st.markdown("#### Baseline ROC Curves (Historical Test Set)")
        fig_roc_base, ax_roc_base = plt.subplots(figsize=(7, 5))
        colors = ["#4299E1", "#48BB78", "#9F7AEA", "#ED8936"]
        for i, (name, ev) in enumerate(st.session_state.ev_d1_test.items()):
            fpr, tpr, auc_v = ModelPipeline.roc_curve_data(st.session_state.y_h_test, ev["y_score"])
            if fpr is not None:
                ax_roc_base.plot(fpr, tpr, label=f"{name} (AUC={auc_v:.3f})", color=colors[i % len(colors)], linewidth=2)
        ax_roc_base.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax_roc_base.set_xlabel("False Positive Rate", fontweight='bold')
        ax_roc_base.set_ylabel("True Positive Rate", fontweight='bold')
        ax_roc_base.legend(loc="lower right", frameon=False)
        ax_roc_base.spines['top'].set_visible(False)
        ax_roc_base.spines['right'].set_visible(False)
        st.pyplot(fig_roc_base)
        plt.close(fig_roc_base)
        
        # Feature Diagnostics Expander
        with st.expander("Model Interpretability & Feature Significance"):
            c_fp1, c_fp2 = st.columns(2)
            with c_fp1:
                st.markdown("**Decision Tree Gini Importance**")
                imp = st.session_state.model_pipeline.decision_tree_importance(fn)
                imp_df = pd.DataFrame(imp, columns=["feature", "importance"])
                fig4, ax = plt.subplots(figsize=(6, 5))
                ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#4299E1")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig4)
                plt.close(fig4)
            with c_fp2:
                st.markdown("**Linear SVM Learned Support Vector Constraints**")
                svm_w = st.session_state.model_pipeline.svm_linear_weights(fn, top_k=18)
                if svm_w:
                    sw_df = pd.DataFrame(svm_w, columns=["feature", "|coef|"])
                    figs, axs = plt.subplots(figsize=(6, 5))
                    axs.barh(sw_df["feature"][::-1], sw_df["|coef|"][::-1], color="#3182CE")
                    axs.spines['top'].set_visible(False)
                    axs.spines['right'].set_visible(False)
                    st.pyplot(figs)
                    plt.close(figs)

            st.markdown("---")
            c_fp3, c_fp4 = st.columns(2)
            with c_fp3:
                st.markdown("**MLP Permutation Importance**")
                # Using a placeholder/spinner since it could take a few seconds
                with st.spinner("Computing MLP permutation importance..."):
                    mlp_w = st.session_state.model_pipeline.mlp_permutation_importance(
                        st.session_state.X_h_test, st.session_state.y_h_test, fn, n_repeats=5
                    )
                    if mlp_w:
                        mw_df = pd.DataFrame(mlp_w, columns=["feature", "importance"])
                        figm, axm = plt.subplots(figsize=(6, 5))
                        axm.barh(mw_df["feature"][::-1], mw_df["importance"][::-1], color="#9F7AEA")
                        axm.spines['top'].set_visible(False)
                        axm.spines['right'].set_visible(False)
                        st.pyplot(figm)
                        plt.close(figm)
                        
            with c_fp4:
                st.markdown("**Decision Tree Visualization**")
                from sklearn.tree import plot_tree
                fig_tree, ax_tree = plt.subplots(figsize=(12, 7), dpi=120)
                # Force white background so text is readable in both light & dark mode
                fig_tree.patch.set_facecolor('white')
                ax_tree.set_facecolor('white')
                dt_model = st.session_state.model_pipeline.models.get("Decision Tree")
                if dt_model:
                    plot_tree(
                        dt_model, max_depth=3, feature_names=fn,
                        filled=True, rounded=True, fontsize=8,
                        ax=ax_tree, impurity=True, proportion=False,
                    )
                    # Override every text element to dark color for readability
                    for text_obj in ax_tree.texts:
                        text_obj.set_color('#1a1a1a')
                        text_obj.set_fontweight('bold')
                    st.pyplot(fig_tree)
                    plt.close(fig_tree)
    else:
        st.info("Awaiting execution of pipeline... Train models to display analytics.")

# --- TAB 3: CONTINUAL LEARNING ---
with tab3:
    st.markdown("### Adaptive Intelligence & Continual Learning")
    
    if not st.session_state.get("models_trained"):
        st.warning("⚠️ Baseline models are required before executing continuous adaptation algorithms.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("""
            **Adaptation Strategy:**
            * Models iteratively absorb real-world data patterns from the *Current Training Cohort*.
            * Preserves historical representation while tuning operational threshold probabilities.
            """)
            if st.button("🚀 Execute Continual Adaptation Protocols", type="primary", use_container_width=True):
                with st.spinner("Initiating Partial fit tuning & recalibration..."):
                    pipe = st.session_state.model_pipeline
                    pipe.continual_learning_update(
                        st.session_state.X_c_train, st.session_state.y_c_train,
                        st.session_state.X_h_train, st.session_state.y_h_train,
                    )
                    st.session_state.ev_d2_after = pipe.evaluate_models(
                        st.session_state.X_c_test, st.session_state.y_c_test, "Current Cohort (Post-CL)"
                    )
        
        if st.session_state.get("ev_d2_after"):
            st.markdown("---")
            st.markdown("#### ROC Diagnostics Post-Adaptation")
            fig_roc_cl = _plot_rocs_before_after(
                st.session_state.ev_d2_test,
                st.session_state.ev_d2_after,
                st.session_state.y_c_test,
            )
            st.pyplot(fig_roc_cl)
            plt.close(fig_roc_cl)
            
            st.markdown("#### Core Operational Metrics Comparison")
            c_f1, c_acc = st.columns(2)
            with c_f1:
                fig_f1_shift = _plot_metric_grouped_bars("F1-Score", st.session_state.ev_d1_test, st.session_state.ev_d2_test, st.session_state.ev_d2_after)
                st.pyplot(fig_f1_shift)
                plt.close(fig_f1_shift)
            with c_acc:
                fig_acc_shift = _plot_metric_grouped_bars("ROC-AUC", st.session_state.ev_d1_test, st.session_state.ev_d2_test, st.session_state.ev_d2_after)
                st.pyplot(fig_acc_shift)
                plt.close(fig_acc_shift)
                
            c_prec, c_rec = st.columns(2)
            with c_prec:
                fig_prec_shift = _plot_metric_grouped_bars("Precision", st.session_state.ev_d1_test, st.session_state.ev_d2_test, st.session_state.ev_d2_after)
                st.pyplot(fig_prec_shift)
                plt.close(fig_prec_shift)
            with c_rec:
                fig_rec_shift = _plot_metric_grouped_bars("Recall", st.session_state.ev_d1_test, st.session_state.ev_d2_test, st.session_state.ev_d2_after)
                st.pyplot(fig_rec_shift)
                plt.close(fig_rec_shift)
                
            st.markdown("---")
            st.markdown("#### Automated Key Insights")
            
            f1_base = st.session_state.ev_d1_test["MLP"]["metrics"]["F1-Score"]
            f1_drift = st.session_state.ev_d2_test["MLP"]["metrics"]["F1-Score"]
            f1_cl = st.session_state.ev_d2_after["MLP"]["metrics"]["F1-Score"]
            
            if f1_drift < f1_base:
                st.warning(f"**Temporal drift detected:** MLP F1-Score reduced by {f1_base - f1_drift:.3f} on the current cohort.")
            else:
                st.success("**Stable performance:** Models proved resilient to temporal drift on current cohort.")
                
            if f1_cl > f1_drift:
                st.info(f"**Continual learning successful:** Retuning improved MLP F1-Score by {f1_cl - f1_drift:.3f}.")
            else:
                st.info("**Continual learning stabilized:** Retuning did not significantly alter the primary performance metric, suggesting the baseline representation remains optimal or the drift was not severe.")
