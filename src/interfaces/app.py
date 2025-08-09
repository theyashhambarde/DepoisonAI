import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# --- Path Correction ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Backend Imports ---
from src.poison_detection.tabular_detectors import detect_label_outliers_lof
from src.depoisoning_methods.tabular_cleaners import filter_samples
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Page Configuration ---
st.set_page_config(
    page_title="DepoisonAI | Security Scanner",
    page_icon="🛡️",
    layout="wide"
)

# --- Custom CSS for a futuristic look ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
    }
    .stButton>button {
        border: 2px solid #4A90E2;
        border-radius: 20px;
        color: #4A90E2;
        background-color: transparent;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        border-color: #FFFFFF;
        color: #FFFFFF;
    }
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem 10rem; /* Add padding to bottom */
    }
    .st-emotion-cache-1avcm0n {
        background: rgba(38, 39, 48, 0.4);
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def preprocess_data(df, target_column):
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(with_mean=False), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        X_processed = preprocessor.fit_transform(X)
        return X_processed, y
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None, None

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Header and Title ---
st.title("🛡️ DepoisonAI: Dataset Security Scanner")
st.markdown("An advanced tool to detect and neutralize data poisoning threats in your datasets.")

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload a CSV file to begin analysis", type=['csv'])

if uploaded_file is not None:
    if 'file_name' not in st.session_state or st.session_state.file_name != uploaded_file.name:
        for key in ['df', 'suspicious_indices', 'cleaned_df', 'file_name']:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.file_name = uploaded_file.name

if 'df' in st.session_state:
    st.info("Data Preview:")
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    st.divider()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        target_column = st.selectbox("Select Target Column", st.session_state.df.columns)
    
    with col2:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("🚀 Run Poison Detection", use_container_width=True):
            with st.spinner("Analyzing full dataset... This may take several minutes."):
                df_to_analyze = st.session_state.df
                X_processed, y = preprocess_data(df_to_analyze, target_column)
                if X_processed is not None:
                    suspicious_indices_relative = detect_label_outliers_lof(X_processed, y)
                    if suspicious_indices_relative is not None and len(suspicious_indices_relative) > 0:
                        int_indices = np.array(suspicious_indices_relative).astype(int)
                        st.session_state.suspicious_indices = df_to_analyze.index[int_indices].tolist()
                    else:
                        st.session_state.suspicious_indices = []
    
    if 'suspicious_indices' in st.session_state:
        st.divider()
        st.subheader("Analysis Results")
        
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric(label="Total Rows Scanned", value=f"{len(st.session_state.df):,}")
            st.metric(label="Suspicious Samples Found", value=len(st.session_state.suspicious_indices))
            if st.button("Cleanse Dataset"):
                with st.spinner("Removing suspicious samples..."):
                    cleaned_df, _ = filter_samples(st.session_state.df, st.session_state.df[target_column], st.session_state.suspicious_indices)
                    st.session_state.cleaned_df = cleaned_df
        with result_col2:
            st.write("Indices of Suspicious Samples:")
            st.dataframe(st.session_state.suspicious_indices, height=210)

    if 'cleaned_df' in st.session_state:
        st.divider()
        st.subheader("Cleansed Data")
        st.info(f"Removed {len(st.session_state.suspicious_indices)} samples. The new dataset has {len(st.session_state.cleaned_df)} rows.")
        st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
        
        csv_data = convert_df_to_csv(st.session_state.cleaned_df)
        st.download_button(
           label="📥 Download Cleaned CSV",
           data=csv_data,
           file_name="cleaned_data.csv",
           mime="text/csv",
           use_container_width=True
        )

# --- Footer ---
st.divider()
st.markdown("<p style='text-align: center; color: grey;'>Created by Yash Hambarde</p>", unsafe_allow_html=True)
