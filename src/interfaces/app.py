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

# --- Helper Functions ---
@st.cache_data
def preprocess_data(df, target_column):
    """A helper function to preprocess the data for the model."""
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
    """A helper function to convert a DataFrame to a CSV for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- Page Configuration ---
st.set_page_config(page_title="DepoisonAI", page_icon="🛡️", layout="wide")
st.title("🛡️ DepoisonAI: Dataset Security Scanner")

# --- VERSION CHECK ---
st.success("✅ App Code Version: FINAL")

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload a CSV file to begin analysis", type=['csv'])

if uploaded_file is not None:
    # --- FIX: Only clear state if it's a NEW file ---
    if 'file_name' not in st.session_state or st.session_state.file_name != uploaded_file.name:
        for key in ['df', 'suspicious_indices', 'cleaned_df', 'file_name']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.file_name = uploaded_file.name # Store the name of the current file
    # --------------------------------------------------

if 'df' in st.session_state:
    st.dataframe(st.session_state.df.head(), use_container_width=True)
    st.divider()
    
    target_column = st.selectbox("1. Select the target column to analyze", st.session_state.df.columns)
    
    st.divider()

    if st.button("2. Run Poison Detection on Full Dataset", type="primary"):
        with st.spinner("Analyzing full dataset... This may take several minutes for large files."):
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
        st.success(f"**Analysis complete! Found {len(st.session_state.suspicious_indices)} potentially poisoned samples.**")
        
        st.divider()
        st.header("3. Depoison and Download")
        
        if st.button("Remove Suspicious Samples"):
            with st.spinner("Creating cleaned dataset..."):
                cleaned_df, _ = filter_samples(st.session_state.df, st.session_state.df[target_column], st.session_state.suspicious_indices)
                st.session_state.cleaned_df = cleaned_df

        if 'cleaned_df' in st.session_state:
            st.info(f"Removed {len(st.session_state.suspicious_indices)} samples. The new dataset has {len(st.session_state.cleaned_df)} rows.")
            st.write("Preview of Cleaned Data:")
            st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)
            
            csv_data = convert_df_to_csv(st.session_state.cleaned_df)
            st.download_button(
               label="📥 Download Cleaned CSV",
               data=csv_data,
               file_name="cleaned_data.csv",
               mime="text/csv"
            )