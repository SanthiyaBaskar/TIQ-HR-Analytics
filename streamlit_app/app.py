
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
st.set_page_config(page_title="T-IQ Preview", layout="wide")

# data paths (repo relative)
BASE = Path(__file__).resolve().parents[1] / "data"
EMP_SAMPLE = BASE / "employees_sample.csv"
EMP_CLEAN = BASE / "employees_clean.csv"
JOBS = BASE / "jobs.csv"

def load_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except:
            return pd.read_csv(path, encoding='latin1', errors='replace')
    return pd.DataFrame()

st.title("T-IQ — Preview App (Repository Bundle)")
st.sidebar.info("This is a bundle for demo. Use the 'data' folder in repo to run the app.")

df = load_csv(EMP_SAMPLE)
if df.empty:
    df = load_csv(EMP_CLEAN)
if df.empty:
    st.error("No data found in repo/data. Please add employees_sample.csv or employees_clean.csv")
else:
    st.header("Employees sample")
    st.dataframe(df.head(200))

# show dashboards images if present
dash_dir = Path(__file__).resolve().parents[1] / "dashboards"
if dash_dir.exists():
    imgs = list(dash_dir.glob("*.png"))[:10]
    for img in imgs:
        st.image(str(img), caption=img.name, use_column_width=True)
