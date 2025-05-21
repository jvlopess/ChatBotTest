from pathlib import Path
import pandas as pd
import streamlit as st


def load_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_data(folder: str) -> pd.DataFrame:
    all_datasets = [
        load_file(file) for file in Path(folder).iterdir() if file.suffix == ".csv"
    ]
    df = pd.concat(all_datasets, ignore_index=True)
    return df
