import pandas as pd
import streamlit as st

def show_training_history(MODELS_DIR, TRAINING_CSV_SUFFIX):
     return pd.read_csv(MODELS_DIR+TRAINING_CSV_SUFFIX)
    
def load_models():
     # dataframe all models
     # select the unique ones
     pass

def get_training_dataframe(path_str):
    training_df = pd.read_csv(path_str)
    st.dataframe(training_df)