import streamlit as st
import pandas as pd


def run():
    st.subheader("Under construction")

    st.header("Result of Battles")
    battle_result_df = pd.read_csv("battle_result.csv", encoding="shift_jis")

    st.table(battle_result_df)
