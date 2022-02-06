import streamlit as st
import pandas as pd
import os


def run():
    # st.subheader("Under construction")
    st.header("Result of Battles")

    path = "./resource/battle_result"
    files = os.listdir(path)
    file_list = [f for f in files if os.path.isfile(os.path.join(path, f))]
    st.write(file_list)
    for file in file_list:
        battle_result_df = pd.read_csv(
            "resource/battle_result/" + file, encoding="shift_jis"
        )
        st.table(battle_result_df)
