import streamlit as st
import pandas as pd
import datetime


def run():
    st.header("Result of Battles")
    st.write(
        "注意！　ここを押すと'battle_result.csv'ファイルがリセットされます。（'battle_result.csv'に存在するデータは日付付きのファイルに移動します）"
    )
    if st.checkbox("reset_csv"):

        battle_result_df = pd.read_csv(
            "resource/battle_result/battle_result.csv", encoding="shift_jis"
        )
        battle_result_df.to_csv(
            "resource/battle_result/battle_result_"
            + str(datetime.date.today())
            + ".csv",
            encoding="shift_jis",
            index=False,
        )

        pd.DataFrame(columns=["date", 1, 2, 3, 4, 5, 6]).to_csv(
            "resource/battle_result/battle_result.csv",
            encoding="shift_jis",
            index=False,
        )

