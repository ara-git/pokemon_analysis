"""
メインのファイル

ページを分けて表示するために、このファイルを起点に他のファイルを呼び出していく
"""

import os
import pandas as pd
import streamlit as st
from detect import detect
from summarize import summarize
from reset_df import reset_csv

st.title("Pokemon Battle Supporter")

PAGES = {
    "Detect and support a battle": detect,
    "Summary": summarize,
    "reset_csv": reset_csv,
}
selection = st.sidebar.selectbox("Page", list(PAGES.keys()))

page = PAGES[selection]
page.run()
