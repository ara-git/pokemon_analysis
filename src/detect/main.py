# 既に計算してある重みを使って、画面のポケモンを分類する。

import numpy as np
import cv2

# import detect.function_detect as func
import function_detect as func_detect
import pandas as pd
import streamlit as st

if __name__ == "__main__":
    # ここよくないかもしれない
    # 画像サイズとクラス数を設定
    in_shape, num_classes = ((77, 50, 3), 112)

    # 重みを解凍する
    model = func_detect.get_model(in_shape, num_classes)
    model.load_weights("resource/intermediate/weight.hdf5")

    # 保存していたone-hotとポケモン名の対応表csvを解凍する
    name_onehot_relation = pd.read_csv(
        "resource/intermediate/Name_relation_with_one-hot.csv",
        encoding="shift_jis",
        header=None,
    )
    name_onehot_relation = np.array(name_onehot_relation)

    # 保存していた"SelectTime"データを解凍する。
    SelectTime = np.load(file="resource/intermediate/imgdata_SelectTime.npy")
    FinishBattle = np.load(file="resource/intermediate/imgdata_FinishBattle.npy")
    # st.write(FinishBattle)

    """
    cv2.imshow("", FinishBattle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # 実行する
    func_detect.main(
        model,
        name_onehot_relation,
        SelectTime,
        FinishBattle,
        battle_limit=120,
        waiting_limit=900,
    )
