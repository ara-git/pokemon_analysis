# 既に計算してある重みを使って、画面のポケモンを分類する。

import numpy as np
from numpy.core.shape_base import stack
import function_detect as func
from keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
    array_to_img,
    save_img,
)
import pandas as pd
from sklearn.model_selection import train_test_split

# ディレクトリの変更
# os.chdir(r"C:\Users\ara-d\pokemon_analisis")

if __name__ == "__main__":
    # ここよくないかもしれない
    # 画像サイズとクラス数を設定
    in_shape, num_classes = ((77, 50, 3), 112)

    # 重みを解凍する
    model = func.get_model(in_shape, num_classes)
    model.load_weights("resource/interim/weight.hdf5")

    # 保存していたone-hotとポケモン名の対応表csvを解凍する
    name_onehot_relation = pd.read_csv(
        "resource/interim/Name_relation_with_one-hot.csv",
        encoding="shift_jis",
        header=None,
    )
    name_onehot_relation = np.array(name_onehot_relation)

    # 保存していた"SelectTime"データを解凍する。
    SelectTime = np.load(file="resource/interim/imgdata_SelectTime.npy")

    # 実行する
    func.main(
        model, name_onehot_relation, SelectTime, battle_limit=120, waiting_limit=900
    )
