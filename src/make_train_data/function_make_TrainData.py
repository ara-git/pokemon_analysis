#!/usr/bin/env python
# coding: utf-8

import cv2
from pathlib import Path
import numpy as np
import os
from pywinauto.application import Application
import pandas as pd
from keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
    array_to_img,
    save_img,
)


# ディレクトリの変更
os.chdir(r"C:\Users\ara-d\pokemon_analisis")

# 参考：「[Python]マルチモニター環境でのウィンドウキャプチャ」
# https://qiita.com/danupo/items/e196e0e07e704796cd42

# 二つの画像データの一致率を計算する関数を作る。
def calc_accuracy(image1, image2):
    A = image1 == image2
    A = np.reshape(A, (1, -1))
    accuracy = sum(A[0]) / len(A[0])

    return accuracy


# 保存していたポケモンの名前ファイルを呼び起こす
##引数：ファイルがある場所, ファイルの形式（基本はcsv）、出力：ポケモンの名前numpy配列
def call_name_file(path, use_csv=True):
    dir_ = Path(path)

    # csvを読み込む処理（こちらをメインで使う）
    if use_csv:
        # ファイル名を持ってくる
        # ここ↓、複数csvがあるなら変更するべき（for文で回して、dfを結合する）
        poke_names_FileName = sorted(dir_.glob("*.csv"))
        Names = pd.read_csv(
            str(poke_names_FileName[0]), header=None, encoding="shift_jis"
        )

    # numpy arrayを読み込む処理（もう使わない）
    else:
        # ファイル名を持ってくる
        poke_names_FileName = sorted(dir_.glob("*.npy"))

        ##名前データを呼び起こす
        Names = []
        for i in range(len(poke_names_FileName)):
            new = list(np.load(file=str(poke_names_FileName[i])))
            ##データを解凍する。
            Names.extend(new)

    return np.array(Names)


# keras用にBGRをRGBに変換する
##入力：Image_data: cv2で読み込んだ画像ファイルのnumpyarray（shape: 画像の数 × 縦 × 横 × 3）
def convert_image_BGR_to_RGB(Image_data):
    Converted_Image_data = []
    for img in Image_data:
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Converted_Image_data.append(im_rgb)

    return np.array(Converted_Image_data)


"""
width_shift_range = 0.1
height_shift_range = 0.1
"""


def make_additional_images(Image_data, datagen, num_of_additional_images, save=False):
    # 画像をkeras用にRGBに変換する
    converted_Image_data = convert_image_BGR_to_RGB(Image_data)

    # オリジナルのデータの数だけイテレート
    for i in range(len(converted_Image_data)):
        # datagenのflow関数に合うように、reshapeする。（形だけ変える）
        x = converted_Image_data[i]
        x = x.reshape((1,) + x.shape)
        g = datagen.flow(x, batch_size=1)

        # （オリジナル一枚につき）新たに生成する画像の分だけイテレート
        for j in range(num_of_additional_images):
            batches = g.next()
            # 画像として表示するため、４次元から3次元データにし、配列から画像にする。
            gen_img = array_to_img(batches[0])

            # save変数がTrueの時だけ保存する（事故防止）
            if save:
                save_img(
                    ".\poke_figs\shifted\shifted_" + str(i) + "," + str(j) + ".jpg",
                    gen_img,
                )


# 名前の配列からone-hot行列を作成する関数
##pandasを用いて、その後numpyに戻す
##参考：https://qiita.com/nomuyoshi/items/c8127787c4ce320729da
##入力 Names:名前データ（教師データ）のnumpy配列、num_of_additional_images:カサ増しした学習画像データの数（オリジナル一枚に対して）
##出力 one_hot_label_original：オリジナルデータに関するone-hotラベル、one_hot_label_shifted:カサ増しデータに対するone-hotラベル
def make_one_hot_data(Names, num_of_additional_images, csv_save=True):
    ##まず、オリジナルデータに関してone-hotラベルを作成する
    df_one_hot_encoded = pd.get_dummies(Names.reshape(len(Names)))
    one_hot_label_original = np.array(df_one_hot_encoded)

    # one-hotと名前リストの対応表csvを保存する
    if csv_save == True:
        pd.DataFrame(list(df_one_hot_encoded.columns)).to_csv(
            "Name_relation_with_one-hot.csv",
            encoding="shift_jis",
            index=False,
            header=None,
        )

    ##次に、加工したデータに対してone-hotラベルを作成する
    one_hot_label_shifted = []
    for i in one_hot_label_original:
        for j in range(num_of_additional_images):
            one_hot_label_shifted.append(list(i))
    one_hot_label_shifted = np.array(one_hot_label_shifted)

    return one_hot_label_original, one_hot_label_shifted
