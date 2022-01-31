import cv2
import datetime
import time
from pathlib import Path
import win32gui
import win32ui
import win32con
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
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import Adam

# ファイルソートの手法を定める関数（下のcall_img_fileで用いる）
def how_to_sort(val):
    val = str(val)
    val = re.search(r"\d+", val).group()
    # print(val)
    return int(val)


# 保存していたポケモンの画像ファイルを呼び起こす
##引数：ファイルがある場所、use_jpg:読み込むファイルがjpgか否か（defaultはTrue）
##出力：画像のnumpy配列
def call_img_file(path, use_jpg=True):
    # ディレクトリを指定
    dir_ = Path(path)

    # jpgを読み込む
    if use_jpg:
        # ファイル名を持ってくる
        poke_images_FileName = sorted(dir_.glob("*.jpg"), key=how_to_sort)
        # poke_images_FileName = list(dir_.glob("*.jpg"))

        Image_data = []

        for i in range(len(poke_images_FileName)):
            ##データを解凍する。
            poke_image_train = cv2.imread(str(poke_images_FileName[i]))

            Image_data.append(poke_image_train)

        Image_data = np.array(Image_data)

    # numpy arrayを読み込む
    else:
        # ファイル名を持ってくる
        poke_images_FileName = sorted(dir_.glob("*.npy"))

        Image_data = []

        for i in range(len(poke_images_FileName)):
            ##データを解凍する。
            poke_image_train = np.load(file=str(poke_images_FileName[i]))

            Image_data.append(poke_image_train)
            # print("image:", poke_image_train)

        Image_data = np.array(Image_data)
        # 何故か色データの次元が4なので、3に直しておく
        Image_data = Image_data[:, :, :, 0:3]

    return Image_data
