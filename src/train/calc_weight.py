from ast import Num
import os
from PIL.Image import Image
import numpy as np
import cv2
from pathlib import Path
from numpy.core.shape_base import stack
import function_train as functions
import datetime
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
    array_to_img,
    save_img,
)
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import EarlyStopping

# ディレクトリの変更
os.chdir(r"C:\Users\ara-d\pokemon_analisis")

# 画像確認用：
"""
cv2.imshow("",SelectTime)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

####################################
# 名前データの読み込み
Names = np.load(file=r".\poke_names\one_hot_label.npy")
# Names = functions.call_name_file(name_file_path, )
# print("Names:", Names)

# 画像データの読み込み
Img_path = r"C:\Users\ara-d\pokemon_analisis\poke_figs"
original_img = functions.call_img_file(Img_path, use_jpg=True)
# original_img.shape

Img_path = r"C:\Users\ara-d\pokemon_analisis\poke_figs\shifted"
shifted_img = functions.call_img_file(Img_path, use_jpg=True)
shifted_img.shape

Images = np.vstack([original_img, shifted_img])

# 値のスケールを調整する
scaled_Images = Images / 255
# scaled_Images


im_rows = Images.shape[1]  # 画像の縦サイズ（ピクセル）
im_cols = Images.shape[2]  # 画像の横サイズ（ピクセル）
im_color = Images.shape[3]  # 画像の色空間
in_shape = (im_rows, im_cols, im_color)
num_classes = Names.shape[1]  # 分類数


# 写真データを読み込み
x = scaled_Images
y = Names

# 読み込んだデータを三次元配列に変換
x = x.reshape(-1, im_rows, im_cols, im_color)

# 全てのデータを使って学習をする
x_train = x
y_train = y

# CNNモデルを取得
# EaelyStoppingの設定
early_stopping = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=2,)

model = functions.get_model(in_shape, num_classes)

# モデルの学習
hist = model.fit(
    x_train, y_train, batch_size=32, epochs=20, verbose=1, callbacks=[early_stopping]
)

# 学習したモデル（重み）を保存
model.save_weights("./weight.hdf5")

"""
# 実験：
# 学習用とテスト用に分けて学習を行う
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle = True)

# CNNモデルを取得
# EaelyStoppingの設定
early_stopping =  EarlyStopping(
                            monitor='val_loss',
                            min_delta=0.0,
                            patience=2,
)

model = functions.get_model(in_shape, num_classes)

# モデルの学習
#hist = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stopping])
hist = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, callbacks=[early_stopping])

#モデルを評価
#score = model.evaluate(x_test, y_test, verbose=1)
#print("正解率=", score[1], "loss=", score[0])

# 学習したモデルを保存
model.save_weights('./weight.hdf5')

#実験してみる
pre = model.predict(x_test)

Predicted_label = []

for prob in pre:
    predicted = prob.argmax()
    Predicted_label.append(predicted)

True_label = np.argmax(y_test, axis=1)
"""

