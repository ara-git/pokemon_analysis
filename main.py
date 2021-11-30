#既に計算してある重みを使って、画面のポケモンを分類する。

from ast import Num
import os
from PIL.Image import Image
import numpy as np
import cv2
from pathlib import Path
from numpy.core.shape_base import stack
import functions
import datetime
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img, save_img
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import EarlyStopping

#ディレクトリの変更
os.chdir(r"C:\Users\ara-d\pokemon_analisis")

#ここよくないかもしれない
#画像サイズとクラス数を設定
in_shape, num_classes = ((77, 50, 3), 112)

#重みを解凍する
model = functions.get_model(in_shape, num_classes)
model.load_weights('./weight.hdf5')

#保存していたone-hotとポケモン名の対応表csvを解凍する
name_onehot_relation = pd.read_csv("Name_relation_with_one-hot.csv", encoding="shift_jis", header = None)
name_onehot_relation = np.array(name_onehot_relation)

#保存していた"SelectTime"データを解凍する。
SelectTime = np.load(file="imgdata_SelectTime.npy")

#実行する
functions.main(model, name_onehot_relation, SelectTime, battle_limit = 120, waiting_limit = 900)