#学習データをnumpy配列からjpgファイルに変更するために作成したコード
#もう使う必要はない

import os
import numpy as np
import cv2
from pathlib import Path
import functions
import datetime
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import pandas as pd

#ディレクトリの変更
os.chdir(r"C:\Users\ara-d\pokemon_analisis")

#保存していた"SelectTime"データを解凍する。
SelectTime = np.load(file="imgdata_SelectTime.npy")
print(SelectTime.shape)

#保存していた名前データを解凍する。
##ディレクトリを指定し、ファイル名を取得する。
name_file_path = r"C:\Users\ara-d\pokemon_analisis\poke_names"
Names = functions.call_name_file(name_file_path, use_csv=False)
print("Names:", Names)

#ここから画像ファイル（学習用）
#trainデータ（画像情報が入ったnumpy array）がある場所を指定する。
Img_path = r"C:\Users\ara-d\pokemon_analisis\poke_images_train"
Image_data = functions.call_img_file(Img_path)
print("Images:", Image_data)

#####################
#ここから行いたい処理（一度限り）

'''
#名前データをnumpy.arrayからcsvファイルに変換する
Names_df = pd.DataFrame(Names)
Names_df.to_csv(".\poke_names\poke_Names.csv", index = False, header = False, encoding = "shift_jis")

#画像ファイルをnumpy.arrayからjpgに変換して保存する
for i in range(len(Image_data)):
    cv2.imwrite(".\poke_figs\image" + str(i) + ".jpg", Image_data[i])
'''