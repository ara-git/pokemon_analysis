#!/usr/bin/env python
# coding: utf-8

import cv2
import datetime
import time
from pathlib import Path
import win32gui
import win32ui
import win32con
import numpy as np
import cv2
import os
from pywinauto.application import Application
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img, save_img
import re
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import Adam



#ディレクトリの変更
os.chdir(r"C:\Users\ara-d\pokemon_analisis")

#参考：「[Python]マルチモニター環境でのウィンドウキャプチャ」
#https://qiita.com/danupo/items/e196e0e07e704796cd42

##アクティブなウィンドウを探して画像配列（numpy array)を返す様な関数を作る
def WindowCapture(window_name: str, bgr2rgb: bool = False):
    # 現在アクティブなウィンドウ名を探す
    process_list = []

    def callback(handle, _):
        process_list.append(win32gui.GetWindowText(handle))
    win32gui.EnumWindows(callback, None)

    #print(process_list)

    # ターゲットウィンドウ名を探す
    for process_name in process_list:
        if window_name in process_name:
            hnd = win32gui.FindWindow(None, process_name)
            #print("found") #見つかったら出力
            break
    else:
        # 見つからなかったら画面全体を取得
        #print("not found") #見つからなかったら出力
        hnd = win32gui.GetDesktopWindow()

    # ウィンドウサイズ取得
    x0, y0, x1, y1 = win32gui.GetWindowRect(hnd)
    width = x1 - x0
    height = y1 - y0
    # ウィンドウのデバイスコンテキスト取得
    windc = win32gui.GetWindowDC(hnd)
    srcdc = win32ui.CreateDCFromHandle(windc)
    memdc = srcdc.CreateCompatibleDC()
    # デバイスコンテキストからピクセル情報コピー, bmp化
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

    # bmpの書き出し
    if bgr2rgb is True:
        img = np.frombuffer(bmp.GetBitmapBits(True), np.uint8).reshape(height, width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_bgr2rgb)
    else:
        img = np.fromstring(bmp.GetBitmapBits(True), np.uint8).reshape(height, width, 4)

    # 後片付け
    # srcdc.DeleteDC()
    memdc.DeleteDC()
    # win32gui.ReleaseDC(hnd, windc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


##############ここは一回だけでいい###################
'''
#"Select Time"だけ切り取る
##やってみる
Select = img[0:80,0:400]
cv2.imshow("",Select)
cv2.waitKey(0)
cv2.destroyAllWindows()

#############################

##画像データを保存
np.save(
    "imgdata" + "_" + "SelectTime",  # ファイル名
    Select # 保存したいオブジェクト
)
'''

#保存していた"SelectTime"データを解凍する。
#SelectTime = np.load(file="imgdata_SelectTime.npy")

'''
cv2.imshow("",SelectTime)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#画像をとりあえずキャプチャする（全体）
#img = WindowCapture("PotPlayer")

'''
cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''



#相手のポケモンの画像を抽出する関数を作る。
def capture_opponent(img):
    a,b,c,d = 255, 961, 175, 225
    #相手ポケモンの画像だけ抽出
    part = img[a:b,c:d]
    
    #一体ずつポケモンの画像を抽出
    poke = []
    for i in range(6):
        new_poke = part[int((b - a)/6) * i + 20:int((b - a)/6) * (i + 1) - 20]
        poke.append(new_poke)
    
    poke = np.array(poke)
    return poke[:,:,:,0:3]


#個別ポケモン画像データ(numpy配列)を保存する関数を作る
##入力は画像データが入っている（６体分）リスト
def save_poke(Img_array):    
    ##今日の日付
    time_ = str(datetime.datetime.now())
    time_ = time_[:19].replace(":", "-").replace(" ", "-") #ファイル名が保存出来るように調整する。

    #ディレクトリを直下の"poke_images"に移動する。
    os.chdir(r"C:\Users\ara-d\pokemon_analisis\poke_figs_未処理")

    #画像を保存する。
    for i in range(len(Img_array)):     
        cv2.imwrite("pokedata" + "_" + time_ + "_" + str(i) + ".jpg", Img_array[i])

        '''
        np.save(
            "pokedata" + "_" + time_ + "_" + str(i),  # ファイル名
            Img_array[i] # 保存したいオブジェクト
        )  
        '''


#二つの画像データの一致率を計算する関数を作る。
def calc_accuracy(image1, image2): 
    A = image1 == image2
    A = np.reshape(A, (1,-1))
    accuracy = sum(A[0])/len(A[0])
    
    return accuracy


#保存していたポケモンの名前ファイルを呼び起こす
##引数：ファイルがある場所, ファイルの形式（基本はcsv）、出力：ポケモンの名前numpy配列
def call_name_file(path, use_csv = True):
    dir_ = Path(path)

    #csvを読み込む処理（こちらをメインで使う）
    if use_csv:
        #ファイル名を持ってくる
        #ここ↓、複数csvがあるなら変更するべき（for文で回して、dfを結合する）
        poke_names_FileName = sorted(dir_.glob("*.csv"))
        Names = pd.read_csv(str(poke_names_FileName[0]), header = None, encoding = "shift_jis")

    #numpy arrayを読み込む処理（もう使わない）
    else:
        #ファイル名を持ってくる
        poke_names_FileName = sorted(dir_.glob("*.npy"))

        ##名前データを呼び起こす
        Names = []
        for i in range(len(poke_names_FileName)):
            new = list(np.load(file=str(poke_names_FileName[i])))
            ##データを解凍する。
            Names.extend(new)
    
    return np.array(Names)

#ファイルソートの手法を定める関数（下のcall_img_fileで用いる）
def how_to_sort(val):
    val = str(val)
    val = re.search(r'\d+', val).group()
    #print(val)
    return int(val)

#保存していたポケモンの画像ファイルを呼び起こす
##引数：ファイルがある場所、use_jpg:読み込むファイルがjpgか否か（defaultはTrue）
##出力：画像のnumpy配列
def call_img_file(path, use_jpg = True):
    #ディレクトリを指定
    dir_ = Path(path)
    
    #jpgを読み込む
    if use_jpg:
        #ファイル名を持ってくる
        poke_images_FileName = sorted(dir_.glob("*.jpg"), key = how_to_sort)
        #poke_images_FileName = list(dir_.glob("*.jpg"))

        Image_data = []

        for i in range(len(poke_images_FileName)):
            ##データを解凍する。
            poke_image_train = cv2.imread(str(poke_images_FileName[i]))
            
            Image_data.append(poke_image_train)
            
        Image_data = np.array(Image_data)
            
    #numpy arrayを読み込む
    else:
        #ファイル名を持ってくる
        poke_images_FileName = sorted(dir_.glob("*.npy"))

        Image_data = []

        for i in range(len(poke_images_FileName)):
            ##データを解凍する。
            poke_image_train = np.load(file=str(poke_images_FileName[i]))
            
            Image_data.append(poke_image_train)
            #print("image:", poke_image_train)
        
        Image_data = np.array(Image_data)
        #何故か色データの次元が4なので、3に直しておく
        Image_data = Image_data[:,:,:,0:3]

    return Image_data

#keras用にBGRをRGBに変換する
##入力：Image_data: cv2で読み込んだ画像ファイルのnumpyarray（shape: 画像の数 × 縦 × 横 × 3）
def convert_image_BGR_to_RGB(Image_data):
    Converted_Image_data = []
    for img in Image_data:
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Converted_Image_data.append(im_rgb)

    return np.array(Converted_Image_data)

'''
width_shift_range = 0.1
height_shift_range = 0.1
'''

def make_additional_images(Image_data, datagen, num_of_additional_images, save = False):
    #画像をkeras用にRGBに変換する
    converted_Image_data = convert_image_BGR_to_RGB(Image_data)
    
    #オリジナルのデータの数だけイテレート
    for i in range(len(converted_Image_data)):
        #datagenのflow関数に合うように、reshapeする。（形だけ変える）
        x = converted_Image_data[i]
        x = x.reshape((1,) + x.shape)
        g = datagen.flow(x, batch_size=1)
        
        #（オリジナル一枚につき）新たに生成する画像の分だけイテレート
        for j in range(num_of_additional_images):
            batches = g.next()
            # 画像として表示するため、４次元から3次元データにし、配列から画像にする。
            gen_img = array_to_img(batches[0])

            #save変数がTrueの時だけ保存する（事故防止）
            if save:
                save_img(".\poke_figs\shifted\shifted_" + str(i) + "," + str(j) +".jpg", gen_img)


#名前の配列からone-hot行列を作成する関数
##pandasを用いて、その後numpyに戻す
##参考：https://qiita.com/nomuyoshi/items/c8127787c4ce320729da
##入力 Names:名前データ（教師データ）のnumpy配列、num_of_additional_images:カサ増しした学習画像データの数（オリジナル一枚に対して）
##出力 one_hot_label_original：オリジナルデータに関するone-hotラベル、one_hot_label_shifted:カサ増しデータに対するone-hotラベル
def make_one_hot_data(Names, num_of_additional_images, csv_save = True):
    ##まず、オリジナルデータに関してone-hotラベルを作成する
    df_one_hot_encoded = pd.get_dummies(Names.reshape(len(Names)))
    one_hot_label_original = np.array(df_one_hot_encoded)

    #one-hotと名前リストの対応表csvを保存する   
    if csv_save == True:
        pd.DataFrame(list(df_one_hot_encoded.columns)).to_csv("Name_relation_with_one-hot.csv", encoding= "shift_jis", index = False, header = None)
    
    ##次に、加工したデータに対してone-hotラベルを作成する
    one_hot_label_shifted = []
    for i in one_hot_label_original:
        for j in range(num_of_additional_images):
            one_hot_label_shifted.append(list(i))
    one_hot_label_shifted = np.array(one_hot_label_shifted)    

    return one_hot_label_original, one_hot_label_shifted

# CNNモデルを定義して返却する関数
def get_model(in_shape, num_classes):

  # 特徴量抽出
  model = Sequential()
  model.add(Conv2D(32,3,input_shape=in_shape))  # 畳み込みフィルタ層
  model.add(Activation('relu'))                 # 最適化関数
  model.add(Conv2D(32,3))
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2)))         # プーリング層
  model.add(Conv2D(64,3))
  model.add(Activation('relu'))
  model.add(MaxPool2D(pool_size=(2,2)))

  # 特徴量に基づいた分類
  model.add(Flatten())                          # 全結合層入力のためのデータの一次元化
  model.add(Dense(1024))                        # 全結合層
  model.add(Activation('relu'))                 # 最適化関数
  model.add(Dropout(0.5))                       # ドロップアウト層
  model.add(Dense(num_classes, activation='softmax'))  # 出力層

  # モデルのコンパイル
  adam = Adam(lr=1e-4)
  model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy"])
  
  return model

#入力：（入力画像データ（numpy array)、名前データ（numpy array、一次元)、train dataがある場所）
def judge_poke(input_image, model, name_onehot_relation): 
    #スケーリング
    input_image = input_image / 255

    #reshapeする
    ##サイズを保存
    shape0 = input_image.shape[0]
    shape1 = input_image.shape[1]
    shape2 = input_image.shape[2]
    input_image = input_image.reshape(-1, shape0, shape1, shape2)
    
    #print(input_image.shape)

    #予測を行う
    pre = model.predict([input_image])[0]
    #予測したラベル
    idx = pre.argmax()
    prob = pre[idx]

    #出力　idx：予測ラベル、prob:確率

    return name_onehot_relation[idx][0], prob

####################メイン部分#####################
####画像を自動でキャプチャ、読み込み画像判別&学習用に保存

def main(model, name_onehot_relation, SelectTime, battle_limit = 120, waiting_limit = 900):
    count = 0
    while True:
        #画像をキャプチャする。
        img = WindowCapture("PotPlayer") # 部分一致
        if (img[0:80,0:400] == SelectTime).all():
            print("Battle Start")
            #画像を出力
            #cv2.imshow("", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            #個別のポケモンの画像を抽出する。
            poke = capture_opponent(img)
            
            #ここの部分は判別用
            for individual_image in poke:
                estimated_poke_name, prob = judge_poke(individual_image, model, name_onehot_relation)
                print(estimated_poke_name, "prob=", prob)

            
            ##個別ポケモン画像データを保存する。
            save_poke(poke)
            
            count = 0 #カウントをリフレッシュする。
            
            #対戦時間中(battle_limit中)は休憩
            for i in range(battle_limit // 30):
                print("Battle Time Remain:" + str(battle_limit - 30 * i)) #30秒毎に警告
                time.sleep(30) 
            else:
                print("Rest was finished. Count starts again.")

        if count % 30 == 0:
            print("Waiting Time Remain:" + str(waiting_limit - count)) #30秒毎に警告
        if count >= waiting_limit: #一定時間（待機時間がwaiting_limitを超えたらbreak、放置対策）
            break
            
        #１秒経過させる。
        time.sleep(1)        
        count += 1