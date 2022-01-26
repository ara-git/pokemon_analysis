"""
画像を自動でキャプチャ、読み込み画像判別&学習用に保存
"""

import cv2
import datetime
import time
import win32gui
import win32ui
import win32con
import numpy as np
from keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
    array_to_img,
    save_img,
)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import Adam
import streamlit as st

##アクティブなウィンドウを探して画像配列（numpy array)を返す様な関数を作る
def WindowCapture(window_name: str, bgr2rgb: bool = False):
    # 現在アクティブなウィンドウ名を探す
    process_list = []

    def callback(handle, _):
        process_list.append(win32gui.GetWindowText(handle))

    win32gui.EnumWindows(callback, None)

    # print(process_list)

    # ターゲットウィンドウ名を探す
    for process_name in process_list:
        if window_name in process_name:
            hnd = win32gui.FindWindow(None, process_name)
            # print("found") #見つかったら出力
            break
    else:
        # 見つからなかったら画面全体を取得
        # print("not found") #見つからなかったら出力
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


# 相手のポケモンの画像を抽出する関数を作る。
def capture_opponent(img):
    a, b, c, d = 255, 961, 175, 225
    # 相手ポケモンの画像だけ抽出
    part = img[a:b, c:d]

    # 一体ずつポケモンの画像を抽出
    poke = []
    for i in range(6):
        new_poke = part[int((b - a) / 6) * i + 20 : int((b - a) / 6) * (i + 1) - 20]
        poke.append(new_poke)

    poke = np.array(poke)
    return poke[:, :, :, 0:3]


# 個別ポケモン画像データ(numpy配列)を保存する関数を作る
##入力は画像データが入っている（６体分）リスト
def save_poke(Img_array):
    ##今日の日付
    time_ = str(datetime.datetime.now())
    time_ = time_[:19].replace(":", "-").replace(" ", "-")  # ファイル名が保存出来るように調整する。

    # ディレクトリを直下の"poke_images"に移動する。
    # os.chdir(r"C:\Users\ara-d\pokemon_analisis\poke_figs_未処理")

    # 画像を保存する。
    for i in range(len(Img_array)):
        cv2.imwrite("pokedata" + "_" + time_ + "_" + str(i) + ".jpg", Img_array[i])

        """
        np.save(
            "pokedata" + "_" + time_ + "_" + str(i),  # ファイル名
            Img_array[i] # 保存したいオブジェクト
        )  
        """


# 入力：（入力画像データ（numpy array)、名前データ（numpy array、一次元)、train dataがある場所）
def judge_poke(input_image, model, name_onehot_relation):
    # スケーリング
    input_image = input_image / 255

    # reshapeする
    ##サイズを保存
    shape0 = input_image.shape[0]
    shape1 = input_image.shape[1]
    shape2 = input_image.shape[2]
    input_image = input_image.reshape(-1, shape0, shape1, shape2)

    # print(input_image.shape)

    # 予測を行う
    pre = model.predict([input_image])[0]
    # 予測したラベル
    idx = pre.argmax()
    prob = pre[idx]

    # 出力　idx：予測ラベル、prob:確率

    return name_onehot_relation[idx][0], prob


def get_model(in_shape, num_classes):

    # 特徴量抽出
    model = Sequential()
    model.add(Conv2D(32, 3, input_shape=in_shape))  # 畳み込みフィルタ層
    model.add(Activation("relu"))  # 最適化関数
    model.add(Conv2D(32, 3))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))  # プーリング層
    model.add(Conv2D(64, 3))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    # 特徴量に基づいた分類
    model.add(Flatten())  # 全結合層入力のためのデータの一次元化
    model.add(Dense(1024))  # 全結合層
    model.add(Activation("relu"))  # 最適化関数
    model.add(Dropout(0.5))  # ドロップアウト層
    model.add(Dense(num_classes, activation="softmax"))  # 出力層

    # モデルのコンパイル
    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def main(model, name_onehot_relation, SelectTime, battle_limit=120, waiting_limit=900):
    count = 0
    while True:
        # 画像をキャプチャする。
        img = WindowCapture("PotPlayer")  # 部分一致
        if (img[0:80, 0:400] == SelectTime).all():
            print("Battle Start")
            # 画像を出力
            # cv2.imshow("", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # 個別のポケモンの画像を抽出する。
            poke = capture_opponent(img)

            # ここの部分は判別用
            for individual_image in poke:
                estimated_poke_name, prob = judge_poke(
                    individual_image, model, name_onehot_relation
                )
                print(estimated_poke_name, "prob=", prob)

            ##個別ポケモン画像データを保存する。
            save_poke(poke)

            count = 0  # カウントをリフレッシュする。

            # 対戦時間中(battle_limit中)は休憩
            for i in range(battle_limit // 30):
                print("Battle Time Remain:" + str(battle_limit - 30 * i))  # 30秒毎に警告
                time.sleep(30)
            else:
                print("Rest was finished. Count starts again.")

        if count % 30 == 0:
            print("Waiting Time Remain:" + str(waiting_limit - count))  # 30秒毎に警告
            st.write(str(waiting_limit - count))
        if count >= waiting_limit:  # 一定時間（待機時間がwaiting_limitを超えたらbreak、放置対策）
            break

        # １秒経過させる。
        time.sleep(1)
        count += 1
