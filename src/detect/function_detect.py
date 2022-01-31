"""
画像を自動でキャプチャ、読み込み画像判別&学習用に保存
"""

import cv2
import datetime
import time
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import Adam
import streamlit as st
import sys
import os

# utilフォルダにあるfunction_commonファイルをインポートする
# モジュール探索パスを追加して、絶対インポートする（参考：https://note.nkmk.me/python-relative-import/）
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util import function_common as func_com


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
        # 学習データを追加するために、ポケモン画像を保存する
        cv2.imwrite(
            "resource/poke_figs_outstanding/pokedata"
            + "_"
            + time_
            + "_"
            + str(i)
            + ".jpg",
            Img_array[i],
        )

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


def main(
    model,
    name_onehot_relation,
    SelectTime,
    FinishBattle,
    battle_limit=120,
    waiting_limit=900,
):
    st.title("Pokemon Battle Supporter")
    flag_in_battle = False

    while True:
        # 画像をキャプチャする。
        img = func_com.WindowCapture("全画面プロジェクター")  # 部分一致
        # img = func_com.WindowCapture("PotPlayer")  # 部分一致
        """
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        time.sleep(1)
        if not flag_in_battle:
            """
            バトルしていない
            """
            if (img[0:80, 0:400] == SelectTime).all():
                # 個別のポケモンの画像を抽出する。
                poke = capture_opponent(img)

                # ここの部分は判別用
                for individual_image in poke:
                    estimated_poke_name, prob = judge_poke(
                        individual_image, model, name_onehot_relation
                    )
                    st.write(estimated_poke_name, "prob=", prob)

                ##個別ポケモン画像データを保存する。
                save_poke(poke)
                time.sleep(30)

                flag_in_battle = True
        else:
            """
            バトル中
            """

            if (img[900:950, 700:1000] == FinishBattle).all():
                st.write("Battle Finish")
                flag_in_battle = False
            # count = 0  # カウントをリフレッシュする。

            # 対戦時間中(battle_limit中)は休憩
            # st.write("Battle Time Remain:" + str(battle_limit - 30 * i))  # 30秒毎に警告


"""


def main(model, name_onehot_relation, SelectTime, battle_limit=120, waiting_limit=900):
    count = 0
    while True:
        # 画像をキャプチャする。
        img = func_com.WindowCapture("全画面プロジェクター")  # 部分一致
        if (img[0:80, 0:400] == SelectTime).all():
            print("Battle Start")

            # 個別のポケモンの画像を抽出する。
            poke = capture_opponent(img)

            # ここの部分は判別用
            for individual_image in poke:
                estimated_poke_name, prob = judge_poke(
                    individual_image, model, name_onehot_relation
                )
                # print(estimated_poke_name, "prob=", prob)
                st.write(estimated_poke_name, "prob=", prob)

            ##個別ポケモン画像データを保存する。
            save_poke(poke)

            count = 0  # カウントをリフレッシュする。

            # 対戦時間中(battle_limit中)は休憩
            for i in range(battle_limit // 30):
                print("Battle Time Remain:" + str(battle_limit - 30 * i))  # 30秒毎に警告
                st.write("Battle Time Remain:" + str(battle_limit - 30 * i))  # 30秒毎に警告
                time.sleep(30)
            else:
                # print("Rest was finished. Count starts again.")
                st.write("Rest was finished. Count starts again.")

        if count % 30 == 0:
            # print("Waiting Time Remain:" + str(waiting_limit - count))  # 30秒毎に警告
            st.write(str(waiting_limit - count))
        if count >= waiting_limit:  # 一定時間（待機時間がwaiting_limitを超えたらbreak、放置対策）
            break

        # １秒経過させる。
        time.sleep(1)
        count += 1
"""

