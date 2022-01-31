"""
対戦画面の判定に必要な、SelectTimeが移った画像を保存する
"""

# from http.client import ImproperConnectionState

# from imp import IMP_HOOK
# import importlib
import cv2
import numpy as np
import sys
import os

# utilフォルダにあるfunction_commonファイルをインポートする
# モジュール探索パスを追加して、絶対インポートする（参考：https://note.nkmk.me/python-relative-import/）
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from util import function_common as func_com

# "Select Time"だけ切り取る
## 画面を取り込む
# img = func_com.WindowCapture("PotPlayer")  # 部分一致
img = func_com.WindowCapture("全画面プロジェクター")  # 部分一致
img = img[900:950, 700:1000]

## 見てみる
cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 画像データを保存
np.save(
    "resource/intermediate/imgdata" + "_" + "FinishBattle", img
)  # ファイル名  # 保存したいオブジェクト
