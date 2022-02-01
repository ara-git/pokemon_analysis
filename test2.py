from PIL import Image
import pyocr
import pyocr.builders
import cv2
import numpy as np
import sys
import os

# utilフォルダにあるfunction_commonファイルをインポートする
# モジュール探索パスを追加して、絶対インポートする（参考：https://note.nkmk.me/python-relative-import/）
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# from util import function_common as func_com
from src.util import function_common as func_com


def render_doc_text(img):
    # ツール取得
    pyocr.tesseract.TESSERACT_CMD = "C:/Program Files/Tesseract-OCR/tesseract.exe"
    tools = pyocr.get_available_tools()
    tool = tools[0]

    # 画像取得
    img = Image.fromarray(img)

    # OCR
    builder = pyocr.builders.TextBuilder()
    result = tool.image_to_string(img, lang="jpn", builder=builder)

    # 結果から空白文字削除
    data_list = [text for text in result.split("\n") if text.strip()]
    data_list

    return data_list


# img = func_com.WindowCapture("PotPlayer")[60:100, 950:1100]
# img = func_com.WindowCapture("PotPlayer")[60:100, 1440:1600]

img = func_com.WindowCapture("PotPlayer")[870:1050, 100:970]
cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

res = render_doc_text(img)

print(res)
