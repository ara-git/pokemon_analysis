"""
共通して使う関数をここに置く
"""
import win32gui
import win32ui
import win32con
import cv2
import numpy as np

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
            # print("found")  # 見つかったら出力
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

