# import matplotlib.pyplot as plt
# import numpy as np

# # %matplotlib inline 表示だとアニメーションしない
# # %matplotlib

# # 描画領域を取得
# fig, ax = plt.subplots(1, 1)

# # y軸方向の描画幅を指定
# ax.set_ylim((-1.1, 1.1))

# # x軸:時刻
# x = np.arange(0, 100, 0.5)

# # 周波数を高くしていく
# for Hz in np.arange(0.1, 10.1, 0.01):
#   # sin波を取得
#   y = np.sin(2.0 * np.pi * (x * Hz) / 100)
#   # グラフを描画する
#   line, = ax.plot(x, y, color='blue')
#   # 次の描画まで0.01秒待つ
#   plt.pause(0.01)
#   # グラフをクリア
#   line.remove()
# -*- coding: utf-8 -*-
"""
matplotlibでリアルタイムプロットする例

無限にsin関数をplotし続ける
"""
from __future__ import unicode_literals, print_function

import numpy as np
import matplotlib.pyplot as plt


def pause_plot():
    fig, ax = plt.subplots(1, 1)
    x = np.arange(-np.pi, np.pi, 0.1)
    y = np.sin(x)
    # 初期化的に一度plotしなければならない
    # そのときplotしたオブジェクトを受け取る受け取る必要がある．
    # listが返ってくるので，注意
    lines, = ax.plot(x, y)

    # ここから無限にplotする
    while True:
        # plotデータの更新
        x += 0.1
        y = np.sin(x)

        # 描画データを更新するときにplot関数を使うと
        # lineオブジェクトが都度増えてしまうので，注意．
        #
        # 一番楽なのは上記で受け取ったlinesに対して
        # set_data()メソッドで描画データを更新する方法．
        lines.set_data(x, y)

        # set_data()を使うと軸とかは自動設定されないっぽいので，
        # 今回の例だとあっという間にsinカーブが描画範囲からいなくなる．
        # そのためx軸の範囲は適宜修正してやる必要がある．
        ax.set_xlim((x.min(), x.max()))

        # 一番のポイント
        # - plt.show() ブロッキングされてリアルタイムに描写できない
        # - plt.ion() + plt.draw() グラフウインドウが固まってプログラムが止まるから使えない
        # ----> plt.pause(interval) これを使う!!! 引数はsleep時間
        plt.pause(.01)

if __name__ == "__main__":
    pause_plot()