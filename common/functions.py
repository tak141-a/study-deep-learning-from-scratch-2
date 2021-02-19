import numpy as np


def softmax(x):
    """
    Softmax関数

    Params:
        x (ndarray): 入力

    Return:
        ndarray: Softmax関数適用結果
    """
    # 最大値や和の出し方が違うため次元数で処理分け
    if x.ndim == 2:  # 次元数が2
        x = x - x.max(axis=1, keepdims=True)  # 行ごとの最大値を引く(オーバーフロー防止)
        x = np.exp(x)  # 分子
        x /= x.sum(axis=1, keepdims=True)  # 分母(行方向の和)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    """
    交差エントロピー誤差

    Params:
        y (ndarray): 入力(softmaxの出力)
        t (ndarray): 教師データ

    Return:
        ndarray: 交差エントロピー誤差適用結果
    """
    # データ一つあたりの交差エントロピー誤差を求める場合、データの形状を整形
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    # np.arange(batch_size) -> 0からbatch_size-1までの配列を生成([0, 1, 2, 3]みたいな)
    # t -> 正解ラベル([2, 7, 0, 9]みたいな)
    # y[np.arange(batch_size), t] -> 各データの正解ラベルに対応するニューラルネットワークの出力
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
