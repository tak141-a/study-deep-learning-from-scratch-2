import numpy as np


class Sigmoid:
    """
    Sigmoidレイヤ

    Attributes:
        params (list): パラメータ(重みやバイアスなど)
    """

    def __init__(self):
        self.params = []    # 学習するパラメータが存在しないため空のリストで初期化

    def forward(self, x):
        """順伝播

        Sigmoidレイヤの順伝播の結果を返す

        Args:
            x (float): 入力パラメータ

        Returns:
            float: 順伝播結果
        """
        return 1 / (1 + np.exp(-x))


class Affine:
    """
    Affineレイヤ

    Attributes:
        params (list): 重みとバイアス
    """

    def __init__(self, W, b):
        """
        Args:
            W (list): 重み
            b (list): バイアス
        """
        self.params = [W, b]

    def forward(self, x):
        """順伝播

        Affineレイヤの順伝播の結果を返す

        Args:
            x (list): 入力
        """
        W, b = self.params      # paramsからWとbを取り出し
        out = np.dot(x, W) + b  # xW + b
        return out
