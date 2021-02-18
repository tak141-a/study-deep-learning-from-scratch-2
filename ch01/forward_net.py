import numpy as np


class Sigmoid:
    """
    Sigmoidレイヤ

    Attributes:
        params (list): パラメータ(重みやバイアスなど)

    """

    def __init__(self):
        self.params = []

    def forward(self, x):
        """順伝播

        Sigmoidレイヤの順伝播の結果を返す

        Args:
            x (float): 入力パラメータ

        Returns:
            float: 順伝播結果
        """
        return 1 / (1 + np.exp(-x))
