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
        W, b = self.params  # paramsからWとbを取り出し
        out = np.dot(x, W) + b  # xW + b
        return out


class TwoLayerNet:
    """2層レイヤ

    (IN) -> Affine -> Sigmoid -> Affine -> (OUT)

    Attributes:
        layers (list): レイヤ
        params (list): 全ての学習パラメータ
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Params:
            input_size (int): 入力層のサイズ
            hidden_size (int): 隠れ層のサイズ
            output_size (int): 出力層のサイズ
        """
        I, H, O = input_size, hidden_size, output_size

        # 重みとバイアスの初期化
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # すべての重みをリストにまとめる(各レイヤのインスタンス変数paramsを連結)
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        """推論

        TwoLayerNetの推論を行う

        Params:
            x (list): 入力

        Returns:
            list: TwoLayerNetの推論結果
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x


# 動作確認
x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(model.params)
