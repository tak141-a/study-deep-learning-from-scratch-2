import numpy as np
from functions import softmax, cross_entropy_error


class SoftmaxWithLoss:
    """
    SoftmaxWithLossレイヤ

    Attributes:
        params (list): パラメータ
        grads (list): 勾配
        y (list): softmaxの出力
        t (list): 教師ラベル
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        """順伝播

        SoftmaxWithLossレイヤの順伝播の結果を返す

        Args:
            x (ndarray): 入力
            t (ndarray): 教師ラベル
        """
        self.t = t
        self.y = softmax(x)

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss


class MatMul:
    """
    MatMulレイヤ(行列の積)

    Attributes:
        params (list): 学習するパラメータ
        grads (list): 勾配
        x (list): 入力
    """
    def __init__(self, W):
        """
        params:
            W (list): 重み
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]  # paramsと同じ形状のゼロ配列を作る
        self.x = None

    def forward(self, x):
        """順伝播

        MatMulレイヤの順伝播の結果を返す

        Args:
            x (ndarray): 入力
        """
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        """逆伝播

        MatMulレイヤの逆伝播の結果を返す

        Args:
            dout ():
        """
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW  # NumPy配列のメモリ位置を固定した上でNumPy配列の要素を上書き
        return dx


class Sigmoid:
    """
    Sigmoidレイヤ

    Attributes:
        params (list): 学習するパラメータ
        grads (list): 勾配
        out (list): 順伝播の出力
    """
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        """順伝播

        Sigmoidレイヤの順伝播の結果を返す

        Args:
            x (ndarray): 入力

        Return:
            out (ndarray): 順伝播結果
        """
        out = 1 / (1 + np.exp(-x))  # Sigmoid関数
        self.out = out  # 順伝播結果をインスタンス変数のoutに保持
        return out

    def backward(self, dout):
        """逆伝播

        Sigmoidレイヤの逆伝播の結果を返す

        Args:
            dout (ndarray): 入力

        Return:
            dx (ndarray): 逆伝播結果
        """
        dx = dout * (1.0 - self.out) * self.out  # 順伝播結果self.outを使って計算を行う
        return dx


class Affine:
    """
    Affineレイヤ

    Attributes:
        params (list): 学習するパラメータ
        grads (list): 勾配
        out (list): 順伝播の出力
    """
    def __init__(self, W, b):
        """
        params:
            W (ndarray): 重み
            b (): バイアス
        """
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        """順伝播

        Affineレイヤの順伝播の結果を返す

        Args:
            x (ndarray): 入力

        Return:
            out (ndarray): 順伝播結果
        """
        W, b = self.params
        out = np.dot(x, W) + b  # 入力と重みの行列の積 + バイアス
        self.x = x
        return out

    def backward(self, dout):
        """逆伝播

        Affineレイヤの逆伝播の結果を返す

        Args:
            dout (ndarray): 入力

        Return
            dx (ndarray): 逆伝播結果
        """
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


W = np.random.rand(2, 3)
x = np.random.rand(3, 2)
sigmoid = Sigmoid()
print(sigmoid.forward(x))
