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


W = np.random.rand(2, 3)
x = np.random.rand(3, 2)
matmul = MatMul(W)
print(matmul.forward(x))
