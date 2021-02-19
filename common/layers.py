from common.functions import softmax, cross_entropy_error


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
