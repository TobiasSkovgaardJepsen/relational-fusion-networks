from mxnet.gluon.nn import HybridBlock


class NoNormalization(HybridBlock):
    def hybrid_forward(self, F, X):
        return X


class L2Normalization(HybridBlock):
    def hybrid_forward(self, F, X):
        return F.L2Normalization(X)
