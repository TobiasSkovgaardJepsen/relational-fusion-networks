from mxnet.gluon.nn import HybridBlock


class MaskedNormalization(HybridBlock):
    def __init__(self, axis, keepdims=False, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def hybrid_forward(self, F, X, M):
        X = M*X
        normalization = F.sum(
            X,
            axis=self.axis,
            keepdims=self.keepdims)
        return F.where(
            F.contrib.isnan(X/normalization),
            F.zeros_like(X),
            X/normalization
        )


class MaskedSoftmax(MaskedNormalization):
    def hybrid_forward(self, F, X, M):
        return super().hybrid_forward(F, F.exp(X), M)
