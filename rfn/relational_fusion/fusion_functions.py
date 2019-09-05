from mxnet.gluon.nn import HybridBlock, Dense
from ..utils import Identity


class AdditiveFusion(HybridBlock):
    def __init__(self, units, in_units,
                 activation=Identity(),
                 **kwargs):
        super().__init__(**kwargs)

        self.units = self.out_units = units
        self.in_units = in_units

        with self.name_scope():
            self.dense = Dense(
                units=units, in_units=in_units,
                use_bias=True,
                flatten=False)
            self.activation = activation

    def hybrid_forward(self, F, X):
        return self.activation(self.dense(X))


class InteractionalFusion(AdditiveFusion):
    def __init__(self, units, in_units,
                 activation=Identity(),
                 **kwargs):
        super().__init__(
            units, in_units, activation, **kwargs)

        with self.name_scope():
            self.interaction = Dense(
                units=in_units, in_units=in_units,
                use_bias=False,
                flatten=False
            )

    def hybrid_forward(self, F, X):
        X_interact = self.interaction(X)*X
        assert X_interact.shape == X.shape

        return super().hybrid_forward(F, X_interact)
