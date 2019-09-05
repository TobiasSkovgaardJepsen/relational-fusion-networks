from mxnet.gluon.nn import HybridBlock, Activation, Dense, HybridSequential


class Identity(HybridBlock):
    def hybrid_forward(self, F, X):
        return X


def get_activation(activation):
    if activation is None:
        return Identity()
    elif type(activation) is str:
        return Activation(activation)
    else:
        return activation


class FeedForward(HybridSequential):
    def __init__(self, units, in_units, activation=Identity(),
                 *args, **kwargs):
        super().__init__(**kwargs)

        dense = Dense(
            units=units, in_units=in_units,
            activation=None, *args, **kwargs)
        super().add(dense)
        super().add(activation)

        self.out_units = self.units = units


class NoFeedForward(Identity):
    pass
