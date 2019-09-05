from mxnet.gluon.nn import (
    HybridBlock, HybridSequential, Dense, LeakyReLU)
from abc import ABC, abstractmethod
from .utils import MaskedSoftmax


class RelationalAggregator(ABC, HybridBlock):
    def __init__(self, *args, **kwargs):
        self._filter_kwargs(kwargs)
        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_args(self, X, Z, M):
        raise NotImplementedError

    def _filter_kwargs(self, kwargs):
        pass


class AttentionalAggregator(RelationalAggregator):
    def __init__(self, in_units, activation=LeakyReLU(0.2), **kwargs):
        super().__init__(**kwargs)

        with self.name_scope():
            self.coefficient_net = HybridSequential()

            dense = Dense(
                    in_units=in_units, units=1,
                    use_bias=False,
                    flatten=False)
            self.coefficient_net.add(dense)
            self.coefficient_net.add(activation)

        self.softmax = MaskedSoftmax(axis=1, keepdims=True)

    def hybrid_forward(self, F, X, Z, M):
        """ X is the concatenation of source, edge, and target
            features of an edge. Z is the fused representation.
            M is a mask over the neighborhood.
        """
        coefficient = self.coefficient_net(X)
        attention_weight = self.softmax(coefficient, M)

        return F.sum(attention_weight*Z, axis=1)

    def get_args(self, X, Z, M, *args):
        return X, Z, M


class NonAttentionalAggregator(RelationalAggregator):
    def hybrid_forward(self, F, Z, M):
        """ Z is the fused representation.
            M is a mask over the neighborhood.
        """
        return F.sum(M*Z, axis=1)/F.sum(M, axis=1)

    def get_args(self, X, Z, M, *args):
        return Z, M

    def _filter_kwargs(self, kwargs):
        kwargs.pop('in_units')
