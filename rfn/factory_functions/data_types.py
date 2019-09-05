from collections import namedtuple
from typing import Union
from mxnet.gluon.nn import Block
from ..relational_fusion.fusion_functions import (
    AdditiveFusion, InteractionalFusion)
from ..relational_fusion.aggregators import (
    AttentionalAggregator, NonAttentionalAggregator)
from ..utils import get_activation

FeatureInfoBase = namedtuple(
    'FeatureInfoBase',
    'no_node_features no_edge_features, no_between_edge_features')


class FeatureInfo(FeatureInfoBase):
    @classmethod
    def from_feature_matrices(cls, X_V, X_E, X_B):
        return cls(*[X.shape[1] for X in (X_V, X_E, X_B)])


RFNLayerSpecificationBase = namedtuple(
    'RFNLayerSpecification',
    'units fusion aggregator normalization activation')


class RFNLayerSpecification:
    def __init__(self, units: int,
                 fusion: {'additive', 'interactional'},
                 aggregator: {'attentional', 'non-attentional'},
                 normalization: Block=None,
                 activation: Union[Block, str]=None
                 ):

        if type(fusion) is str:
            fusion = {
                'additive': AdditiveFusion,
                'interactional': InteractionalFusion
            }[fusion]

        if type(aggregator) is str:
            aggregator = {
                'attentional': AttentionalAggregator,
                'non-attentional': NonAttentionalAggregator
            }[aggregator]

        activation = get_activation(activation)

        self.units = units
        self.fusion = fusion
        self.aggregator = aggregator
        self.normalization = normalization
        self.activation = activation
