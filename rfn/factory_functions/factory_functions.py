from typing import List
from .data_types import FeatureInfo, RFNLayerSpecification
from ..network import RFN
from ..relational_fusion import RelationalFusion, NoRelationalFusion
from ..layer import RFNLayer
from ..utils import FeedForward, NoFeedForward


def make_rfn(input_feature_info,
             hidden_layer_specs: List[RFNLayerSpecification],
             output_layer_spec: RFNLayerSpecification,
             output_mode: {'node', 'edge', 'between-edge', None}=None
             ):
    rfn = RFN(output_mode)

    for layer_spec in hidden_layer_specs:
        layer = make_rfn_layer(input_feature_info, layer_spec)
        rfn.add(layer)
        input_feature_info = FeatureInfo(*layer.out_units)

    output_layer = make_rfn_layer(
        input_feature_info, output_layer_spec, output_mode)
    rfn.add(output_layer)

    return rfn


def make_rfn_layer(input_feature_info, layer_spec,
                   output_mode: {'node', 'edge', 'between-edge', None}=None):
    (no_node_features,
     no_edge_features,
     no_between_edge_features) = input_feature_info

    no_node_features_primal = no_node_features
    no_edge_features_primal = no_edge_features

    fuse_primal_in_units = 2*no_node_features_primal + no_edge_features_primal
    relational_fusion_primal = _make_relational_fusion(
        fuse_primal_in_units, layer_spec,
        disable=_is_output_disabled(output_mode, 'node')
    )

    no_node_features_dual = no_edge_features
    no_edge_features_dual = no_between_edge_features + no_node_features

    fuse_dual_in_units = 2*no_node_features_dual + no_edge_features_dual
    relational_fusion_dual = _make_relational_fusion(
        fuse_dual_in_units, layer_spec,
        disable=_is_output_disabled(output_mode, 'edge'))

    feed_forward = _make_feed_forward(
        in_units=input_feature_info.no_between_edge_features,
        layer_spec=layer_spec,
        disable=_is_output_disabled(output_mode, 'between-edge')
    )

    return RFNLayer(
        relational_fusion_primal,
        relational_fusion_dual,
        feed_forward)


def _make_relational_fusion(fuse_in_units, layer_spec, disable):
    if disable:
        return NoRelationalFusion()

    fuse = layer_spec.fusion(
        in_units=fuse_in_units,
        units=layer_spec.units,
        activation=layer_spec.activation)

    return RelationalFusion(
        fuse=fuse,
        aggregate=layer_spec.aggregator(in_units=fuse_in_units),
        normalize=layer_spec.normalization)


def _make_feed_forward(in_units, layer_spec, disable):
    if disable:
        return NoFeedForward()

    return FeedForward(
        in_units=in_units,
        units=layer_spec.units,
        activation=layer_spec.activation
    )


def _is_output_disabled(output_mode, output):
    return output_mode is not None and output != output_mode
