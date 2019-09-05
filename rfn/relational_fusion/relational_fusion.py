from mxnet.gluon.nn import HybridBlock


class RelationalFusion(HybridBlock):
    def __init__(self, fuse, aggregate, normalize, **kwargs):
        super().__init__(**kwargs)

        self.fuse = fuse
        self.out_units = self.fuse.out_units
        self.aggregate = aggregate
        self.normalize = normalize

        with self.name_scope():
            child_blocks = [self.fuse, self.aggregate, self.normalize]
            for block in child_blocks:
                self.register_child(block)

    def _apply_mask(self, F, array, mask):
        return F.sparse.broadcast_mul(mask, array)

    def hybrid_forward(self, F, X_node, X_edge, N_node, N_edge, mask):
        assert (X_node.shape[0] == N_node.shape[0] ==
                N_edge.shape[0] == mask.shape[0])
        X_neighbor_node = F.take(X_node, indices=N_node)
        X_neighbor_edge = F.take(X_edge, indices=N_edge)
        X_node = X_node.expand_dims(1).broadcast_like(X_neighbor_node)

        X = F.concat(X_node, X_neighbor_edge, X_neighbor_node, dim=2)
        Z_node = self.fuse(X)
        Z_node = self._apply_mask(F, Z_node, mask)

        aggregate_args = self.aggregate.get_args(
            X, Z_node, mask, X_node, X_neighbor_node, X_neighbor_edge)
        Z_node = self.aggregate(*aggregate_args)

        Z_node = self.normalize(Z_node)
        return Z_node


class NoRelationalFusion(HybridBlock):
    def hybrid_forward(self, F, X_node, X_edge, N_node, N_edge, mask):
        return X_node
