from mxnet.gluon.nn import HybridBlock


class RFNLayer(HybridBlock):
    class Join(HybridBlock):
        """ Block that handles merging of node and between-edge features
            for use in graph fusion on a dual graph representation.
        """
        def hybrid_forward(self, F, X_N, X_B, N_shared_node_dual):
            X_N_prime = F.take(X_N, indices=N_shared_node_dual).flatten()
            return F.concat(X_B, X_N_prime, dim=1)

    def __init__(self,
                 relational_fusion_primal, relational_fusion_dual,
                 feed_forward,
                 **kwargs):
        super().__init__(**kwargs)

        self.relational_fusion_primal = relational_fusion_primal
        self.join = self.Join()
        self.relational_fusion_dual = relational_fusion_dual
        self.feed_forward = feed_forward

        self.register_children(
            self.relational_fusion_primal,
            self.join,
            self.relational_fusion_dual,
            self.feed_forward)

    def hybrid_forward(self, F, X_N, X_E, X_B,
                       N_node_primal, N_edge_primal, node_mask_primal,
                       N_node_dual, N_edge_dual, N_shared_node_dual,
                       node_mask_dual):
        Z_N = self.relational_fusion_primal(
            X_N, X_E,
            N_node_primal, N_edge_primal, node_mask_primal)

        X_B_prime = self.join(
            X_N, X_B, N_shared_node_dual)
        Z_E = self.relational_fusion_dual(
            X_E, X_B_prime,
            N_node_dual, N_edge_dual, node_mask_dual)

        Z_B = self.feed_forward(X_B)

        return Z_N, Z_E, Z_B

    def register_children(self, *blocks):
        for block in blocks:
            self.register_child(block)

    @property
    def out_units(self):
        return (
            self.relational_fusion_primal.out_units,
            self.relational_fusion_dual.out_units,
            self.feed_forward.out_units
        )
