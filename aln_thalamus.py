"""
Thalamocortical models based on `neurolib` MultiModel framework.
"""

import numpy as np
from neurolib.models.multimodel.builder.aln import (
    ALN_EXC_DEFAULT_PARAMS,
    ALN_INH_DEFAULT_PARAMS,
    ALN_NODE_DEFAULT_CONNECTIVITY,
    ALN_NODE_DEFAULT_DELAYS,
    ALNNode,
)
from neurolib.models.multimodel.builder.base.constants import EXC, INH
from neurolib.models.multimodel.builder.base.network import Network
from neurolib.models.multimodel.builder.thalamus import (
    TCR_DEFAULT_PARAMS,
    THALAMUS_NODE_DEFAULT_CONNECTIVITY,
    TRN_DEFAULT_PARAMS,
    ThalamicNode,
)


class ALNThalamusMiniNetwork(Network):
    """
    Simple thalamo-cortical motif: 1 node cortical ALN + 1 node NMM thalamus.
    """

    name = "ALN 1 node + Thalamus"
    label = "1ALNThlmNet"

    sync_variables = [
        "network_exc_exc",
        "network_exc_exc_sq",
        "network_inh_exc",
    ]

    default_output = f"r_mean_{EXC}"
    output_vars = [f"r_mean_{EXC}", f"r_mean_{INH}", f"I_A_{EXC}"]

    def __init__(
        self,
        connectivity_matrix,
        delay_matrix,
        thalamus_exc_inh_multiplier=1.0,
        exc_aln_params=ALN_EXC_DEFAULT_PARAMS,
        inh_aln_params=ALN_INH_DEFAULT_PARAMS,
        aln_node_connectivity=ALN_NODE_DEFAULT_CONNECTIVITY,
        aln_node_delays=ALN_NODE_DEFAULT_DELAYS,
        tcr_params=TCR_DEFAULT_PARAMS,
        trn_params=TRN_DEFAULT_PARAMS,
        thalamus_node_connectivity=THALAMUS_NODE_DEFAULT_CONNECTIVITY,
    ):
        """
        :param thalamus_exc_inh_multiplier: multiplier of thalamic connectivity
            - EXC->INH from cortex is scaled by this number w.r.t EXC->EXC
        :type thalamus_exc_inh_multiplier: float
        """
        # self connections are resolved within nodes, so zeroes at the diagonal
        assert np.all(np.diag(connectivity_matrix) == 0.0)

        # init ALN node with index 0
        aln_node = ALNNode(
            exc_params=exc_aln_params,
            inh_params=inh_aln_params,
            connectivity=aln_node_connectivity,
            delays=aln_node_delays,
        )
        aln_node.index = 0
        aln_node.idx_state_var = 0
        # set correct indices of noise input - one per mass
        for mass in aln_node:
            mass.noise_input_idx = [mass.index]

        # init thalamus node with index 1
        thalamus = ThalamicNode(
            tcr_params=tcr_params,
            trn_params=trn_params,
            connectivity=thalamus_node_connectivity,
        )
        thalamus.index = 1
        thalamus.idx_state_var = aln_node.num_state_variables
        # set correct indices of noise input - one per mass, after ALN noise
        # indices
        for mass in thalamus:
            mass.noise_input_idx = [aln_node.num_noise_variables + mass.index]

        # init network with these two nodes
        super().__init__(
            nodes=[aln_node, thalamus],
            connectivity_matrix=connectivity_matrix,
            delay_matrix=delay_matrix,
        )
        # assert we have 3 sync variables - network exc -> exc (for both),
        # network exc -> exc squared (for ALN), and network exc -> inh
        # (for thalamus)
        assert len(self.sync_variables) == 3
        self.thalamus_exc_inh_mult = thalamus_exc_inh_multiplier

    def _sync(self):
        """
        Set coupling variables.
        """
        # get indices of coupling variables from all nodes
        exc_indices = [
            next(
                iter(
                    node.all_couplings(
                        mass_indices=node.excitatory_masses.tolist()
                    )
                )
            )
            for node in self
        ]
        assert len(exc_indices) == len(self)

        return (
            # basic EXC <-> EXC coupling
            self._additive_coupling(
                within_node_idx=exc_indices, symbol="network_exc_exc"
            )
            # squared EXC <-> EXC coupling (only to ALN)
            + self._additive_coupling(
                within_node_idx=exc_indices,
                symbol="network_exc_exc_sq",
                connectivity=self.connectivity * self.connectivity,
            )
            # EXC -> INH coupling (only to thalamus)
            + self._additive_coupling(
                within_node_idx=exc_indices,
                symbol="network_inh_exc",
                connectivity=self.connectivity * self.thalamus_exc_inh_mult,
            )
            + super()._sync()
        )
