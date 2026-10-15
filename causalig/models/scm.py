"""
Structural Causal Model (SCM) implementation.
"""

import torch
import torch.nn as nn


class DifferentiableSCM(nn.Module):
    """
    Differentiable Structural Causal Model.

    This model implements a DAG-based structural causal model where each node
    is computed from its parents using assignment functions.
    """

    def __init__(self, adjacency_matrix, assignments, topological_order):
        """
        Initialize the SCM.

        Args:
            adjacency_matrix: adjacency matrix (p x p) as numpy or torch, `[i, j]`
                nonzero if feature i causes feature j
            assignments: list of p functions, each taking a tensor of parent values
            topological_order: list of ints representing a valid ordering
        """
        super().__init__()

        adjacency_matrix = torch.as_tensor(adjacency_matrix, dtype=torch.float32)
        self.register_buffer("adjacency_matrix", adjacency_matrix)

        self.assignments = assignments
        self.topological_order = topological_order
        self.p = adjacency_matrix.shape[0]

    def forward(self, X_input=None, structural=False):
        """
        Forward pass of the SCM.

        Args:
            X_input: (batch, p) tensor of exogenous inputs or proposed node values
            structural (bool):
                - False: standard SCM sampling (nodes computed sequentially from parents)
                - True: structural mode, all nodes treated as independent inputs;
                        outputs = direct mechanisms applied to inputs

        Returns:
            X: (batch, p) tensor of node values
        """
        batch = 1 if X_input is None else X_input.shape[0]
        device = self.adjacency_matrix.device if X_input is None else X_input.device

        X = [None] * self.p

        noise = torch.randn(batch, self.p, device=device) * 0.01

        for node in self.topological_order:
            parents = torch.where(self.adjacency_matrix[:, node] != 0)[0]

            if structural:
                # Treat all nodes as independent inputs, compute only via assignments
                if len(parents) == 0:
                    # root nodes: copy input
                    if X_input is None:
                        X[node] = torch.zeros(batch, device=device)
                    else:
                        X[node] = X_input[:, node]
                else:
                    # use provided inputs for parents
                    parent_vals = (
                        X_input[:, parents]
                        if X_input is not None
                        else torch.zeros(batch, len(parents), device=device)
                    )
                    X[node] = self.assignments[node](parent_vals)

            else:
                # Original SCM behavior: propagate values sequentially
                if len(parents) == 0:
                    if X_input is not None:
                        X[node] = X_input[:, node]
                    else:
                        X[node] = torch.zeros(batch, device=device)
                else:
                    parent_vals = torch.stack([X[p] for p in parents], dim=-1)
                    X[node] = self.assignments[node](parent_vals) + noise[:, node]

        return torch.stack(X, dim=-1)

    def forward_do(self, X_input, noise, do_nodes=None, do_values=None):
        """
        Forward pass with multiple do-interventions.

        Args:
            X_input: (batch, p) tensor of exogenous inputs
            noise: (batch, p) tensor of noise values
            do_nodes: list or tensor of intervened node indices
            do_values: (batch, len(do_nodes)) tensor with intervention values

        Returns:
            X: (batch, p) tensor of node values under intervention
        """

        device = self.adjacency_matrix.device
        p = self.p

        X = [None] * p

        # create intervention mask
        intervene_mask = torch.zeros(p, dtype=torch.bool, device=device)

        if do_nodes is not None:
            do_nodes = torch.as_tensor(do_nodes, device=device)
            intervene_mask[do_nodes] = True

            if do_values.dim() == 1:
                do_values = do_values.unsqueeze(1)

        for node in self.topological_order:

            if intervene_mask[node]:
                # index of node inside do_nodes
                idx = (do_nodes == node).nonzero(as_tuple=True)[0]

                # set intervention value
                X[node] = do_values[:, idx].squeeze(-1)
                continue

            parents = torch.where(self.adjacency_matrix[:, node] != 0)[0]

            if len(parents) == 0:
                X[node] = X_input[:, node]  # + noise[:, node]

            else:
                parent_vals = torch.stack([X[p] for p in parents], dim=-1)
                X[node] = self.assignments[node](parent_vals) + noise[:, node]

        return torch.stack(X, dim=-1)
