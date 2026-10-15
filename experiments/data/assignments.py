"""
Assignment function generators for structural causal models.
"""

import numpy as np
import torch


def generate_random_assignments_torch_and_numpy(dag, return_metadata=False):
    """
    Generate both PyTorch-differentiable and NumPy-compatible assignment functions
    for a DAG. PyTorch functions are device-safe: coefficients automatically
    move to the same device as the input tensor.

    Args:
        dag (np.ndarray): adjacency matrix (i -> j is dag[i, j] != 0)
        return_metadata (bool): if True, also return a per-node metadata list
            describing the true assignment (see below). Default False,
            preserving the original 2-tuple return for existing callers.

    Returns:
        assignments_torch (list): list of callable PyTorch functions
            Each f(X_parents) returns a tensor of shape (batch,)
        assignments_numpy (list): list of callable NumPy functions
            Each f(X_parents) returns a numpy array of shape (batch,)
        metadata (list, only if return_metadata=True): one dict per node:
            - "type": "root" | "linear" | "tanh" | "interaction"
              ("interaction" degenerates to "linear" here whenever the node
              has fewer than 2 parents, matching what the function actually
              computes in that fallback case.)
            - "parents": list of parent node indices, in the same order as
              the columns of `Xp` consumed by the assignment function.
            - "weights": np.ndarray of per-parent coefficients aligned with
              "parents" (None for root nodes). For "interaction", only the
              first two parents are mechanistically active (the assignment
              only ever reads `Xp[:, 0]` and `Xp[:, 1]`); any further parents
              are true zeros, and "weights" is zero-padded accordingly, with
              "active_parents" listing the two indices that actually matter.
    """
    num_nodes = dag.shape[0]
    assignments_torch = []
    assignments_numpy = []
    metadata = []

    for node in range(num_nodes):
        parents = list(np.where(dag[:, node] != 0)[0])

        # Root node: no parents
        if len(parents) == 0:
            assignments_torch.append(None)
            assignments_numpy.append(None)
            metadata.append({"type": "root", "parents": [], "weights": None})
            continue

        # Random function type
        func_type = np.random.choice(["linear", "tanh", "interaction"])

        if func_type == "linear":
            coefs = np.random.uniform(-1, 1, size=len(parents))
            coefs_torch = torch.tensor(coefs, dtype=torch.float32)

            def make_linear(coefs_torch, coefs_np):
                def fn_torch(Xp):
                    coefs = coefs_torch.to(Xp.device)
                    return (Xp * coefs).sum(dim=1)

                def fn_numpy(Xp):
                    return (Xp * coefs_np).sum(axis=1)

                return fn_torch, fn_numpy

            fn_t, fn_np = make_linear(coefs_torch, coefs)
            assignments_torch.append(fn_t)
            assignments_numpy.append(fn_np)
            metadata.append({"type": "linear", "parents": parents, "weights": coefs})

        elif func_type == "tanh":
            coefs = np.random.uniform(-1, 1, size=len(parents))
            coefs_torch = torch.tensor(coefs, dtype=torch.float32)

            def make_tanh(coefs_torch, coefs_np):
                def fn_torch(Xp):
                    coefs = coefs_torch.to(Xp.device)
                    return torch.tanh((Xp * coefs).sum(dim=1))

                def fn_numpy(Xp):
                    return np.tanh((Xp * coefs_np).sum(axis=1))

                return fn_torch, fn_numpy

            fn_t, fn_np = make_tanh(coefs_torch, coefs)
            assignments_torch.append(fn_t)
            assignments_numpy.append(fn_np)
            metadata.append({"type": "tanh", "parents": parents, "weights": coefs})

        elif func_type == "interaction":
            if len(parents) >= 2:
                w = np.random.uniform(-1, 1, 2)
                w_torch = torch.tensor(w, dtype=torch.float32)

                def make_interact(w_torch, w_np):
                    def fn_torch(Xp):
                        w_local = w_torch.to(Xp.device)
                        x1 = Xp[:, 0]
                        x2 = Xp[:, 1]
                        return (
                            w_local[0] * x1
                            + w_local[1] * x2
                            + (w_local[0] * w_local[1]) * x1 * x2
                        )

                    def fn_numpy(Xp):
                        x1 = Xp[:, 0]
                        x2 = Xp[:, 1]
                        return (
                            w_np[0] * x1 + w_np[1] * x2 + (w_np[0] * w_np[1]) * x1 * x2
                        )

                    return fn_torch, fn_numpy

                fn_t, fn_np = make_interact(w_torch, w)
                assignments_torch.append(fn_t)
                assignments_numpy.append(fn_np)
                padded_weights = np.zeros(len(parents))
                padded_weights[:2] = w
                metadata.append(
                    {
                        "type": "interaction",
                        "parents": parents,
                        "weights": padded_weights,
                        "active_parents": [parents[0], parents[1]],
                    }
                )

            else:
                # fallback to linear
                coefs = np.random.uniform(-1, 1, size=len(parents))
                coefs_torch = torch.tensor(coefs, dtype=torch.float32)

                def make_linear(coefs_torch, coefs_np):
                    def fn_torch(Xp):
                        coefs = coefs_torch.to(Xp.device)
                        return (Xp * coefs).sum(dim=1)

                    def fn_numpy(Xp):
                        return (Xp * coefs_np).sum(axis=1)

                    return fn_torch, fn_numpy

                fn_t, fn_np = make_linear(coefs_torch, coefs)
                assignments_torch.append(fn_t)
                assignments_numpy.append(fn_np)
                metadata.append(
                    {"type": "linear", "parents": parents, "weights": coefs}
                )

    if return_metadata:
        return assignments_torch, assignments_numpy, metadata
    return assignments_torch, assignments_numpy
