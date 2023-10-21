import torch
import linops as lo
import linops.diag as lod


def diagonal(solution, input_vec, strategy, parameters):
    J = lo.VectorJacobianOperator(solution, input_vec)

    if strategy == "exact":
        return lod.exact_diag(J, **parameters)[0]
    elif strategy == "default" or strategy == "xdiag":
        return lod.xdiag(J, **parameters)[0]
    else:
        raise RuntimeError("Unknown divergence strategy.")
