import torch

class UniformAdam(torch.optim.Optimizer):
    """
    UniformAdam is a variant of the Adam optimizer with isotropic momentum tracking
    and uniform step scaling.

    - Isotropic: The second moment estimate (g2) is reduced across the channel dimension,
      producing a single scalar per sample (e.g., per point in 3D). This treats all channels
      equally when tracking gradient magnitudes — useful in geometric domains like 3D points.

    - Uniform: The final update is normalized by the maximum RMS value across the batch,
      resulting in a single shared learning rate scale for all parameters. This avoids
      per-element normalization and ensures updates are uniformly scaled per step.

    These changes are beneficial in spatial optimization problems where uniform scaling
    and rotational symmetry across channels are desired.

    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Coefficients for computing running averages of
                                     gradient and its square (default: (0.9, 0.999)).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999)):
        defaults = dict(lr=lr, betas=betas)
        super(UniformAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(UniformAdam, self).__setstate__(state)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Lazy state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['g1'] = torch.zeros_like(p.data)
                    state['g2'] = torch.zeros_like(p.data[..., :1])  # Isotropic accumulator

                state['step'] += 1
                g1 = state['g1']
                g2 = state['g2']

                # Update exponential moving averages
                g1.mul_(b1).add_(grad, alpha=1 - b1)
                g2.mul_(b2).add_(grad.square().sum(dim=-1, keepdim=True), alpha=1 - b2)

                # Bias correction
                step = state['step']
                m1 = g1 / (1 - b1 ** step)
                m2 = g2 / (1 - b2 ** step)

                # Uniform normalization (shared scalar denominator)
                denom = m2.sqrt().max() + 1e-8
                update = m1 / denom

                p.data.sub_(update, alpha=lr)

def laplacian_edges(verts, edges):
    """
    Compute the uniform laplacian

    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    edges : torch.Tensor
        array of edges.
    """
    V = verts.shape[0]
    E = edges.shape[0]

    # Neighbor indices
    ii = edges[:, [0]].flatten()
    jj = edges[:, [1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device='cuda', dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()
