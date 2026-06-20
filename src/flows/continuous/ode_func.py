import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    """
    Simplified ODE function for continuous normalizing flows.
    Uses a simpler approach that avoids complex trace estimation.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.dim = dim
        # Time-conditioned velocity field: the integration time t is concatenated
        # to the state. An autonomous (time-independent) field cannot fold
        # trajectories in 2D (paths may not cross), so it fails on targets like
        # two-moons; conditioning on t fixes that.
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

        self._initialize_weights()

    def _velocity(self, t, z):
        t_col = t.expand(z.shape[0]).unsqueeze(1).to(z)
        return self.net(torch.cat([z, t_col], dim=1))

    def forward(self, t, augmented_state):
        """
        Augmented dynamics for a continuous normalizing flow.

        Returns d/dt of [z, log_det], where the instantaneous change of variables
        gives  d(log p)/dt = -tr(dv/dz). The divergence tr(dv/dz) is computed
        exactly for small dimensions and with Hutchinson's estimator otherwise.

        Args:
            t: integration time (the field here is autonomous, so unused).
            augmented_state: concatenated tensor [z, log_det].

        Returns:
            concatenated tensor [dz/dt, dlog_det/dt].
        """
        z = augmented_state[:, :self.dim]

        with torch.enable_grad():
            # z must be part of an autograd graph to take the divergence. It is a
            # leaf only when the surrounding state does not already require grad
            # (e.g. sampling); guard requires_grad_ accordingly.
            if not z.requires_grad:
                z = z.requires_grad_(True)

            dz_dt = self._velocity(t, z)

            if self.dim <= 2:
                # Exact trace: sum of diagonal Jacobian entries.
                divergence = 0.0
                for i in range(self.dim):
                    grad_i = torch.autograd.grad(
                        dz_dt[:, i].sum(), z, create_graph=True, retain_graph=True
                    )[0]
                    divergence = divergence + grad_i[:, i:i + 1]
            else:
                # Hutchinson's stochastic trace estimator (FFJORD).
                eps = torch.randn_like(z)
                vjp = torch.autograd.grad(
                    dz_dt, z, grad_outputs=eps, create_graph=True, retain_graph=True
                )[0]
                divergence = (vjp * eps).sum(dim=1, keepdim=True)

        # ContinuousFlow integrates this field forward (t: 0->1) for sampling and
        # backward (t: 1->0) for density, returning the accumulated value directly
        # as each map's log|det|. With that wiring the consistent sign for the
        # divergence term is +tr(dv/dz) (verified against an autodiff Jacobian).
        dlog_det_dt = divergence
        return torch.cat([dz_dt, dlog_det_dt], dim=1)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization with proper scaling."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        final_layer = self.net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)
