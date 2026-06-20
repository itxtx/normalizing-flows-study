import torch
import torch.nn as nn
from .flow import Flow

class SequentialFlow(Flow):
    """
    A sequence of flows that are applied sequentially.
    """
    def __init__(self, flows):
        super().__init__()
        if not isinstance(flows, (list, nn.ModuleList)):
            raise ValueError("flows must be a list or nn.ModuleList")
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        """
        Computes the forward transformation through all flows.
        """
        total_log_det = torch.zeros(z.size(0), device=z.device)
        for flow in self.flows:
            z, log_det = flow.forward(z)
            total_log_det += log_det
        return z, total_log_det

    def inverse(self, x):
        """
        Computes the inverse transformation through all flows.
        The inverse is applied in the reverse order of the forward pass.
        """
        total_log_det = torch.zeros(x.size(0), device=x.device)
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            total_log_det += log_det
        return x, total_log_det
