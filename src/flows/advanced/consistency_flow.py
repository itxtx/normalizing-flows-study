import torch
from src.flows.flow.flow import Flow

class ConsistencyFlow(Flow):
    """
    Flow with consistency training for single-step generation.
    """
    
    def __init__(self, base_flow, ema_decay=0.999):
        super().__init__()
        self.base_flow = base_flow
        self.ema_decay = ema_decay
        
        # Create target network (EMA of base flow)
        self.target_flow = type(base_flow)(*self._get_flow_params(base_flow))
        self._update_target_network(0.0)  # Initialize with same weights
    
    def _get_flow_params(self, flow):
        """Extract parameters from base flow for target initialization."""
        if hasattr(flow, 'dim'):
            return [flow.dim]
        return []
    
    def _update_target_network(self, decay):
        """Update target network with exponential moving average."""
        with torch.no_grad():
            for target_param, param in zip(self.target_flow.parameters(), self.base_flow.parameters()):
                target_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def forward(self, z, update_target=True):
        """Forward pass with consistency training."""
        # Online prediction
        x_online, log_det_online = self.base_flow.forward(z)
        
        # Target prediction
        with torch.no_grad():
            x_target, log_det_target = self.target_flow.forward(z)
        
        # Update target network
        if update_target:
            self._update_target_network(self.ema_decay)
        
        return x_online, log_det_online, x_target, log_det_target
    
    def inverse(self, x, update_target=True):
        """Inverse pass with consistency training."""
        # Online prediction
        z_online, log_det_online = self.base_flow.inverse(x)
        
        # Target prediction
        with torch.no_grad():
            z_target, log_det_target = self.target_flow.inverse(x)
        
        # Update target network
        if update_target:
            self._update_target_network(self.ema_decay)
        
        return z_online, log_det_online, z_target, log_det_target
