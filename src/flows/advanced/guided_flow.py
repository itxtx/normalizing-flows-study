from src.flows.flow.flow import Flow

class GuidedFlow(Flow):
    """
    Flow with classifier-free guidance for conditional generation.
    """
    
    def __init__(self, base_flow, guidance_strength=7.5):
        super().__init__()
        self.base_flow = base_flow
        self.guidance_strength = guidance_strength
        
    def forward(self, z, condition=None):
        """Forward pass with guidance."""
        if condition is None:
            # Unconditional generation
            return self.base_flow.forward(z)
        
        # Conditional generation
        x_cond, log_det_cond = self.base_flow.forward(z)
        
        # Unconditional generation (using null condition)
        x_uncond, log_det_uncond = self.base_flow.forward(z)
        
        # Apply guidance
        x_guided = x_uncond + self.guidance_strength * (x_cond - x_uncond)
        
        # Interpolate log-det
        log_det_guided = log_det_uncond + self.guidance_strength * (log_det_cond - log_det_uncond)
        
        return x_guided, log_det_guided
    
    def inverse(self, x, condition=None):
        """Inverse pass with guidance."""
        if condition is None:
            # Unconditional evaluation
            return self.base_flow.inverse(x)
        
        # Conditional evaluation
        z_cond, log_det_cond = self.base_flow.inverse(x)
        
        # Unconditional evaluation
        z_uncond, log_det_uncond = self.base_flow.inverse(x)
        
        # Apply guidance
        z_guided = z_uncond + self.guidance_strength * (z_cond - z_uncond)
        
        # Interpolate log-det
        log_det_guided = log_det_uncond + self.guidance_strength * (log_det_cond - log_det_uncond)
        
        return z_guided, log_det_guided
