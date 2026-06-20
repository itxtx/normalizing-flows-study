import torch.nn as nn
import torch

class NormalizingFlowModel(nn.Module):
    """
    A generic model to chain a sequence of normalizing flow layers.
    """
    def __init__(self, flows, batch_norm_between_layers=False):
        super().__init__()
        self.batch_norm_between_layers = batch_norm_between_layers
        
        # If using batch norm between layers, create a list of BatchNorm1d layers
        if self.batch_norm_between_layers:
            # Assuming all flows have the same data dimension, which they should
            data_dim = flows[0].data_dim if hasattr(flows[0], 'data_dim') else None
            if data_dim is None:
                raise ValueError("Cannot use batch_norm_between_layers if flows do not have a 'data_dim' attribute.")
            
            self.batch_norms = nn.ModuleList(
                [nn.BatchNorm1d(data_dim) for _ in range(len(flows))]
            )
        
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        """
        Apply a sequence of flows to a base distribution sample.
        This is the sampling direction. z -> x
        """
        log_det_jacobian_sum = 0
        for i, flow in enumerate(self.flows):
            z, log_det_jacobian = flow(z)
            log_det_jacobian_sum += log_det_jacobian
            
            # Apply batch normalization between layers if enabled
            if self.batch_norm_between_layers and i < len(self.flows) - 1:
                bn = self.batch_norms[i]
                z = bn(z)
                # Batch norm affects the log-determinant
                log_det_jacobian_sum += self._batch_norm_log_det_jacobian(bn, z)
                
        return z, log_det_jacobian_sum

    def inverse(self, x):
        """
        Apply the inverse of a sequence of flows.
        This is the density estimation direction. x -> z
        """
        log_det_jacobian_sum = 0
        for i, flow in reversed(list(enumerate(self.flows))):
            # Apply inverse batch normalization first
            if self.batch_norm_between_layers and i < len(self.flows) - 1:
                bn = self.batch_norms[i]
                x = self._inverse_batch_norm(bn, x)
                # Inverse batch norm also affects the log-determinant
                log_det_jacobian_sum -= self._batch_norm_log_det_jacobian(bn, x)
            
            x, log_det_jacobian = flow.inverse(x)
            log_det_jacobian_sum += log_det_jacobian
            
        return x, log_det_jacobian_sum

    def _batch_norm_log_det_jacobian(self, bn_layer, x):
        """
        Computes the log-determinant of the Jacobian of a BatchNorm1d layer.
        The transformation is y = (x - mean) / sqrt(var + eps) * gamma + beta.
        The Jacobian is a diagonal matrix, so the log-determinant is the sum
        of the logs of the diagonal elements.
        log|det(J)| = sum(log|gamma / sqrt(var + eps)|)
        """
        # gamma is the learned scaling parameter (weight)
        gamma = bn_layer.weight
        
        # var is the running variance
        var = bn_layer.running_var
        
        # eps is a small value for numerical stability
        eps = bn_layer.eps
        
        # Log-determinant for each dimension
        log_det_per_dim = torch.log(torch.abs(gamma)) - 0.5 * torch.log(var + eps)
        
        # Sum over all dimensions for each sample in the batch
        return log_det_per_dim.sum()

    def _inverse_batch_norm(self, bn_layer, y):
        """
        Computes the inverse of a BatchNorm1d layer.
        x = (y - beta) / gamma * sqrt(var + eps) + mean
        """
        gamma = bn_layer.weight
        beta = bn_layer.bias
        mean = bn_layer.running_mean
        var = bn_layer.running_var
        eps = bn_layer.eps
        
        # Reshape for broadcasting
        gamma = gamma.view(1, -1)
        beta = beta.view(1, -1)
        mean = mean.view(1, -1)
        var = var.view(1, -1)
        
        x = (y - beta) / gamma * torch.sqrt(var + eps) + mean
        return x
