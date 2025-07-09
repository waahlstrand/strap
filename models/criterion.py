import torch
import torch.nn as nn
from torch import Tensor

class PartialLikelihood(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super(PartialLikelihood, self).__init__()

        self.eps = eps
        
    def forward(self, 
                risk_scores: Tensor, 
                time_to_event: Tensor, 
                event_indicators: Tensor) -> Tensor:
        """
        Computes the partial likelihood loss for Cox Proportional Hazards model.

        Args:
            risk_scores (Tensor): Risk scores for each sample (N,).
            time_to_event (Tensor): Survival times for each sample (N,).
            event_indicators (Tensor): Event indicators (1 if event occurred, 0 if censored) for each sample (N,).

        Returns:
            Tensor: Computed loss value.
        """

        # Must sort the risk scores in descending order of survival time for the risk sets    
        sorted_indices = torch.argsort(time_to_event, descending=True)
        risk_scores = risk_scores[sorted_indices] if risk_scores.shape[0] > 1 else risk_scores
        event_indicators = event_indicators[sorted_indices] if risk_scores.shape[0] > 1 else event_indicators
            
        hazard_ratio = torch.exp(risk_scores)
        log_cumsum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0) + self.eps)

        uncensored_likelihood = risk_scores - log_cumsum_hazard
        loss = -torch.mean(uncensored_likelihood * event_indicators)
        return loss