import torch
import math


def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """
    Compute conditional entropy
    Args:
        log_probs_N_K_C: torch.Tensor, the log probabilities. With the following ordering:
            N: Batch Size
            K: Monte-Carlo Samples
            C: Number of Classes
    Returns:
        entropies_N: torch.Tensor, conditional entropy for each sample in batch
    """
    N, K, C = log_probs_N_K_C.shape

    nats_N_K_C = log_probs_N_K_C * torch.exp(log_probs_N_K_C)
    entropies_N = -torch.sum(nats_N_K_C, dim=(1, 2)) / K # simply average across MC samples

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy
    Args:
        log_probs_N_K_C: torch.Tensor, the log probabilities. With the following ordering:
            N: Batch Size
            K: Monte-Carlo Samples
            C: Number of Classes
    Returns:
        entropies_N: torch.Tensor, entropy for each sample in batch
    """
    # N (batch size), K (MC), C (Classes)
    N, K, C = log_probs_N_K_C.shape

    mean_log_probs_N_C = torch.logsumexp(log_probs_N_K_C, dim=1) - math.log(K) # average over posterior samples
    nats_N_C = mean_log_probs_N_C * torch.exp(mean_log_probs_N_C)
    entropies_N = -torch.sum(nats_N_C, dim=1)

    return entropies_N


def get_bald(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """
    Return BALD score
    Args:
        log_probs_N_K_C: torch.Tensor, the log probabilities. With the following ordering:
            N: Batch Size
            K: Monte-Carlo Samples
            C: Number of Classes
    Returns:
        scores_N: torch.Tensor, bald scores for each sample in batch

        Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel. Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745, 2011.
    """
    # N (batch size), K (MC), C (Classes)
    scores_N = -compute_conditional_entropy(log_probs_N_K_C)
    scores_N += compute_entropy(log_probs_N_K_C)

    return scores_N


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
