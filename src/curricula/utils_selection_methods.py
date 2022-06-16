import torch


def top_x_indices(vec, x, largest):
    """
    Returns the indices of the x largest/smallest entries in vec.

    Args:
        vec: tensor, number of samples to be selected
        x: int, number of indices to be returned
        smallest: bool, if true, the x largest entries are selected; if false,
        the x smallest entries are selected
    Returns:
        top_x_indices: tensor, top x indices, sorted
        other_indices: tensor, the indices that were not selected
    """

    sorted_idx = torch.argsort(vec, descending=largest)

    top_x_indices = sorted_idx[:x]
    other_indices = sorted_idx[x:]

    return top_x_indices, other_indices


def create_logging_dict(variables_to_log, selected_minibatch, not_selected_minibatch):
    """
    Create the dictionary for logging, in which, for each variable/metric, the
    selected and the not selected entries are logged separately as
    "selected_<var_name>" and "not_selected_<var_name>".

    Args:
        variables_to-log: dict, with key:var_name to be logged, value: tensor of values to be logger.
    """

    metrics_to_log = {}
    for name, metric in variables_to_log.items():
        metrics_to_log["selected_" + name] = metric[selected_minibatch].cpu().numpy()
        metrics_to_log["not_selected_" + name] = (
            metric[not_selected_minibatch].cpu().numpy()
        )
    return metrics_to_log
