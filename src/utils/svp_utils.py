import torch
import numpy as np
from src.models.SVPModels import ForgettingEventsModel
from tqdm import tqdm

def get_coreset(selection_method, model, dataloader, percent_train):
    """
    Returns sequence, which is np.array representing (len(dataset), )
    """
    model.eval()
    model = model.to(torch.device('cuda:0'))
    dataset_size = len(dataloader.dataset)
    selected_dataset_size = int(dataset_size * percent_train)

    sequence = np.asarray([])
    if selection_method == "uniform":
        sequence = np.random.permutation(dataset_size)[:selected_dataset_size]
    elif selection_method == "forgetting":
        if not isinstance(model, ForgettingEventsModel):
            raise TypeError("Model must be ForgettingEventsModel ")
        correct, n_forgotten, was_correct = (
            model.correct,
            model.n_forgotten,
            model.was_correct,
        )
        n_forgotten[~was_correct] = np.inf
        ranked = n_forgotten.argsort()[::-1]
        sequence = ranked[:selected_dataset_size]
    else:
        preds = []
        global_indices = []
        with torch.inference_mode():
            for batch_idx, (index, data, target) in enumerate(tqdm(dataloader)):
                index = index.to(model.device)
                data = data.to(model.device)
                target = target.to(model.device)

                output = model(data).softmax(dim=1)
                preds.append(output.detach().cpu())
                global_indices.append(index.detach().cpu())

        preds = torch.cat(preds).numpy()
        global_indices = torch.cat(global_indices).numpy()
        if selection_method == "least_confidence":
            probs = preds.max(axis=1)
            indices = probs.argsort(axis=0)
        elif selection_method in "entropy":
            entropy = (np.log(preds) * preds).sum(axis=1) * -1.0
            indices = entropy.argsort(axis=0)[::-1]
        else:
            raise NotImplementedError(f"'{selection_method}' method doesn't exist")
        ranked = global_indices[indices]  # Map back to original indices
        sequence = ranked[:selected_dataset_size]

    return sequence
