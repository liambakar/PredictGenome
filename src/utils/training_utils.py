import torch.optim as optim
import torch
import os
import numpy as np
from lifelines.utils import concordance_index
from typing import Iterable
import tqdm
from transformers.optimization import (get_constant_schedule_with_warmup,
                                       get_linear_schedule_with_warmup,
                                       get_cosine_schedule_with_warmup)

from utils.datasets import PDataset


def get_lr_scheduler(epochs: int,
                     optimizer: optim.Optimizer,
                     dataloader_length: int,
                     scheduler_name: str = 'linear',
                     warmup_epochs: int = 1) -> optim.lr_scheduler.LRScheduler:
    accum_steps = 1
    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * (dataloader_length // accum_steps)
    else:
        warmup_steps = 0

    if scheduler_name == 'constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                         num_warmup_steps=warmup_steps)
    elif scheduler_name == 'cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=warmup_steps,
                                                       num_training_steps=(
                                                           dataloader_length // accum_steps * epochs),
                                                       )
    elif scheduler_name == 'linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=(dataloader_length // accum_steps) * epochs,
        )
    else:
        raise NotImplementedError
    return lr_scheduler


def get_optim(opt: str, model: torch.nn.Module, lr: float = 1e-3) -> optim.Optimizer:
    parameters = model.parameters()
    if opt == "adamW":
        optimizer = optim.AdamW(parameters, lr=lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=0.9)
    elif opt == 'RAdam':
        optimizer = optim.RAdam(parameters, lr=lr)
    else:
        raise NotImplementedError
    return optimizer


def print_network(net: torch.nn.Module) -> None:
    num_params = 0
    num_params_train = 0

    print(str(net))

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print(f'Total number of parameters: {num_params}')
    print(f'Total number of trainable parameters: {num_params_train}')


def save_checkpoint(fold: int, model: torch.nn.Module, score: float, save_dir: str) -> None:
    save_state = {
        'model': model.state_dict(),
        'score': score,
        'fold': fold
    }
    save_path = os.path.join(save_dir, f'ckpt_fold_{fold}.pth')

    torch.save(save_state, save_path)


def calculate_c_index(y_pred: np.ndarray | torch.Tensor, y_true: np.ndarray):
    """
    Calculates the concordance index (c-index) for survival predictions.
    The c-index is a measure of the predictive accuracy of a risk score, commonly used in survival analysis.
    This function extracts the risk scores for the highest risk class and computes the c-index using the true labels.
    Args:
        y_pred (np.ndarray | torch.Tensor): Predicted probabilities or risk scores, shape (n_samples, n_classes).
        y_true (np.ndarray): True class labels or survival outcomes, shape (n_samples,).
    Returns:
        float: The concordance index if computable; otherwise, returns negative infinity and prints a warning.
    Notes:
        - Requires at least two unique classes in y_true to compute the c-index.
        - Assumes that the highest risk class corresponds to the maximum value in y_true.
    """

    high_risk_class_index = int(y_true.max())
    risk_score = y_pred[:, high_risk_class_index]
    if y_true is not None and len(np.unique(y_true)) > 1:
        c_index = concordance_index(y_true, risk_score)
    else:
        print("Cannot compute c-index (not enough classes in y_test)")
        c_index = -float('inf')
    return c_index


def train_loop(
    model: torch.nn.Module,
    epochs: int,
    fold_idx: int,
    dataloader: torch.utils.data.DataLoader,
    test_data: PDataset,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    save_model: bool = False,
    run_multimodal: bool = False,
) -> tuple[list[float], float, float]:
    """
    Trains a PyTorch model for a specified number of epochs and evaluates its performance on a test dataset.

    Args:
        model (torch.nn.Module): The neural network model to train.
        epochs (int): Number of training epochs.
        fold_idx (int): Index of the current fold (used for checkpoint naming).
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        test_data (PDataset): Dataset object containing test data and labels.
        loss_func (torch.nn.Module): Loss function to optimize.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        lr_scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run computations on (CUDA, MPS, or CPU).
        save_model (bool, optional): Whether to save the model checkpoint after training. Defaults to False.
        run_multimodal (bool, optional): Whether to use multimodal input (genomic and clinical features). Defaults to False.

    Returns: 
        List of training losses per epoch, list of test accuracies, and a list of concordance indices (c-index) on the test set."""

    train_loss = []

    for epoch in range(epochs):
        model.train()

        running_loss = 0

        for batch in tqdm.tqdm(dataloader):
            if run_multimodal:
                features, clinical_features, label = batch
                features = [feature.to(device).float() for feature in features]
                clinical_features = clinical_features.to(device).float()
                out = model(features, clinical_features)
            else:
                features, label = batch
                features = [feature.to(device).float() for feature in features]
                out = model(features)
            label = label.to(device).long()

            loss = loss_func(out, label)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()

            running_loss += loss.item()

            # insert validation?

        print(f'Epoch {epoch+1}/{epochs} Loss: {running_loss}')
        train_loss.append(running_loss)

    # find test_accuracy

    model.eval()
    with torch.no_grad():
        if run_multimodal:
            test_genomic_features, test_clinical_features = test_data.get_features()
        else:
            test_genomic_features = test_data.get_features()

        test_labels = test_data.get_labels()
        test_preds = []  # To store output logits for each sample

        for sample_idx in range(len(test_genomic_features)):
            test_sample = test_genomic_features[sample_idx]
            # It needs to be a list of M tensors, where each tensor is shape (1, L_j)
            reshaped_input = []
            for hallmark in test_sample:
                # Shape: (L_j,)
                hallmark_tensor = torch.from_numpy(
                    hallmark).to(device).float()
                # Shape: (1, L_j)
                hallmark_tensor_batched = hallmark_tensor.unsqueeze(0)
                reshaped_input.append(hallmark_tensor_batched)

            if run_multimodal:
                output = model(
                    reshaped_input,
                    torch.tensor(
                        test_clinical_features[sample_idx],  # type: ignore
                        device=device,
                        dtype=torch.float32
                    ).unsqueeze(dim=0)
                )
            else:
                output = model(reshaped_input)
            test_preds.append(output.cpu())

        y_pred = torch.cat(test_preds, dim=0)

        # Calculate c-index (use risk_score and true labels)
        c_index = calculate_c_index(y_pred, test_labels)

        # Calculate accuracy
        predicted_indices = torch.argmax(y_pred, dim=1).numpy()
        correct = (predicted_indices == test_labels.squeeze(1)).sum()

        total = test_labels.shape[0]
        test_acc = float(correct) / total if total > 0 else 0.0

    if save_model:
        save_dir = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        save_checkpoint(fold_idx, model, test_acc, save_dir)

    return train_loss, test_acc, c_index


def print_average_metrics(test_accuracies: Iterable, c_indices: Iterable) -> None:
    """Prints the average Test Accuracy and Test C-index plus/minus their standard deviations.

    Args:
        test_accuracies: An iterable of test accuracies for each fold.
        c_indices: An iterable of c-indices for each fold.
    """
    test_accuracies = np.array(test_accuracies)
    mean_acc = np.mean(
        test_accuracies) if test_accuracies.size > 0 else 0
    std_acc = np.std(
        test_accuracies) if test_accuracies.size > 0 else 0
    mean_acc_percent = mean_acc * 100
    std_acc_percent = std_acc * 100
    format_string = f"{{:.{2}f}}%"
    mean_str = format_string.format(mean_acc_percent)
    std_str = format_string.format(std_acc_percent)

    c_indices = np.array(c_indices)
    mean_c_index = np.mean(
        c_indices) if c_indices.size > 0 else 0
    std_c_index = np.std(
        c_indices) if c_indices.size > 0 else 0

    # print(f"Fold Accuracies: {test_accuracies}")
    print(f"Mean Test Accuracy: {mean_str} ± {std_str}")
    # print(f"Fold C-indices: {c_indices}")
    print(f"Mean C-Index: {mean_c_index:.4f} ± {std_c_index:.4f}")
    print(f"(K={len(test_accuracies)} folds)")
