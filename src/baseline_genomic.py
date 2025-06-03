from model.genome_model import HallmarkSurvivalModel
from utils.get_dataset import (
    get_final_rna_folds,
    convert_df_to_dataloader,
    OnlyGenomicDataset,
    convert_to_32_bit
)
from utils.training_utils import (
    get_lr_scheduler,
    get_optim,
    print_network,
    save_checkpoint,
)
from typing import Iterable

import os
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train_loop(
    model: torch.nn.Module,
    epochs: int,
    fold_idx: int,
    dataloader: torch.utils.data.DataLoader,
    test_data: OnlyGenomicDataset,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    save_model: bool = False,
) -> tuple[list[float], float, float]:

    train_loss = []

    for epoch in range(epochs):
        model.train()

        running_loss = 0

        for features, label in tqdm.tqdm(dataloader):

            label = label.to(device)

            features = [feature.to(device) for feature in features]

            out = model(features)

            loss = loss_func(out, label)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()

            train_loss.append(loss.item())

            running_loss += loss.item()

            # insert validation?

        print(f'Epoch {epoch+1}/{epochs} Loss: {running_loss}')

    # find test_accuracy

    model.eval()
    with torch.no_grad():
        test_features, test_labels = test_data.get_features(), test_data.get_labels()
        test_preds = []  # To store output logits for each sample

        for sample_idx in range(len(test_features)):
            test_sample = test_features[sample_idx]
            # It needs to be a list of M tensors, where each tensor is shape (1, L_j)
            reshaped_input = []
            for hallmark in test_sample:
                # Shape: (L_j,)
                hallmark_tensor = torch.from_numpy(
                    hallmark).to(device)
                # Shape: (1, L_j)
                hallmark_tensor_batched = hallmark_tensor.unsqueeze(0)
                reshaped_input.append(hallmark_tensor_batched)

            output = model(reshaped_input)
            test_preds.append(output.cpu())

        y_pred = torch.cat(test_preds, dim=0)

        # Calculate c-index (use risk_score and true labels)
        high_risk_class_index = int(test_labels.max())
        risk_score = y_pred[:, high_risk_class_index]
        if test_labels is not None and len(np.unique(test_labels)) > 1:
            c_index = concordance_index(test_labels, risk_score)
        else:
            print("Cannot compute c-index (not enough classes in y_test)")
            c_index = -float('inf')

        # Calculate accuracy
        predicted_indices = torch.argmax(y_pred, dim=1).numpy()
        correct = (predicted_indices == test_labels.squeeze(1)).sum()

        total = test_labels.shape[0]
        test_acc = float(correct) / total if total > 0 else 0.0

    if save_model:
        save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        save_checkpoint(fold_idx, model, test_acc, save_dir)

    return train_loss, test_acc, c_index


if __name__ == "__main__":
    folds = get_final_rna_folds()

    loss_fn = torch.nn.CrossEntropyLoss()
    NUM_CLASSES = 4

    test_accuracies = []
    c_indices = []
    train_losses = []

    print(f'Training on device: {device}')
    if device == "cuda":
        print(f'Using {torch.cuda.device_count()}')

    print('\n\n')

    for fold_idx, fold in enumerate(folds):

        print(f'Training fold {fold_idx}...\n')

        batchsize = 16

        feature_cols = list(fold['train'].columns[1:-1])
        label_col = 'disc_label'

        if device.type == "mps":
            fold['train'] = convert_to_32_bit(fold['train'])
            fold['test'] = convert_to_32_bit(fold['test'])

        dataloader = convert_df_to_dataloader(
            fold['train'], feature_cols, label_col, batch_size=batchsize
        )

        test_data = OnlyGenomicDataset(fold['test'], feature_cols, label_col)

        model = HallmarkSurvivalModel(
            len(feature_cols),
            NUM_CLASSES,
            hallmark_embedding_dim=32,
            cnn_filters=64,
            cnn_kernel_size=3
        ).to(device)

        # print_network(model)

        epochs = 30
        optimizer = get_optim('adamW', model, lr=1e-2)
        lr_scheduler = get_lr_scheduler(epochs, optimizer, len(dataloader))

        train_loss, test_acc, c_index = train_loop(
            model, epochs, fold_idx, dataloader, test_data, loss_fn, optimizer, lr_scheduler)

        print(
            f'\nFold {fold_idx} had a test accuracy of {test_acc:.2%} and a c-index of {c_index:.4f}')
        print('\n========================\n\n')

        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        c_indices.append(c_index)

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

    plt.figure(figsize=(10, 8))
    for i, loss in enumerate(train_losses):
        plt.plot(loss, label=f'Fold {i}')
    plt.legend()
    plt.title('Training Losses for each fold')
    plt.ylabel('Training Loss')
    plt.xlabel('Iteration')
    plt.show()
