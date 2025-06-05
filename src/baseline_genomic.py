import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import os
from model.genome_model import HallmarkSurvivalModel, MultimodalHallmarkSurvivalModel
from utils.datasets import (
    GenomicAndClinicalDataset,
    PDataset,
    OnlyGenomicDataset
)
from utils.get_dataset import (
    get_clinical_and_genomic_data,
    get_merged_folds,
    get_rna_folds,
    convert_to_32_bit
)
from utils.training_utils import (
    get_lr_scheduler,
    get_optim,
    save_checkpoint,
    calculate_c_index,
)
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(
    action='ignore', category=pd.errors.SettingWithCopyWarning)


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
    test_data: PDataset,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    save_model: bool = False,
    run_multimodal: bool = False,
) -> tuple[list[float], float, float]:

    train_loss = []

    for epoch in range(epochs):
        model.train()

        running_loss = 0

        for batch in tqdm.tqdm(dataloader):
            if run_multimodal:
                features, clinical_features, label = batch
                features = [feature.to(device) for feature in features]
                clinical_features = clinical_features.to(device)
                out = model(features, clinical_features)
            else:
                features, label = batch
                features = [feature.to(device) for feature in features]
                out = model(features)
            label = label.to(device)

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
                    hallmark).to(device)
                # Shape: (1, L_j)
                hallmark_tensor_batched = hallmark_tensor.unsqueeze(0)
                reshaped_input.append(hallmark_tensor_batched)

            if run_multimodal:
                output = model(
                    reshaped_input,
                    torch.tensor(
                        test_clinical_features[sample_idx],  # type: ignore
                        device=device
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
        save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        save_checkpoint(fold_idx, model, test_acc, save_dir)

    return train_loss, test_acc, c_index


if __name__ == "__main__":
    full_dataset, clinical_cols, rna_feature_cols = get_clinical_and_genomic_data()
    folds = get_merged_folds(full_dataset)

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

        batchsize = 64

        label_col = 'disc_label'

        if device.type == "mps":
            fold['train'] = convert_to_32_bit(fold['train'])
            fold['test'] = convert_to_32_bit(fold['test'])

        train_data = GenomicAndClinicalDataset(
            fold['train'], rna_feature_cols, clinical_cols, label_col
        )
        dataloader = train_data.get_dataloader(batch_size=batchsize)

        test_data = GenomicAndClinicalDataset(
            fold['test'], rna_feature_cols, clinical_cols, label_col
        )

        # model = HallmarkSurvivalModel(
        #     len(rna_feature_cols),
        #     NUM_CLASSES,
        #     hallmark_embedding_dim=256,
        #     cnn_filters=32,
        #     cnn_kernel_size=3
        # ).to(device)

        model = MultimodalHallmarkSurvivalModel(
            M=len(rna_feature_cols),
            N_CLASSES=NUM_CLASSES,
            clinical_input_dim=len(clinical_cols)
        ).to(device)

        # print_network(model)

        epochs = 30
        optimizer = get_optim('adamW', model, lr=1e-3)
        lr_scheduler = get_lr_scheduler(epochs, optimizer, len(dataloader))

        train_loss, test_acc, c_index = train_loop(
            model, epochs, fold_idx, dataloader, test_data, loss_fn, optimizer, lr_scheduler, run_multimodal=True)

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
