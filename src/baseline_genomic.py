import matplotlib.pyplot as plt
import torch
from model.genome_model import HallmarkSurvivalModel, MultimodalHallmarkSurvivalModel
from utils.datasets import (
    GenomicAndClinicalDataset,
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
    print_average_metrics,
    print_network,
    train_loop,
)
import pandas as pd
# suppress Pandas Warnings to see output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(
    action='ignore', category=pd.errors.SettingWithCopyWarning)


def main(
    device: torch.device,
    run_multimodal: bool,
    display_loss_graph: bool = True,
    display_network: bool = False,
    save_model: bool = False,
):
    if run_multimodal:
        full_dataset, clinical_cols, rna_feature_cols = get_clinical_and_genomic_data()
        folds = get_merged_folds(full_dataset)
    else:
        folds = get_rna_folds()
        rna_feature_cols = list(folds[0]['train'].columns[1:-1])
        clinical_cols = []

    # Multi-class classification
    loss_fn = torch.nn.CrossEntropyLoss()
    NUM_CLASSES = 4

    # Track metrics
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

        if run_multimodal:
            train_data = GenomicAndClinicalDataset(
                fold['train'], rna_feature_cols, clinical_cols, label_col
            )
            dataloader = train_data.get_dataloader(batch_size=batchsize)

            test_data = GenomicAndClinicalDataset(
                fold['test'], rna_feature_cols, clinical_cols, label_col
            )

            model = MultimodalHallmarkSurvivalModel(
                M=len(rna_feature_cols),
                N_CLASSES=NUM_CLASSES,
                clinical_input_dim=len(clinical_cols)
            ).to(device)
        else:
            train_data = OnlyGenomicDataset(
                fold['train'], rna_feature_cols, label_col
            )
            dataloader = train_data.get_dataloader(batch_size=batchsize)

            test_data = OnlyGenomicDataset(
                fold['test'], rna_feature_cols, label_col
            )

            model = HallmarkSurvivalModel(
                len(rna_feature_cols),
                NUM_CLASSES,
                hallmark_embedding_dim=256,
                cnn_filters=32,
                cnn_kernel_size=3
            ).to(device)

        epochs = 30
        optimizer = get_optim('adamW', model, lr=1e-3)
        lr_scheduler = get_lr_scheduler(epochs, optimizer, len(dataloader))

        train_loss, test_acc, c_index = train_loop(
            model=model,
            epochs=epochs,
            fold_idx=fold_idx,
            dataloader=dataloader,
            test_data=test_data,
            loss_func=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            run_multimodal=run_multimodal,
            save_model=save_model,
        )

        print(
            f'\nFold {fold_idx} had a test accuracy of {test_acc:.2%} and a c-index of {c_index:.4f}')
        print('\n========================\n\n')

        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        c_indices.append(c_index)

        if (fold_idx == len(folds) - 1) and display_network:
            print_network(model)

    print_average_metrics(test_accuracies, c_indices)

    if display_loss_graph:
        plt.figure(figsize=(10, 8))
        for i, loss in enumerate(train_losses):
            plt.plot(loss, label=f'Fold {i}')
        plt.legend()
        plt.title('Training Losses for each fold')
        plt.ylabel('Training Loss')
        plt.xlabel('Iteration')
        plt.show()


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    main(device=device, run_multimodal=True)
