import torch
from itertools import product
from matplotlib import pyplot as plt

from baseline_genomic import train_loop
from model.genome_model import HallmarkSurvivalModel
from utils.get_dataset import OnlyGenomicDataset, convert_df_to_dataloader, convert_to_32_bit, get_final_rna_folds
from utils.training_utils import get_lr_scheduler, get_optim

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Define the grid of hyperparameters
param_grid = {
    'lr': [1e-3, 1e-2, 0.1],
    'batch_size': [16, 32, 64],
    'embedding_dim': [32, 64, 128],
    'n_filters': [16, 32, 64],
    'cnn_kernel_size': [3, 5],
    'optimizer': ['adamW', 'sgd']

}

# Create a list of all combinations of hyperparameters
hyperparameters = [
    dict(zip(param_grid.keys(), values))
    for values in product(*param_grid.values())
]


if __name__ == "__main__":
    datasets = get_final_rna_folds()[0]
    print(f"\n\nTotal hyperparameter combinations: {len(hyperparameters)}")
    print(f"Only using first fold as dataset.")
    loss_fn = torch.nn.CrossEntropyLoss()
    num_classes = 4

    test_accuracies = []
    c_indices = []
    train_losses = []

    feature_cols = list(datasets['train'].columns[1:-1])
    label_col = 'disc_label'

    if device.type == "mps":
        datasets['train'] = convert_to_32_bit(datasets['train'])
        datasets['test'] = convert_to_32_bit(datasets['test'])

    best_index_c = -1
    best_index_acc = -1
    best_c_index = -1
    best_acc = -1
    for i, params in enumerate(hyperparameters):

        lr = params['lr']
        batch_size = params['batch_size']
        embedding_dim = params['embedding_dim']
        n_filters = params['n_filters']
        cnn_kernel_size = params['cnn_kernel_size']
        optimizer = params['optimizer']

        print(f"Combination {i+1}:")
        print(f"\tlr: {lr}")
        print(f"\tbatch_size: {batch_size}")
        print(f"\tembedding_dim: {embedding_dim}")
        print(f"\tn_filters: {n_filters}")
        print(f"\tcnn_kernel_size: {cnn_kernel_size}")
        print(f"\toptimizer: {optimizer}")

        dataloader = convert_df_to_dataloader(
            datasets['train'], feature_cols, label_col, batch_size=batch_size)

        test_data = OnlyGenomicDataset(
            datasets['test'], feature_cols, label_col)

        model = HallmarkSurvivalModel(
            len(feature_cols),
            num_classes,
            hallmark_embedding_dim=embedding_dim,
            cnn_filters=n_filters,
            cnn_kernel_size=cnn_kernel_size,
        ).to(device)

        epochs = 30
        optimizer = get_optim(optimizer, model, lr=lr)
        lr_scheduler = get_lr_scheduler(epochs, optimizer, len(dataloader))

        train_loss, test_acc, c_index = train_loop(
            model, epochs, i, dataloader, test_data, loss_fn, optimizer, lr_scheduler)

        print(
            f'Permutation {i} had a test accuracy of {test_acc:.2%} and a c-index of {c_index:.4f}')
        print('\n========================\n\n')

        test_accuracies.append(test_acc)
        c_indices.append(c_index)
        train_losses.append(train_loss)

        if c_index > best_c_index:
            best_c_index = c_index
            best_index_c = i
        if test_acc > best_acc:
            best_acc = test_acc
            best_index_acc = i

    print('\n\n\n<====Grid Search Finished.====>\n\n')
    print(
        f'The best performance in terms of C-index ({best_c_index}) came from the model with the following parameters:'
    )

    c_params = hyperparameters[best_index_c]
    lr = c_params['lr']
    batch_size = c_params['batch_size']
    embedding_dim = c_params['embedding_dim']
    n_filters = c_params['n_filters']
    cnn_kernel_size = c_params['cnn_kernel_size']
    optimizer = c_params['optimizer']

    print(f"\tlr: {lr}")
    print(f"\tbatch_size: {batch_size}")
    print(f"\tembedding_dim: {embedding_dim}")
    print(f"\tn_filters: {n_filters}")
    print(f"\tcnn_kernel_size: {cnn_kernel_size}")
    print(f"\toptimizer: {optimizer}")

    print('\n\n')
    print(
        f'The best performance in terms of test accuracy ({best_acc}) came from the model with the following parameters:'
    )

    acc_params = hyperparameters[best_index_acc]
    lr = acc_params['lr']
    batch_size = acc_params['batch_size']
    embedding_dim = acc_params['embedding_dim']
    n_filters = acc_params['n_filters']
    cnn_kernel_size = acc_params['cnn_kernel_size']
    optimizer = acc_params['optimizer']

    print(f"\tlr: {lr}")
    print(f"\tbatch_size: {batch_size}")
    print(f"\tembedding_dim: {embedding_dim}")
    print(f"\tn_filters: {n_filters}")
    print(f"\tcnn_kernel_size: {cnn_kernel_size}")
    print(f"\toptimizer: {optimizer}")

    plt.figure(figsize=(10, 8))
    plt.plot(train_losses[best_index_acc], label='Best Test Accuracy Model')
    plt.plot(train_losses[best_index_c], label='Best C-Index Model')
    plt.legend()
    plt.title('Training losses for two best models')
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.show()
