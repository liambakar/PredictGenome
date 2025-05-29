from utils.get_dataset import get_folded_rna_dataset, convert_df_to_dataloader, OnlyGenomicDataset
from utils.training_utils import get_lr_scheduler, get_optim, print_network, save_checkpoint
from model.genome_model import GenomePrediction
from model.temp_genome_model import Simple

import os
import torch
import tqdm
import numpy as np
from lifelines.utils import concordance_index

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def safe_list_to(data, device) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple):
        return (d.to(device) for d in data)  # type: ignore
    elif isinstance(data, list):
        return [d.to(device) for d in data]  # type: ignore
    elif isinstance(data, dict):
        return {k: v.to(device) for (k, v) in data.keys()}  # type: ignore
    else:
        raise RuntimeError("data should be a Tensor, tuple, list or dict, but"
                           " not {}".format(type(data)))


def train_loop(
    model: torch.nn.Module,
    epochs: int,
    dataloader: torch.utils.data.DataLoader,
    test_data: OnlyGenomicDataset,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[list[float], float, float]:
    train_loss = []
    for epoch in range(epochs):
        model.train()

        running_loss = 0

        for features, label in tqdm.tqdm(dataloader):

            features = safe_list_to(features, device)
            label = safe_list_to(label, device)

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
        test_features, test_labels = test_data.get_features(), test_data.get_labels().numpy()
        test_features = safe_list_to(test_features, device)
        outputs = model(test_features)
        predicted = np.argmax(outputs.cpu(), axis=1)
        correct = (predicted == test_labels).sum()
        total = test_labels.shape[0]
        test_acc = correct / total

        # calculate c_index
        c_index = float('inf')

    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint(fold_idx, model, test_acc, save_dir)

    return train_loss, test_acc, c_index


if __name__ == "__main__":
    folds = get_folded_rna_dataset()

    loss_fn = torch.nn.NLLLoss()
    NUM_CLASSES = 4
    EMBED_DIM = 128
    NUM_COATTENTION_QUERIES = 16
    NUM_ATTN_HEADS = 8
    NUM_TRANSFORMER_LAYERS = 4
    DIM_FEEDFORWARD = 512
    DROPOUT_RATE = 0.1

    test_accuracies = []

    train_losses = []

    print(f'Training on device: {device}')
    if device == "cuda":
        print(f'Using {torch.cuda.device_count()}')

    for fold_idx, fold in enumerate(folds):

        print(f'Training fold {fold_idx+1}...\n')

        batchsize = 32
        feature_cols = fold['train'].columns[2:]
        label_col = 'disc_label'

        dataloader = convert_df_to_dataloader(
            fold['train'], feature_cols, label_col, batch_size=batchsize
        )

        test_data = OnlyGenomicDataset(fold['test'], feature_cols, label_col)

        # model = GenomePrediction(
        #     num_classes=NUM_CLASSES,
        #     num_genes=len(feature_cols),
        #     embed_dim=EMBED_DIM,
        #     num_coattention_queries=NUM_COATTENTION_QUERIES,
        #     num_attn_heads=NUM_ATTN_HEADS,
        #     num_transformer_layers=NUM_TRANSFORMER_LAYERS,
        #     dim_feedforward=DIM_FEEDFORWARD,
        #     dropout=DROPOUT_RATE
        # ).to(device)

        model = Simple(len(feature_cols), NUM_CLASSES).to(device)

        # print_network(model)

        epochs = 45
        optimizer = get_optim('adamW', model)
        lr_scheduler = get_lr_scheduler(epochs, optimizer, len(dataloader))

        train_loss, test_acc, c_index = train_loop(
            model, epochs, dataloader, test_data, loss_fn, optimizer, lr_scheduler)

        print(
            f'\nFold {fold_idx + 1} had a test accuracy of {test_acc:.2%} and a c-index of {c_index:.4f}')
        print('\n========================\n\n')

        test_accuracies.append(test_acc)
        train_losses.append(train_loss)

    print(test_accuracies)
