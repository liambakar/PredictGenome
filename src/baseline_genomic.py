from utils.get_dataset import get_folded_rna_dataset, convert_df_to_dataloader
from utils.training_utils import get_lr_scheduler, get_optim, print_network, save_checkpoint
from model.genome_model import GenomePrediction

import torch
import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def safe_list_to(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple):
        return (d.to(device) for d in data)
    elif isinstance(data, list):
        return [d.to(device) for d in data]
    elif isinstance(data, dict):
        return {k: v.to(device) for (k, v) in data.keys()}
    else:
        raise RuntimeError("data should be a Tensor, tuple, list or dict, but"
                           " not {}".format(type(data)))


def train_model(
    model: torch.nn.Module,
    epochs: int, dataloader:
    torch.utils.data.DataLoader,
    loss_func: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    train_loss = []
    for epoch in range(epochs):
        model.train()

        running_loss = 0

        for features, label in tqdm.tqdm(dataloader):

            features = safe_list_to(features, device)
            label.to(device)

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
        if epoch % 10 == 0:
            save_checkpoint(epoch, model, running_loss, 'save_directory')
    return train_loss


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

    for fold in folds:

        batchsize = 32
        feature_cols = fold['train'].columns[2:]
        label_col = 'disc_label'

        dataloader = convert_df_to_dataloader(
            fold['train'], feature_cols, label_col, batch_size=batchsize
        )

        model = GenomePrediction(
            num_classes=NUM_CLASSES,
            num_genes=len(feature_cols),
            embed_dim=EMBED_DIM,
            num_coattention_queries=NUM_COATTENTION_QUERIES,
            num_attn_heads=NUM_ATTN_HEADS,
            num_transformer_layers=NUM_TRANSFORMER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT_RATE
        ).to(device)

        # print_network(model)

        epochs = 50
        optimizer = get_optim('adamW', model)
        lr_scheduler = get_lr_scheduler(epochs, optimizer, len(dataloader))

        train_loss = train_model(
            model, epochs, dataloader, loss_fn, optimizer, lr_scheduler)
