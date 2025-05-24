import json
import pandas as pd
import torch
import os
from os.path import join as j_

from models.model_multimodal import coattn
from training.trainer import train_loop_survival, validate_survival
from utils.file_utils import save_pkl
from utils.losses import NLLSurvLoss
from utils.utils import (
    print_network,
    EarlyStopping,
    save_checkpoint,
    merge_dict,
    array2list,
    get_optim,
    get_lr_scheduler,
)
from wsi_datasets.genome_dataset import build_datasets


def main(model, datasets, es_metric, max_epochs, save_dir, device, early_stopping=True):
    print_network(model)
    optimizer = get_optim(model)
    epochs = 10
    lr_scheduler = get_lr_scheduler(epochs, optimizer, datasets['train'])
    es_patience = 5
    es_min_epochs = 3
    if early_stopping:
        early_stopper = EarlyStopping(save_dir=save_dir,
                                      patience=es_patience,
                                      min_stop_epoch=es_min_epochs,
                                      better='min' if es_metric == 'loss' else 'max',
                                      verbose=True)
    loss_fn = NLLSurvLoss(alpha=1e-4)

    for epoch in range(max_epochs):
        step_log = {'epoch': epoch, 'samples_seen': (
            epoch + 1) * len(datasets['train'].dataset)}

        # Train Loop
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        train_results = train_loop_survival(model=model,
                                            loader=datasets['train'],
                                            optimizer=optimizer,
                                            lr_scheduler=lr_scheduler,
                                            loss_fn=loss_fn,
                                            device=device)

        # Validation Loop (Optional)
        if 'val' in datasets.keys():
            print('#' * 11, f'VAL Epoch: {epoch}', '#' * 11)
            val_results, _ = validate_survival(
                model, datasets['val'], loss_fn, verbose=True)

            # Check Early Stopping (Optional)
            if early_stopper is not None:
                if es_metric == 'loss':
                    score = val_results['loss']

                else:
                    raise NotImplementedError
                config = {
                    'model': str(model),
                    'loss_fn': str(loss_fn),
                    'optimizer': str(optimizer),
                    'lr_scheduler': str(lr_scheduler),
                }
                save_ckpt_kwargs = dict(config=config,
                                        epoch=epoch,
                                        model=model,
                                        score=score,
                                        fname=f's_checkpoint.pth')
                stop = early_stopper(
                    epoch, score, save_checkpoint, save_ckpt_kwargs)
                if stop:
                    break
        print('#' * (22 + len(f'TRAIN Epoch: {epoch}')), '\n')

    # End of epoch: Load in the best model (or save the latest model with not early stopping)
    if early_stopping:
        model.load_state_dict(torch.load(
            j_(save_dir, f"s_checkpoint.pth"))['model'])
    else:
        torch.save(model.state_dict(), j_(save_dir, f"s_checkpoint.pth"))

    # End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
    for k, loader in datasets.items():
        print(f'End of training. Evaluating on Split {k.upper()}...:')
        return_attn = True  # True for MMP
        results[k], dumps[k] = validate_survival(
            model, loader, loss_fn, dump_results=True, return_attn=return_attn, verbose=False)

        if k == 'train':
            # Train results by default are not saved in the summary, but train dumps are
            _ = results.pop('train')

    # writer.close()
    return results, dumps


def create_gene_survival_model():

    model = coattn(path_proj_dim=256,
                   num_classes=4,  # using negative log-likelihood loss
                   num_coattn_layers=1,
                   modality='gene',
                   histo_agg='mean',
                   histo_model='mil',
                   )

    return model


if __name__ == "__main__":
    save_dir = './models/checkpoints'
    gene_path = './data_csvs/rna/rna_clean.csv'
    clinical_path = './data_csvs/TCGA_BRCA_overall_survival_k=0'

    print(
        f'Building datasets from...\n\tRNA data path:{gene_path}\n\tClincal Data Path:{clinical_path}\n')
    if os.path.isfile(gene_path):
        gene_df = pd.read_csv(gene_path, engine='python', index_col=0)
        assert 'Unnamed: 0' not in gene_df.columns

        gene_df = gene_df.reset_index()
        gene_df = gene_df.rename(columns={'index': 'case_id'})
    else:
        raise FileNotFoundError(f"{gene_path} not found!")

    try:
        # merge clinical data with gene data to get labels and features
        train_df = pd.read_csv(
            os.path.join(clinical_path, 'train.csv'), engine='python')
        test_df = pd.read_csv(
            os.path.join(clinical_path, 'test.csv'), engine='python')

        assert 'case_id' in train_df.columns, "case_id not found in train_df"
        assert 'case_id' in test_df.columns, "case_id not found in test_df"

        # remove suffix from sample to match test and train case_id
        gene_df['sample'] = gene_df['sample'].str[:-3]

        train_df = pd.merge(train_df, gene_df,
                            left_on='case_id', right_on='sample', how='left')

        test_df = pd.merge(test_df, gene_df, left_on='case_id',
                           right_on='sample', how='left')

    except FileNotFoundError:
        print(f"Train/Test split files not found in {clinical_path}.")
    except Exception as e:
        print(f"Error reading train/test split files: {e}")

    splits_csvs = {'train': train_df, 'test': test_df}

    dataset_splits = build_datasets(csv_splits=splits_csvs)
    print('\nDone!')

    all_results, all_dumps = {}, {}

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f'\nUsing device: {device}')
    print('\nCreating model...')
    model = create_gene_survival_model()
    model.to(device)
    print('Done!')

    print_network(model)

    fold_results, fold_dumps = main(model,
                                    dataset_splits,
                                    es_metric='loss',
                                    max_epochs=10,
                                    save_dir=save_dir,
                                    device=device)

    for split, split_results in fold_results.items():
        all_results[split] = merge_dict({}, split_results) if (
            split not in all_results.keys()) else merge_dict(all_results[split], split_results)
        # saves per-split, per-fold results to pkl
        save_pkl(
            j_(save_dir, f'{split}_results.pkl'), fold_dumps[split])

    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val)
                          for metric, val in split_results.items()})
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(save_dir, save_name), index=False)
    with open(j_(save_dir, save_name + '.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))

    dump_path = j_(save_dir, 'all_dumps.h5')
    save_pkl(dump_path, fold_dumps)
    print('\n\nDone!')
