import numpy as np
import torch

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise

from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from utils.utils import (AverageMeter, safe_list_to)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def get_gene_data(batch, device):
    gene_data = []
    with open('./data_csvs/rna/rna_clean.csv', 'r') as f:
        first_line = f.readline().strip()
        gene_names = first_line.split(',')[2:]
    for key in gene_names:
        gene_data.append(torch.tensor(
            batch[key], dtype=torch.float32).to(device))
    return torch.tensor(gene_data).to(device)


def train_loop_survival(model, loader, optimizer, device, lr_scheduler, loss_fn=None,
                        print_every=50, accum_steps=32):
    model.train()
    meters = {'bag_size': AverageMeter()}
    all_risk_scores, all_censorships, all_event_times = [], [], []

    for batch_idx, batch in enumerate(loader.dataset):
        label = torch.tensor(batch['disc_label'],
                             dtype=torch.float32).to(device)

        # event_time = batch['survival_time'].to(device)
        # censorship = batch['censorship'].to(device)

        gene_data = get_gene_data(batch, device)

        out, log_dict = model(gene_data, label=label, loss_fn=loss_fn)

        if out['loss'] is None:
            continue

        # Get loss + backprop
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # End of iteration survival-specific metrics to calculate / log
        # all_risk_scores.append(out['risk'].detach().cpu().numpy())
        # all_censorships.append(censorship.cpu().numpy())
        # all_event_times.append(event_time.cpu().numpy())

        # for key, val in log_dict.items():
        #     if key not in meters:
        #         meters[key] = AverageMeter()
        #     meters[key].update(val, n=len(data))

        # bag_size_meter.update(data.size(1), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    return results


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_omic_attn, all_cross_attn, all_path_attn = [], [], []

    for batch_idx, batch in enumerate(loader):
        data = batch['img'].to(device)
        label = batch['label'].to(device)
        omics = safe_list_to(batch['omics'], device)

        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)
        attn_mask = batch['attn_mask'].to(device) if (
            'attn_mask' in batch) else None

        out, log_dict = model(data, omics, attn_mask=attn_mask, label=label,
                              censorship=censorship, loss_fn=loss_fn, return_attn=return_attn)
        # if return_attn:
        #     all_omic_attn.append(out['omic_attn'].detach().cpu().numpy())
        #     all_cross_attn.append(out['cross_attn'].detach().cpu().numpy())
        #     all_path_attn.append(out['path_attn'].detach().cpu().numpy())
        # # End of iteration survival-specific metrics to calculate / log
        # bag_size_meter.update(data.size(1), n=len(data))
        # for key, val in log_dict.items():
        #     if key not in meters:
        #         meters[key] = AverageMeter()
        #     meters[key].update(val, n=len(data))
        # all_risk_scores.append(out['risk'].cpu().numpy())
        # all_censorships.append(censorship.cpu().numpy())
        # all_event_times.append(event_time.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            print(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    if return_attn:
        if len(all_omic_attn[0].shape) == 2:
            all_omic_attn = np.stack(all_omic_attn)
            all_cross_attn = np.stack(all_cross_attn)
            all_path_attn = np.stack(all_path_attn)
        else:
            all_omic_attn = np.vstack(all_omic_attn)
            all_cross_attn = np.vstack(all_cross_attn)
            all_path_attn = np.vstack(all_path_attn)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits=torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(
                                     all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update(
            {k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        print("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
        if return_attn:
            dumps['all_omic_attn'] = all_omic_attn
            dumps['all_cross_attn'] = all_cross_attn
            dumps['all_path_attn'] = all_path_attn
    return results, dumps
