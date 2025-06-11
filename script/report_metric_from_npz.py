import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan, return_used_sample_num=False):
    """Masked mean absolute error.

    Args:
        return_used_sample_num:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        # mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
        mask = labels > null_val
    used_sample_num = mask.sum()
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if return_used_sample_num:
        return torch.mean(loss), used_sample_num
    else:
        return torch.mean(loss)


def masked_mape(prediction: torch.Tensor, target: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Masked mean absolute percentage error.

    Args:
        prediction (torch.Tensor): predicted values
        target (torch.Tensor): labels
        null_val (float, optional): null value.
                                    In the mape metric, null_val is set to 0.0 by all default.
                                    We keep this parameter for consistency, but we do not allow it to be changed.

    Returns:
        torch.Tensor: masked mean absolute percentage error
    """
    # we do not allow null_val to be changed
    null_val = 0.0
    # delete small values to avoid abnormal results
    # TODO: support multiple null values
    target = torch.where(torch.abs(target) < 1e-4, torch.zeros_like(target), target)
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.abs(prediction - target) / target)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked mean squared error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean squared error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
        # mask = labels > null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """root mean squared error.

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value . Defaults to np.nan.

    Returns:
        torch.Tensor: root mean squared error
    """

    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_wape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """Masked weighted absolute percentage error (WAPE)

    Args:
        preds (torch.Tensor): predicted values
        labels (torch.Tensor): labels
        null_val (float, optional): null value. Defaults to np.nan.

    Returns:
        torch.Tensor: masked mean absolute error
    """

    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        eps = 5e-5
        mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
    mask = mask.float()
    preds, labels = preds * mask, labels * mask
    loss = torch.sum(torch.abs(preds - labels)) / torch.sum(torch.abs(labels))
    return torch.mean(loss)


def eval_metrics(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan):
    mae = masked_mae(preds, labels, null_val=null_val)
    rmse = masked_rmse(preds, labels, null_val=null_val)
    mape = masked_mape(preds, labels) * 100
    wape = masked_wape(preds, labels) * 100
    return mae, rmse, mape, wape


def report_trial_result(trial_dir, reported_param=None, horizon="avg"):
    special_params, special_name = [], ''

    if reported_param is not None:
        trial_param_path = os.path.join(trial_dir, 'seed_0', 'hparams.yaml')
        with open(trial_param_path, 'r') as f:
            hparams = yaml.safe_load(f.read())
        for hparam_name in reported_param:
            special_params.append(hparams[hparam_name])

    results = []
    
    for version_name in os.listdir(trial_dir):
        if 'seed' in version_name:
            if not os.path.exists(os.path.join(trial_dir, version_name, 'test_outputs.npz')):
                return None, None
    
    for version_name in os.listdir(trial_dir):
        if 'seed' in version_name:
            # print(version_name)
            test_outputs = np.load(os.path.join(trial_dir, version_name, 'test_outputs.npz'))
            if horizon == "avg":
                prediction = test_outputs['prediction']
                label = test_outputs['label']
            elif horizon in ["@3", "@6", "@12"]:
                test_step = int(horizon[1:])
                prediction = test_outputs['prediction'][:, test_step-1:test_step, :]
                label = test_outputs['label'][:, test_step-1:test_step, :]
            else:
                raise ValueError(f"Invalid horizon: {horizon}")
            time_marker = test_outputs['time_marker'][:, :, 0, :]
            tod = time_marker[..., 0]
            eod = time_marker[..., 4]
            peak_mask = ((tod > 7 * 6 - 1) & (tod < 10 * 6)) | ((tod > 17 * 6 - 1) & (tod < 20 * 6))
            event_mask = eod > 0
            
            event_peak_mask = peak_mask & event_mask

            prediction = torch.tensor(prediction)
            label = torch.tensor(label)
            general_peak_mask = torch.tensor(peak_mask, dtype=torch.bool).unsqueeze(-1)
            event_peak_mask = torch.tensor(event_peak_mask, dtype=torch.bool).unsqueeze(-1)
            event_overall_mask = torch.tensor(event_mask, dtype=torch.bool).unsqueeze(-1)
            
            # general_overall_mae, general_overall_rmse, general_overall_mape, general_overall_wape = eval_metrics(prediction, label, null_val=0.0)
            # general_peak_mae, general_peak_rmse, general_peak_mape, general_peak_wape = eval_metrics(prediction*general_peak_mask, label*general_peak_mask, null_val=0.0)
            event_overall_mae, event_overall_rmse, event_overall_mape, event_overall_wape = eval_metrics(prediction*event_overall_mask, label*event_overall_mask, null_val=0.0)
            event_peak_mae, event_peak_rmse, event_peak_mape, event_peak_wape = eval_metrics(prediction*event_peak_mask, label * event_peak_mask, null_val=0.0)
            results.append([event_overall_mae, event_overall_rmse, event_overall_mape, 
                            event_peak_mae, event_peak_rmse, event_peak_mape,])

    result_df = pd.DataFrame(results, 
                             columns=[
                                      'event_overall_mae', 'event_overall_rmse', 'event_overall_mape',
                                      'event_peak_mae', 'event_peak_rmse', 'event_peak_mape'], index=None)
    result_df = result_df.astype(float)

    # print(result_df)
    
    # result_df.loc['mean', 'general_overall_mae'], result_df.loc['std', 'general_overall_mae'] = result_df.general_overall_mae.mean(), result_df.general_overall_mae.std()
    # result_df.loc['mean', 'general_overall_rmse'], result_df.loc['std', 'general_overall_rmse'] = result_df.general_overall_rmse.mean(), result_df.general_overall_rmse.std()
    # result_df.loc['mean', 'general_overall_mape'], result_df.loc['std', 'general_overall_mape'] = result_df.general_overall_mape.mean(), result_df.general_overall_mape.std()
    
    # result_df.loc['mean', 'general_peak_mae'], result_df.loc['std', 'general_peak_mae'] = result_df.general_peak_mae.mean(), result_df.general_peak_mae.std()
    # result_df.loc['mean', 'general_peak_rmse'], result_df.loc['std', 'general_peak_rmse'] = result_df.general_peak_rmse.mean(), result_df.general_peak_rmse.std()
    # result_df.loc['mean', 'general_peak_mape'], result_df.loc['std', 'general_peak_mape'] = result_df.general_peak_mape.mean(), result_df.general_peak_mape.std()
    
    result_df.loc['mean', 'event_overall_mae'], result_df.loc['std', 'event_overall_mae'] = result_df.event_overall_mae.mean(), result_df.event_overall_mae.std()
    result_df.loc['mean', 'event_overall_rmse'], result_df.loc['std', 'event_overall_rmse'] = result_df.event_overall_rmse.mean(), result_df.event_overall_rmse.std()
    result_df.loc['mean', 'event_overall_mape'], result_df.loc['std', 'event_overall_mape'] = result_df.event_overall_mape.mean(), result_df.event_overall_mape.std()
    
    result_df.loc['mean', 'event_peak_mae'], result_df.loc['std', 'event_peak_mae'] = result_df.event_peak_mae.mean(), result_df.event_peak_mae.std()
    result_df.loc['mean', 'event_peak_rmse'], result_df.loc['std', 'event_peak_rmse'] = result_df.event_peak_rmse.mean(), result_df.event_peak_rmse.std()
    result_df.loc['mean', 'event_peak_mape'], result_df.loc['std', 'event_peak_mape'] = result_df.event_peak_mape.mean(), result_df.event_peak_mape.std()
    # result_df.loc['mean', 'general_overall_wape'], result_df.loc['std', 'general_overall_wape'] = result_df.general_overall_wape.mean(), result_df.general_overall_wape.std()

    return result_df, special_params


def report_exp_abstract(exp_dir, reported_param, horizon):
    abstract = []
    abstract_header = ['config_hash'] + reported_param + [
                                      'event_overall_mae', 'event_overall_rmse', 'event_overall_mape',
                                      'event_peak_mae', 'event_peak_rmse', 'event_peak_mape']

    for config_hash in os.listdir(exp_dir):
        print(config_hash)
        config_cp_dir = os.path.join(exp_dir, config_hash)
        if not os.path.isdir(config_cp_dir):
            continue

        result_df, special_params = report_trial_result(config_cp_dir, reported_param, horizon)
        if result_df is None:
            print(config_hash)
            continue
        
        metric_mean_std = []
        latex_str = ""
        for metric in ['event_overall_mae', 'event_overall_rmse', 'event_overall_mape',
                        'event_peak_mae', 'event_peak_rmse', 'event_peak_mape']:
            # metric_mean_std.append("{:.2f}".format(result_df.loc['mean'][metric]) + "\\footnotesize{$\pm$" + "{:.3f}".format(result_df.loc['std'][metric]) + "}")
            # metric_mean_std.append('{:.2f}+{:.3f}'.format(result_df.loc['mean'][metric], result_df.loc['std'][metric]))
            metric_mean_std.append('{:.3f}'.format(result_df.loc['mean'][metric]))
            latex_str += "& {:.3f}".format(result_df.loc['mean'][metric]) + "$\pm$" + "{:.3f}".format(result_df.loc['std'][metric])
        print(latex_str) # [1:-1].replace(" & ", "\t& ")
        abstract.append([config_hash] + special_params + metric_mean_std)
    abstract_df = pd.DataFrame(abstract, columns=abstract_header)
    save_path = os.path.join(exp_dir, '{}_{}.csv'.format(exp_dir.split('/')[1], horizon))
    abstract_df.to_csv(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="D:\\Data\\Codes\\GAME\\ETF-data\\checkpoint\\Shenzhen\\STIDSSL", type=str, help="config save dir")
    parser.add_argument("--horizon", default="avg", type=str, help="avg")
    args = parser.parse_args()

    if args.exp_dir is not None:
        report_exp_abstract(
            exp_dir=args.exp_dir,
            reported_param=["lr"],
            # reported_param=["hidden_event_emb_dim", "event_fuse_type", "ssl_batch_size", "ssl_loss", "ssl_margin", "ssl_weight", "lr"],
            # reported_param=["lr", "dropout", "ssl_weight", "ssl_margin", "pretrained_seed"],
            # reported_param=["doy2embedding_path", "event_hidden_dim", "event_emb_compress_type", "event_fuse_type", "lr", "margin"],
            # reported_param=['use_event_gate', 'dropout', 'ssl_weight', 'lr', 'lrs_factor'],
            horizon=args.horizon, # "avg", "@3"
        )