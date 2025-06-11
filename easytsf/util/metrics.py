import numpy as np
import torch
from torchmetrics import Metric


class MaskedMAE(Metric):
    def __init__(self, null_val: float = 0.0, eps: float = 5e-5):
        super().__init__()
        self.add_state("masked_mae", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.null_val = torch.tensor(null_val)
        self.eps = eps

    def _get_mask(self, target: torch.Tensor):
        if np.isnan(self.null_val):
            mask = ~torch.isnan(target)
        else:
            mask = ~torch.isclose(target, self.null_val.expand_as(target).to(target.device), atol=self.eps, rtol=0.)
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        return mask

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        we only count the mae of masked area
        Args:
            preds:
            target:

        Returns:

        """
        assert preds.shape == target.shape

        mask = self._get_mask(target)
        mask_num = mask.sum()

        loss = torch.abs(preds - target) * mask

        self.masked_mae += torch.sum(loss)
        self.total += mask_num

    def compute(self):
        return self.masked_mae / self.total


class MaskedRMSE(Metric):
    def __init__(self, null_val: float = 0.0, eps: float = 5e-5):
        super().__init__()
        self.add_state("masked_mse", default=torch.tensor(0.).double(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.null_val = torch.tensor(null_val)
        self.eps = eps

    def _get_mask(self, target: torch.Tensor):
        if np.isnan(self.null_val):
            mask = ~torch.isnan(target)
        else:
            mask = ~torch.isclose(target, self.null_val.expand_as(target).to(target.device), atol=self.eps, rtol=0.)
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        return mask

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        mask = self._get_mask(target)
        mask_num = mask.sum()
        loss = torch.square(torch.sub(preds, target)) * mask

        self.masked_mse += torch.sum(loss).double()
        self.total += mask_num

    def compute(self):
        return torch.sqrt(self.masked_mse.float() / self.total)


class MaskedMAPE(Metric):
    def __init__(self, null_val: float = 0.0, eps: float = 5e-5):
        super().__init__()
        self.add_state("masked_mape", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.null_val = torch.tensor(null_val)
        self.eps = eps

    def _get_mask(self, target: torch.Tensor):
        if np.isnan(self.null_val):
            mask = ~torch.isnan(target)
        else:
            mask = ~torch.isclose(target, self.null_val.expand_as(target).to(target.device), atol=self.eps, rtol=0.)
            mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
        return mask

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        we only count the mae of masked area
        Args:
            preds:
            target:

        Returns:

        """
        assert preds.shape == target.shape

        mask = self._get_mask(target)
        mask_num = mask.sum()

        loss = torch.abs(torch.abs(preds - target) / target) * mask
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

        self.masked_mape += torch.sum(loss) * 100
        self.total += mask_num

    def compute(self):
        return self.masked_mape / self.total


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
        # mask = ~torch.isclose(labels, torch.tensor(null_val).expand_as(labels).to(labels.device), atol=eps, rtol=0.)
        mask = labels > null_val
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


def eval_metrics_detail(preds: torch.Tensor, labels: torch.Tensor, null_val: float = np.nan, pred_len=12):
    """_summary_

    Args:
        preds (torch.Tensor):  (B, output_len, N, 1)
        labels (torch.Tensor):  (B, output_len, N, 1)
        null_val (float, optional): _description_. Defaults to np.nan.
        horizon (list, optional): _description_. Defaults to [3, 6, 12].
        metric_name (list, optional): _description_. Defaults to .
    """
    detail_report = []
    for h in range(pred_len):
        h_preds, h_labels = preds[:, h, :], labels[:, h, :]
        h_masked_mae = masked_mae(h_preds, h_labels, null_val=null_val)
        h_masked_rmse = masked_rmse(h_preds, h_labels, null_val=null_val)
        h_masked_mape = masked_mape(h_preds, h_labels) * 100
        h_masked_wape = masked_wape(h_preds, h_labels) * 100
        detail_report.append(['@{}'.format(h + 1), h_masked_mae.cpu().numpy(), h_masked_rmse.cpu().numpy(),
                              h_masked_mape.cpu().numpy(), h_masked_wape.cpu().numpy()])

    avg_mmae = masked_mae(preds, labels, null_val=null_val)
    avg_mrmse = masked_rmse(preds, labels, null_val=null_val)
    avg_mmape = masked_mape(preds, labels) * 100
    avg_mwape = masked_wape(preds, labels) * 100
    detail_report.append(
        ['avg', avg_mmae.cpu().numpy(), avg_mrmse.cpu().numpy(), avg_mmape.cpu().numpy(), avg_mwape.cpu().numpy()])
    return avg_mmae, avg_mrmse, avg_mmape, avg_mwape, detail_report
