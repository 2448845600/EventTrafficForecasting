import os

import numpy as np
import torch

from easytsf.runner.base_runner import BaseRunner
from easytsf.util.metrics import eval_metrics, masked_mae


class Runner(BaseRunner):
    def __init__(self, **kargs):
        super().__init__(**kargs)

        self.test_result = []

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        label = var_y[:, -self.hparams.pred_len:, :, 0]

        prediction = self.model(var_x, marker_x)[:, -self.hparams.pred_len:, :]

        rescaled_prediction = self.inverse_transform_var(prediction)
        rescaled_label = self.inverse_transform_var(label)
        rescaled_label_marker = self.inverse_transform_time_marker(marker_y)
        return prediction, label, rescaled_prediction, rescaled_label, rescaled_label_marker

    def training_step(self, batch, batch_idx):
        loss = self.loss_function(*self.forward(batch, batch_idx)[2:4])
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_function(*self.forward(batch, batch_idx)[2:4])
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        _, _, rescaled_prediction, rescaled_label, rescaled_label_marker = self.forward(batch, batch_idx)
        self.test_result.append({
            'prediction': rescaled_prediction.cpu(),
            'label': rescaled_label.cpu(),
            'time_marker': rescaled_label_marker.cpu(),
        })

    def on_test_epoch_end(self) -> None:
        prediction = torch.cat([batch['prediction'] for batch in self.test_result])
        labels = torch.cat([batch['label'] for batch in self.test_result])
        time_marker = torch.cat([batch['time_marker'] for batch in self.test_result])
        mae, rmse, mape, wape = eval_metrics(prediction, labels, null_val=0.0)
        self.log('test/mae', mae, on_step=False, on_epoch=True)
        self.log('test/rmse', rmse, on_step=False, on_epoch=True)
        self.log('test/mape', mape, on_step=False, on_epoch=True)

        np.savez(os.path.join(self.hparams.exp_dir, 'test_outputs.npz'),
                 prediction=prediction.numpy(), label=labels.numpy(), time_marker=time_marker.numpy())

    def configure_loss(self):
        self.loss_function = masked_mae
