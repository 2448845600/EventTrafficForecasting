import torch
import torch.nn as nn

from easytsf.runner.base_runner import BaseRunner


class Runner(BaseRunner):

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        label = var_y[:, -self.hparams.pred_len:, :, 0]
        prediction = self.model(var_x, marker_x)[:, -self.hparams.pred_len:, :]
        return prediction, label

    def training_step(self, batch, batch_idx):
        loss = self.loss_function(*self.forward(batch, batch_idx))

        # find_unused_parameters
        # loss.backward(retain_graph=True)
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss_function(*self.forward(batch, batch_idx))
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        prediction, label = self.forward(batch, batch_idx)
        mae = torch.nn.functional.l1_loss(prediction, label)
        mse = torch.nn.functional.mse_loss(prediction, label)
        self.log('test/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mse', mse, on_step=False, on_epoch=True, sync_dist=True)

    def configure_loss(self):
        self.loss_function = nn.MSELoss()
