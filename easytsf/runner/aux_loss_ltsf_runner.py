import torch

from easytsf.runner.base_runner import BaseRunner
from easytsf.util.metrics import masked_mae


class Runner(BaseRunner):

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        label = var_y[:, -self.hparams.pred_len:, :, 0]
        prediction, aux_loss = self.model(var_x, marker_x)
        prediction = prediction[:, -self.hparams.pred_len:, :]
        return prediction, label, aux_loss

    def training_step(self, batch, batch_idx):
        prediction, label, aux_loss = self.forward(batch, batch_idx)
        pred_loss = self.loss_function(prediction, label)
        loss = pred_loss + aux_loss

        # find_unused_parameters
        # loss.backward(retain_graph=True)
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/pred_loss', pred_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/aux_loss', aux_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction, label, _ = self.forward(batch, batch_idx)
        mse = torch.nn.functional.mse_loss(prediction, label)
        self.log('val/loss', mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        prediction, label, _ = self.forward(batch, batch_idx)
        mae = torch.nn.functional.l1_loss(prediction, label)
        mse = torch.nn.functional.mse_loss(prediction, label)
        self.log('test/mae', mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mse', mse, on_step=False, on_epoch=True, sync_dist=True)

    def configure_loss(self):
        self.loss_function = masked_mae
