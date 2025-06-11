import lightning.pytorch as pl
from torch.utils.data import DataLoader

from easytsf.data.dataset import data_provider


class DataInterface(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_workers = kwargs['num_workers']
        self.batch_size = kwargs['batch_size']
        self.kwargs = kwargs

    def train_dataloader(self):
        train_set = data_provider(self.kwargs, mode='train')
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        val_set = data_provider(self.kwargs, mode='valid')
        return DataLoader(val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        test_set = data_provider(self.kwargs, mode='test')
        return DataLoader(test_set, batch_size=1, num_workers=self.num_workers, shuffle=False)
