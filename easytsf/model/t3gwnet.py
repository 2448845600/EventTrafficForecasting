import os
import pickle
from collections import OrderedDict

import torch
from torch import nn

from .gwnet import GWNet


class T3GWNet(nn.Module):
    def __init__(self, pred_len, var_num, data_root, dataset_name, event_emb_dim=1024, hidden_event_emb_dim=256,
                 end_channels=512, pretrained_model_path=None, doy2embedding_path=None):
        super(T3GWNet, self).__init__()
        self.event_emb_dim = event_emb_dim
        self.hidden_event_emb_dim = hidden_event_emb_dim
        self.pretrained_model_path = pretrained_model_path
        self.doy2embedding_path = os.path.join(data_root, doy2embedding_path)

        self.ts_encoder = GWNet(pred_len, var_num, data_root, dataset_name, in_dim=2)
        self.event_emb_transform = nn.Linear(self.event_emb_dim, self.hidden_event_emb_dim)
        self.mm_forecaster = nn.Sequential(
            nn.Conv1d(in_channels=end_channels + hidden_event_emb_dim, out_channels=end_channels,
                      kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv1d(in_channels=end_channels, out_channels=pred_len, kernel_size=1, bias=True),
        )
        self.ts_forecaster = nn.Sequential(
            nn.Conv1d(in_channels=end_channels, out_channels=end_channels, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Conv1d(in_channels=end_channels, out_channels=pred_len, kernel_size=1, bias=True),
        )

        self.load_pretrained_model()
        self.doy2ind, self.event_embeddings = self.load_event_embedding()

    def load_pretrained_model(self):
        state_dict = torch.load(self.pretrained_model_path)["state_dict"]
        new_state_dict = OrderedDict()
        for k in state_dict:
            if "model." in k:
                new_state_dict[k.replace("model.", "")] = state_dict[k]
        self.ts_encoder.load_state_dict(new_state_dict)
        for param in self.ts_encoder.parameters():
            param.requires_grad = False

    def load_event_embedding(self):
        with open(self.doy2embedding_path, "rb") as f:
            doy2emb = pickle.load(f)
        doy2ind = {}
        embeddings = [torch.zeros(self.event_emb_dim)]  # Todo: 没有事件，设为 0 合理吗？
        emb_ind = 0
        for doy in doy2emb:
            for _, _, _, embedding in doy2emb[doy]:
                emb_ind += 1
                if doy not in doy2ind:
                    doy2ind[doy] = []
                doy2ind[doy].append(emb_ind)
                embeddings.append(torch.tensor(embedding))
        for i in range(366):
            if i not in doy2ind:
                doy2ind[i] = [0]
        embeddings = torch.stack(embeddings)
        return doy2ind, embeddings

    def forward(self, var_x, marker_x) -> torch.Tensor:
        ts_emb = self.ts_encoder.encoder_forward(var_x, marker_x)  # (B, D, N, 1)
        B, _, N = ts_emb.shape

        batch_doy = marker_x[:, -1, :, 3] * 365  # day of year [0 ~ 365]
        batch_event_emb = []
        for one_doy in batch_doy:
            event_emb = self.event_embeddings[self.doy2ind[int(one_doy)]]  # (-1, D)
            event_emb = self.event_emb_transform(event_emb.to(var_x.device)).max(dim=0, keepdim=True).values  # (-1, D)
            batch_event_emb.append(event_emb)
        batch_event_emb = torch.stack(batch_event_emb).reshape(B, -1, 1).repeat(1, 1, N)  # (B, D, N, 1)
        fused_emb = torch.cat((batch_event_emb, ts_emb), dim=1)

        mm_pred = self.mm_forecaster(fused_emb)
        ts_pred = self.ts_forecaster(ts_emb)
        return ts_pred + mm_pred
