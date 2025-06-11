import os
import pickle
from collections import OrderedDict

import torch
from torch import nn

from .staeformer import STAEformer


class T3STAEformer(nn.Module):
    def __init__(self, hist_len, pred_len, var_num, data_root, dataset_name, event_emb_dim=1024, hidden_event_emb_dim=256,
                 end_channels=512, pretrained_model_path=None, doy2embedding_path=None, dropout=0.1):
        super(T3STAEformer, self).__init__()
        self.event_emb_dim = event_emb_dim
        self.hidden_event_emb_dim = hidden_event_emb_dim
        self.pretrained_model_path = pretrained_model_path
        self.doy2embedding_path = os.path.join(data_root, doy2embedding_path)

        self.ts_encoder = STAEformer(
            var_num,
            hist_len=12,
            pred_len=12,
            freq=10,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            tod_embedding_dim=24,
            dow_embedding_dim=24,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=80,
            feed_forward_dim=256,
            num_heads=4,
            num_layers=3,
            dropout=0.1,
            use_mixed_proj=True
        )

        self.event_emb_transform = nn.Linear(self.event_emb_dim, self.hidden_event_emb_dim)
        self.mm_forecaster = nn.Linear(hist_len * (self.ts_encoder.model_dim + self.hidden_event_emb_dim), pred_len)
        self.ts_forecaster = nn.Linear(hist_len * self.ts_encoder.model_dim, pred_len)

        # self.load_pretrained_model()
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
            for _, _, embedding in doy2emb[doy]:
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
        ts_emb = self.ts_encoder.encoder_forward(var_x, marker_x).transpose(1, 2)  # (B, N, L, D)
        B, N, L, D = ts_emb.shape

        batch_doy = marker_x[:, -1, :, 3] * 365  # day of year [0 ~ 365]
        batch_event_emb = []
        for one_doy in batch_doy:
            event_emb = self.event_embeddings[self.doy2ind[int(one_doy)]]  # (-1, D)
            event_emb = self.event_emb_transform(event_emb.to(var_x.device)).max(dim=0, keepdim=True).values  # (-1, D)
            batch_event_emb.append(event_emb)
        batch_event_emb = torch.stack(batch_event_emb).reshape(B, 1, 1, self.hidden_event_emb_dim).repeat(1, N, L, 1)
        fused_emb = torch.cat((batch_event_emb, ts_emb), dim=-1)  # (B, N, L, D')

        ts_emb = ts_emb.reshape(B, N, -1)
        fused_emb = fused_emb.reshape(B, N, -1)

        mm_pred = self.mm_forecaster(fused_emb)
        ts_pred = self.ts_forecaster(ts_emb)

        return (ts_pred + mm_pred).transpose(1, 2)
