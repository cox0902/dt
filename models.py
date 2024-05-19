from typing import *

import torch
from torch import nn
import torchvision


class VisModel(nn.Module):

    def __init__(self, model: Literal["resnet", "resnext"] = "resnet", load_weight: bool = False, copy_weight: bool = False,  
                 use_logits: bool = False):
        super(VisModel, self).__init__()
        self.use_logits = use_logits

        if model == "resnet":
            if load_weight:
                resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            else:
                resnet = torchvision.models.resnet50()
        elif model == "resnext":
            if load_weight:
                resnet = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
            else:
                resnet = torchvision.models.resnext50_32x4d()
        else:
            assert False, model
        
        old_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(in_channels=4, out_channels=old_conv1.out_channels, kernel_size=old_conv1.kernel_size, 
                              stride=old_conv1.stride, padding=old_conv1.padding, bias=old_conv1.bias)

        if load_weight:
            new_conv1.weight.data[:, :old_conv1.in_channels, :, :] = old_conv1.weight.data.clone()
            if copy_weight:
                new_conv1.weight.data[:, old_conv1.in_channels:, :, :] = old_conv1.weight.data.mean(dim=1, keepdim=True)
        else:
            assert not copy_weight

        resnet.conv1 = new_conv1
        sequential = [
            nn.Linear(in_features=2048, out_features=1),
        ]
        if not use_logits:
            sequential.append(nn.Sigmoid())
        resnet.fc = nn.Sequential(*sequential)
        self.resnet = resnet
        
    def forward(self, batch):
        logits = self.resnet(batch["image"])
        predicts = torch.sigmoid(logits) if self.use_logits else logits
        return logits, predicts, batch["target"]

    def predict(self, batch):
        with torch.no_grad():
            fextractor = nn.Sequential(*list(self.resnet.children())[:-1])
            classifier = self.resnet.fc
            features = fextractor(batch["image"]).squeeze()
            predicts = torch.sigmoid(classifier(features))
        return features, predicts


class VisCodeModel(nn.Module):

    def __init__(self, vis_model: VisModel, vocab_size: int, max_len: int, embedding_size: int, hidden_size: int, 
                 dropout: float = 0.):
        super(VisCodeModel, self).__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len

        self.vis_model = nn.Sequential(*list(vis_model.resnet.children())[:-1])
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm_cell = nn.LSTMCell(embedding_size + 2048 + 1, hidden_size, bias=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def inner_forward(self, batch):
        x = self.embedding(batch["code"])
        seq_lens = batch["code_len"]
        images = batch["image"]

        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)
        images_sort = torch.index_select(images, dim=0, index=idx_sort)

        h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        c = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)

        y_sort = torch.zeros(x.size(0), self.max_len, 1).to(x.device)
        x_prev = torch.ones(x.size(0), 1).to(x.device)

        lengths = seq_lens_sort.tolist()
        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])

            vis = self.vis_model(images_sort[:batch_size_t, t, :, :, :])
            print(vis.shape)
            vis = vis.squeeze()

            h, c = self.lstm_cell(torch.cat([x_sort[:batch_size_t, t, :], vis, x_prev[:batch_size_t, :]], dim=1), 
                                  (h[:batch_size_t], c[:batch_size_t]))

            out = self.fc(self.dropout(h))

            if self.training:
                y_sort[:batch_size_t, t, :] = out
            else:
                y_sort[:batch_size_t, t, :] = out
                y_sort[:batch_size_t, t, :][x_prev[:batch_size_t, :] <= 0.5] = -4.

            x_prev[:batch_size_t, :] = torch.sigmoid(out)

        logits = torch.index_select(y_sort, dim=0, index=idx_unsort)
        return logits
    
    def forward(self, batch):
        logits = self.inner_forward(batch)

        mask = batch["code"] != 0
        logits = torch.masked_select(logits.squeeze(), mask)
        targets = torch.masked_select(batch["target"], mask)

        predicts = torch.sigmoid(logits)
        return logits, predicts, targets

    def predict(self, batch):
        logits = self.inner_forward(batch).squeeze()
        predicts = torch.sigmoid(logits)

        mask = batch["code"] != 0
        masked = torch.all(mask * batch["target"] == mask * (predicts > 0.5), dim=-1)
        return { "corrects": masked }
