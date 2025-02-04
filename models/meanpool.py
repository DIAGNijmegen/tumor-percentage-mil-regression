import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPoolInstance(nn.Module):
    def __init__(self, size_arg="resnet50", n_classes=2, dropout=False, dp_rate=0):
        super(MeanPoolInstance, self).__init__()
        self.dropout = dropout
        self.dp_rate = dp_rate
        self.size_dict = {"resnet50": [1024, 512, 256]}
        size = self.size_dict[size_arg]
        self.n_classes = n_classes
        self.L = size[1]
        self.D = size[2]
        self.encoder = [nn.Linear(size[0], self.L), nn.ReLU()]
        if self.dropout:
            self.encoder.append(nn.Dropout(self.dp_rate))
        self.encoder = nn.Sequential(*self.encoder)
        self.final_fc = nn.Linear(self.L, n_classes)

    def forward(self, x):
        embeddings = self.encoder(x)
        instance_scores = self.final_fc(embeddings)
        preds = torch.mean(instance_scores)
        return preds, instance_scores, None, None