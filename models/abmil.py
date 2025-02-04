import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_modules import Attention, AttentionGated


class ABMIL(nn.Module):
    def __init__(self, size_arg="resnet50", n_classes=2, gate=True, dropout=False, dp_rate=0):
        super(ABMIL, self).__init__()
        self.gate = gate
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
        if gate:
            self.attention_net = AttentionGated(self.L, self.D, 1, self.dropout, self.dp_rate)
        else:
            self.attention_net = Attention(self.L, self.D, 1, self.dropout, self.dp_rate)
        self.final_fc = nn.Linear(self.L, n_classes)

    def forward(self, x):
        embeddings = self.encoder(x)
        attention_scores_raw, _ = self.attention_net(embeddings)
        attention_scores = torch.softmax(attention_scores_raw, dim=1)
        weighted_embedding_sums = (attention_scores * embeddings).sum(-2)
        preds = self.final_fc(weighted_embedding_sums)
        return preds, attention_scores, attention_scores_raw, None
    
class ABMIL_Instance(nn.Module):
    def __init__(self, size_arg="resnet50", n_classes=2, gate=True, dropout=False, dp_rate=0):
        super(ABMIL_Instance, self).__init__()
        self.gate = gate
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
        if gate:
            self.attention_net = AttentionGated(self.L, self.D, 1, self.dropout, self.dp_rate)
        else:
            self.attention_net = Attention(self.L, self.D, 1, self.dropout, self.dp_rate)
        self.final_fc = nn.Linear(self.L, n_classes)

    def forward(self, x):
        embeddings = self.encoder(x)
        attention_scores_raw, _ = self.attention_net(embeddings)
        attention_scores = torch.softmax(attention_scores_raw, dim=1)
        instance_scores = self.final_fc(embeddings)
        preds = (instance_scores * attention_scores).sum(dim = 1) / (attention_scores).sum(dim = 1)
        return preds, instance_scores, attention_scores_raw, None

