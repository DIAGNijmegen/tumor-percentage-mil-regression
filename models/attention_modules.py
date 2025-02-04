import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, n_input, n_latent, n_classes=2, dropout=False, dp_rate=0):
        super(Attention, self).__init__()
        self.module = [nn.Linear(n_input, n_latent), torch.nn.Tanh()]
        if dropout:
            self.module.append(nn.Dropout(dp_rate))
        self.module.append(nn.Linear(n_latent, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x


class AttentionGated(nn.Module):
    def __init__(self, n_input, n_latent, n_classes=2, dropout=False, dp_rate=0):
        super(AttentionGated, self).__init__()
        self.attention_a = [nn.Linear(n_input, n_latent), torch.nn.Tanh()]
        self.attention_b = [nn.Linear(n_input, n_latent), torch.nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dp_rate))
            self.attention_b.append(nn.Dropout(dp_rate))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.fc = nn.Linear(n_latent, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.fc(A)
        return A, x
