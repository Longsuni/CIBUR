import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import math
class Scheduler:
    def __call__(self, **kwargs):
        raise NotImplemented()


class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (iteration - self.start_iteration) * self.m + self.start_value


class ExponentialScheduler(LinearScheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0, base=10):
        self.base = base

        super(ExponentialScheduler, self).__init__(start_value=math.log(start_value, base),
                                                   end_value=math.log(end_value, base),
                                                   n_iterations=n_iterations,
                                                   start_iteration=start_iteration)

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return self.base ** linear_value


class MIEstimator(nn.Module):
    def __init__(self, in_size):
        super(MIEstimator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size * 2, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )

    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1))
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


class Processor(nn.Module):
    def __init__(self):
        super(Processor, self).__init__()

        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        self.fc = nn.Linear(128, 48)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv_net(x)
        x = x.view(x.size(0), -1)

        x = x.unsqueeze(1)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.squeeze(1)

        params = self.fc(x)
        return params


class Encoder(nn.Module):
    def __init__(self, z_dim, in_channels,hidden_size):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 48),
            nn.BatchNorm1d(48),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        params = self.net(x)

        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:]
        sigma = softplus(sigma) + 1e-7

        return params, Independent(Normal(loc=mu, scale=sigma), 1)


class Decoder(nn.Module):
    def __init__(self, encoder_dim,hidden_size):
        super(Decoder, self).__init__()
        self._decoder = nn.Sequential(
            nn.Linear(48, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU()
        )

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat
