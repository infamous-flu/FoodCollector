import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, n_mixtures):
        super(MDN, self).__init__()
        self.n_mixtures = n_mixtures
        self.output_dim = output_dim
        self.mu = nn.Linear(input_dim, output_dim * n_mixtures)
        self.sigma = nn.Linear(input_dim, output_dim * n_mixtures)
        self.pi = nn.Linear(input_dim, n_mixtures)

    def forward(self, x):
        mus = self.mu(x)
        sigmas = torch.exp(self.sigma(x))
        logpi = F.log_softmax(self.pi(x), dim=-1)

        return mus, sigmas, logpi


class MDNRNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim, n_mixtures=5):
        super(MDNRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_mixtures = n_mixtures
        self.rnn = nn.LSTM(latent_dim + action_dim, hidden_dim, batch_first=True)
        self.mdn = MDN(hidden_dim, latent_dim, n_mixtures)

    def forward(self, latent, action, hx=None, cx=None):
        batch_size, sequence_length = latent.size(0), latent.size(1)
        x = torch.cat([latent, action], dim=-1)
        if hx is None or cx is None:
            output, (hx_next, cx_next) = self.rnn(x)
        else:
            output, (hx_next, cx_next) = self.rnn(x, (hx, cx))
        mus, sigmas, logpi = self.mdn(output)
        mus = mus.view(batch_size, sequence_length, self.n_mixtures, self.latent_dim)
        sigmas = sigmas.view(batch_size, sequence_length, self.n_mixtures, self.latent_dim)
        logpi = logpi.view(batch_size, sequence_length, self.n_mixtures)

        return mus, sigmas, logpi, hx_next, cx_next

    def initial_state(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim), torch.zeros(1, batch_size, self.hidden_dim)


def mdn_loss(mus, sigmas, logpi, targets):
    targets = targets.unsqueeze(-2)
    normal_dist = Normal(mus, sigmas)
    g_log_probs = normal_dist.log_prob(targets)
    g_log_probs = logpi + torch.sum(g_log_probs, dim=-1)
    max_log_probs = torch.max(g_log_probs, dim=-1, keepdim=True)[0]
    g_log_probs = g_log_probs - max_log_probs

    g_probs = torch.exp(g_log_probs)
    probs = torch.sum(g_probs, dim=-1)
    log_prob = max_log_probs.squeeze() + torch.log(probs)

    return -torch.mean(log_prob)
