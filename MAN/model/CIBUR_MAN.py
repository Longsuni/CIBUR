import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import math
from utils.tasks import lu_classify, predict_popus


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
        ).to("cuda")

        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8).to("cuda")

        self.fc = nn.Linear(128, 48).to("cuda")

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
    def __init__(self, z_dim, in_channels):
        super(Encoder, self).__init__()

        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 48),
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
    def __init__(self, encoder_dim):
        super(Decoder, self).__init__()
        self._decoder = nn.Sequential(
            nn.Linear(48, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.ReLU()
        )

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat


class CIBUR():

    def __init__(self):

        beta_start_value = 1e-3
        beta_end_value = 1
        beta_n_iterations = 100000
        beta_start_iteration = 50000
        self.beta_scheduler = ExponentialScheduler(start_value=beta_start_value, end_value=beta_end_value,
                                                   n_iterations=beta_n_iterations, start_iteration=beta_start_iteration)
        self.iterations = 0

        self.encoder_z1 = Encoder(24, 244)
        self.encoder_z2 = Encoder(24, 270)
        self.encoder_z3 = Encoder(24, 270)
        self.autoencoder_a = Decoder(244)
        self.autoencoder_b = Decoder(270)
        self.autoencoder_c = Decoder(270)
        self.mi_estimator_soft1 = MIEstimator(24)
        self.pr1 = Processor()
        self.pr2 = Processor()


        self.cir_num = 2
        self.mask_ratio = 0.4
        self.random_size = 0

        self.mask_token_244 = nn.Parameter(torch.randn(244))
        self.mask_token_270 = nn.Parameter(torch.randn(270))

    def mask_input(self, x):
        B, D = x.shape
        num_mask = int(D * self.mask_ratio)

        mask = torch.ones(B, D, device=x.device)
        mask_indices = torch.randperm(D)[:num_mask]
        mask[:, mask_indices] = 0
        x_masked = x * mask
        if D == 244:
            mask_token = self.mask_token_244
        elif D == 270:
            mask_token = self.mask_token_270
        else:
            raise ValueError(f"Unexpected input dimension: {D}")

        if self.random_size > 0:
            for idx in mask_indices:
                rand = torch.rand(1).item()
                if rand < self.random_size:
                    mask[:, idx] = 2
                else:
                    mask[:, idx] = 1
            mask_tokens = mask_token.unsqueeze(0).repeat(B, 1).to("cuda")
            x_masked = x_masked + mask_tokens * (mask == 1)

            randoms = torch.randn_like(x)
            x_masked = x_masked + randoms * (mask == 2)
        else:
            mask_tokens = mask_token.unsqueeze(0).repeat(B, 1).to("cuda")
            x_masked = x_masked + mask_tokens * (mask == 0)

        return x_masked, mask

    def mask_and_encode(self, x, encoder):
        masked_versions = []
        encoded_versions = []
        distributions = []
        for _ in range(self.cir_num):
            x_masked, _ = self.mask_input(x)
            masked_versions.append(x_masked)
            z, dist = encoder(x_masked)
            encoded_versions.append(z)
            distributions.append(dist)

        combined_encoded = torch.mean(torch.stack(encoded_versions), dim=0)
        combined_dist = Independent(Normal(
            loc=torch.mean(torch.stack([d.base_dist.loc for d in distributions]), dim=0),
            scale=torch.mean(torch.stack([d.base_dist.scale for d in distributions]), dim=0)
        ), 1)
        return combined_encoded, combined_dist

    def _compute_loss(self, v_yuan, v_hou):
        p_z1_given_v1 = v_yuan
        p_z2_given_v2 = v_hou
        z1 = v_yuan.rsample()
        z2 = v_hou.rsample()
        mi_gradient, mi_estimation = self.mi_estimator_soft1(z1, z2)
        mi_gradient = mi_gradient.mean()
        kl_1_2 = p_z1_given_v1.log_prob(z1) - p_z2_given_v2.log_prob(z1)
        kl_2_1 = p_z2_given_v2.log_prob(z2) - p_z1_given_v1.log_prob(z2)
        skl = (kl_1_2 + kl_2_1).mean() / 2
        beta = self.beta_scheduler(self.iterations)
        loss = -mi_gradient + beta * skl
        return loss

    def to_device(self, device):

        self.encoder_z1.to(device)
        self.encoder_z2.to(device)
        self.encoder_z3.to(device)
        self.autoencoder_a.to(device)
        self.autoencoder_b.to(device)
        self.autoencoder_c.to(device)
        self.mi_estimator_soft1.to(device)
        self.pr1 = Processor().to(device)
        self.pr2 = Processor().to(device)

    def train(self, config, xs, optimizer, scheduler):
        for epoch in range(config['training']['epoch']):

            z_1, id1 = self.mask_and_encode(xs[0], self.encoder_z1)
            z_2_1, id2_1 = self.mask_and_encode(xs[1], self.encoder_z2)
            z_2_2, id2_2 = self.mask_and_encode(xs[2], self.encoder_z3)
            z_2 = (z_2_1 + z_2_2) / 2

            de_z1 = self.autoencoder_a.decoder(z_1)
            de_z2_1 = self.autoencoder_b.decoder(z_2_1)
            de_z2_2 = self.autoencoder_c.decoder(z_2_2)

            reconstruction_loss1 = F.mse_loss(xs[0], de_z1)
            reconstruction_loss2 = F.mse_loss(xs[1], de_z2_1)
            reconstruction_loss3 = F.mse_loss(xs[2], de_z2_2)

            reconstruction_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss3

            mib_loss = self._compute_loss(id1, id2_1)
            mib_loss2 = self._compute_loss(id1, id2_2)
            cvc_loss = mib_loss + mib_loss2

            p2mo = self.pr1(z_1)
            mo2p = self.pr2(z_2)

            pre1 = F.mse_loss(p2mo, z_2)
            pre2 = F.mse_loss(mo2p, z_1)
            cri_loss = 0.0001 * pre1 + pre2

            loss = cvc_loss * 0.06 + reconstruction_loss * 0.1 + cri_loss * 0.3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)

            if (epoch + 1) % config['print_num'] == 0:
                print("Epoch : {:.0f}/{:.0f} ===>Loss = {:.4f}".format((epoch + 1), config['training']['epoch'], loss))
                self.test(xs[0], xs[1], xs[2], epoch)

            self.iterations += 1

    def test(self, attribute_m, source_matrix, destina_matrix, epoch):
        with (((torch.no_grad()))):
            self.encoder_z1.eval(), self.encoder_z2.eval(), self.encoder_z3.eval()
            self.pr1.eval(), self.pr2.eval()

            latent_a, _ = self.encoder_z1(attribute_m)
            latent_s, _ = self.encoder_z2(source_matrix)
            latent_d, _ = self.encoder_z3(destina_matrix)

            latent_m = (latent_s + latent_d) / 2

            latent_fusion = torch.cat([latent_a, latent_m], dim=1).cpu().numpy()

            lu_scores = lu_classify(latent_fusion)
            popus_scores = predict_popus(latent_fusion)

            self.encoder_z1.train()
            self.encoder_z2.train()
            self.encoder_z3.train()
            self.autoencoder_a.train()
            self.autoencoder_b.train()
            self.autoencoder_c.train()
            self.mi_estimator_soft1.train()
            self.pr1.train()
            self.pr2.train()

        return lu_scores, popus_scores, latent_fusion
