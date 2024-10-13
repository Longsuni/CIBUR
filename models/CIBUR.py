import numpy as np
import torch.nn.functional as F
from data_utils.utils import test_model
from models.modules import *


class CIBUR():
    def __init__(self, args):
        self.beta_scheduler = ExponentialScheduler(start_value=args.beta_start_value, end_value=args.beta_end_value,
                                                   n_iterations=args.beta_n_iterations,
                                                   start_iteration=args.beta_start_iteration)
        self.hidden_size = args.hidden_dim
        self.POI_dim = args.POI_dim
        self.landUse_dim = args.landUse_dim
        self.region_num = args.region_num
        self.z_dim = args.z_dim
        self.encoder_z1 = Encoder(self.z_dim, self.POI_dim, self.hidden_size)
        self.encoder_z2 = Encoder(self.z_dim, self.landUse_dim, self.hidden_size)
        self.encoder_z3 = Encoder(self.z_dim, self.region_num, self.hidden_size)
        self.autoencoder_a = Decoder(self.POI_dim, self.hidden_size)
        self.autoencoder_b = Decoder(self.landUse_dim, self.hidden_size)
        self.autoencoder_c = Decoder(self.region_num, self.hidden_size)
        self.mi_estimator_soft1 = MIEstimator(self.z_dim)
        self.pr1 = Processor()
        self.pr2 = Processor()
        self.pr3 = Processor()
        self.mask_token_POI = nn.Parameter(torch.randn(self.POI_dim))
        self.mask_token_LandUse = nn.Parameter(torch.randn(self.landUse_dim))
        self.mask_token_region = nn.Parameter(torch.randn(self.region_num))
        self.mask_ratio = args.mask_ratio
        self.weight_mask = 1 - args.weight_mask
        self.cir_num = args.cir_num
        self.best_r2 = 0
        self.iterations = 0
        self.best_emb = None

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
        self.pr3 = Processor().to(device)

    def mask_input(self, x):
        B, D = x.shape
        device = x.device

        num_tokens_to_mask = max(1, int(self.mask_ratio * D))

        mask = torch.ones(B, D, device=device)
        mask_indices = torch.randperm(D)[:num_tokens_to_mask]

        for idx in mask_indices:
            rand = torch.rand(1).item()
            if rand < self.weight_mask:
                mask[:, idx] = 0
            else:
                mask[:, idx] = 2

        x_masked = x.clone()

        if D == self.POI_dim:
            mask_token = self.mask_token_POI
        elif D == self.landUse_dim:
            mask_token = self.mask_token_LandUse
        elif D == self.region_num:
            mask_token = self.mask_token_region
        else:
            raise ValueError(f"error dim: {D}")

        mask_tokens = mask_token.unsqueeze(0).repeat(B, 1).to(device)
        x_masked[mask == 0] = mask_tokens[mask == 0]

        random_tokens = torch.randn_like(x_masked)
        x_masked[mask == 2] = random_tokens[mask == 2]

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

    def train_model(self, data, optimizer, task, city, sc, epochs):
        for epoch in range(epochs):
            poi_emb, landUse_emb, mob_emb = data
            poi_emb = poi_emb[0]
            landUse_emb = landUse_emb[0]
            mob_emb = mob_emb[0]

            z_1, id1 = self.mask_and_encode(poi_emb, self.encoder_z1)
            z_2, id2 = self.mask_and_encode(landUse_emb, self.encoder_z2)
            z_3, id3 = self.mask_and_encode(mob_emb, self.encoder_z3)

            de_z1 = self.autoencoder_a.decoder(z_1)
            de_z2 = self.autoencoder_b.decoder(z_2)
            de_z3 = self.autoencoder_c.decoder(z_3)

            reconstruction_loss1 = F.mse_loss(poi_emb, de_z1)
            reconstruction_loss2 = F.mse_loss(landUse_emb, de_z2)
            reconstruction_loss3 = F.mse_loss(mob_emb, de_z3)
            rve_loss = reconstruction_loss1 + reconstruction_loss2 + reconstruction_loss3

            mib_loss = self._compute_loss(id1, id2)
            mib_loss2 = self._compute_loss(id1, id3)
            mib_loss3 = self._compute_loss(id2, id3)
            cvc_loss = mib_loss + mib_loss2 + mib_loss3

            mo1 = self.pr1(z_1)
            mo2 = self.pr2(z_2)
            mo3 = self.pr3(z_3)

            pre1 = F.mse_loss(mo1, z_2)
            pre2 = F.mse_loss(mo1, z_3)
            pre3 = F.mse_loss(mo2, z_1)
            pre4 = F.mse_loss(mo2, z_3)
            pre5 = F.mse_loss(mo3, z_1)
            pre6 = F.mse_loss(mo3, z_2)
            cri_loss = pre1 + pre2 + pre3 + pre4 + pre5 + pre6

            loss = rve_loss + cvc_loss + cri_loss

            latent_fusion = torch.cat([z_1, z_2, z_3], dim=1).cpu().detach().numpy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sc.step(epoch)

            print("Epoch {}".format(epoch))
            self.test(latent_fusion, city, task)

            if epoch == epochs - 1:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>best result:")
                self.test(self.best_emb, city, task)
                np.save("best_emb_{}_{}".format(city, task), self.best_emb)

    def test(self, latent_fusion, city, task):
        with ((torch.no_grad())):
            self.encoder_z1.eval(), self.encoder_z2.eval(), self.encoder_z3.eval()
            self.pr1.eval(), self.pr2.eval(), self.pr3.eval()

            embs = latent_fusion

            _, _, r2 = test_model(city, task, embs)

            if self.best_r2 < r2:
                self.best_r2 = r2
                self.best_emb = embs

            self.encoder_z1.train()
            self.encoder_z2.train()
            self.encoder_z3.train()
            self.autoencoder_a.train()
            self.autoencoder_b.train()
            self.autoencoder_c.train()
            self.mi_estimator_soft1.train()
            self.pr1.train()
            self.pr2.train()
            self.pr3.train()
