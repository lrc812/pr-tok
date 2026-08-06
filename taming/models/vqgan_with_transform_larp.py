import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.modules.transformer.vec_transformer import GPTC
class VQModel_larp(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 transformerconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 prior_loss_weight=0.06,
                 prior_n_rounds=2,
                 prior_avg_loss_over_rounds=True,
                 prior_no_grad_before_last_round=False,
                 use_mix_ss=True,
                 mix_ss_max_ratio=0.5,
                 mix_ss_peak_steps_ratio=0.3,
                 prior_latent_ce_temperature=1.0,
                 transformer_lr_mult=1.0,
                 latent_l2_normalized=True,
                 latent_l2_normalize_eps=1e-6,
                 vq_legacy=False,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.transformer = GPTC(**transformerconfig)
        self.prior_n_rounds = int(prior_n_rounds)
        self.prior_no_grad_before_last_round = bool(prior_no_grad_before_last_round)
        
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape,
                                        legacy=vq_legacy,
                                        l2_normalized=latent_l2_normalized,
                                        l2_normalize_eps=latent_l2_normalize_eps)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.prior_latent_ce_temperature = float(prior_latent_ce_temperature)
        self.prior_avg_loss_over_rounds = bool(prior_avg_loss_over_rounds)
        self.codebook_size = n_embed
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.use_mix_ss = bool(use_mix_ss)
        self.mix_ss_max_ratio = float(mix_ss_max_ratio)
        self.mix_ss_peak_steps_ratio = float(mix_ss_peak_steps_ratio)
        self.prior_loss_weight = float(prior_loss_weight)
        self.transformer_lr_mult = float(transformer_lr_mult)
        self.latent_l2_normalized = bool(latent_l2_normalized)
        self.latent_l2_normalize_eps = float(latent_l2_normalize_eps)
        if self.prior_n_rounds < 1:
            raise ValueError("prior_n_rounds must be at least 1")
        if self.prior_latent_ce_temperature <= 0:
            raise ValueError("prior_latent_ce_temperature must be positive")
        if self.latent_l2_normalize_eps <= 0:
            raise ValueError("latent_l2_normalize_eps must be positive")
        if not 0.0 <= self.mix_ss_max_ratio <= 1.0:
            raise ValueError("mix_ss_max_ratio must be in [0, 1]")
        if self.mix_ss_peak_steps_ratio <= 0:
            raise ValueError("mix_ss_peak_steps_ratio must be positive")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()
    
    def get_emb(self):
        return self.quantize.get_codebook_embeddings().detach()

    @staticmethod
    def _metrics_fp32(metrics):
        """Keep old Lightning's DDP metric collection NumPy-compatible."""
        return {
            key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
            for key, value in metrics.items()
        }

    @staticmethod
    @torch.no_grad()
    def _prior_accuracy_metrics(logits, labels):
        """Return teacher-forced next-token top-1/top-5 accuracies."""
        logits = logits.detach().float()
        labels = labels.detach()
        max_k = min(5, logits.shape[-1])
        predictions = logits.topk(max_k, dim=-1).indices
        matches = predictions.eq(labels.unsqueeze(-1))
        return {
            "prior_top1": matches[..., :1].any(dim=-1).float().mean(),
            "prior_top5": matches.any(dim=-1).float().mean(),
        }

    @staticmethod
    @torch.no_grad()
    def _codebook_usage_metrics(indices, codebook_size):
        """Measure per-batch code usage and token-distribution perplexity."""
        counts = torch.bincount(
            indices.detach().reshape(-1), minlength=codebook_size
        ).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        nonzero = probabilities > 0
        entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
        return {
            "code_usage": nonzero.float().sum() / float(codebook_size),
            "code_perplexity": entropy.exp(),
        }

    def scheduled_sampling_ratio(self, **kwargs):
        if not self.use_mix_ss:
            return 0.0
        trainer = getattr(self, "_trainer", None)
        global_step = kwargs.get("global_step", getattr(trainer, "global_step", 0))
        global_step = float(global_step)
        max_steps = kwargs.get("max_steps", getattr(trainer, "estimated_stepping_batches", None))
        if max_steps is None or float(max_steps) <= 0:
            max_steps = max(global_step, 1.0)
        peak_steps = max(float(max_steps) * self.mix_ss_peak_steps_ratio, 1.0)
        return min(global_step / peak_steps * self.mix_ss_max_ratio, self.mix_ss_max_ratio)

    def logits_to_token_embedding_with_ss(self, logits, ar_input_staring_from_idx_1, mask=None, **kwargs):
        # logits: (b, n - 1, codebook_size), sequence index from 1 to n-1 (inclusive)
        # ar_input_staring_from_idx_1: (b, n - 1, dim=16), requires_grad=True
        if mask is None:
            b, n_minus_1, _ = logits.size()
            ss_ratio = self.scheduled_sampling_ratio(**kwargs)

            mask = torch.rand(b, n_minus_1, 1, device=self.device) < ss_ratio
            mask = mask.expand(-1, -1, ar_input_staring_from_idx_1.shape[-1])

        with torch.autocast(device_type='cuda', enabled=False):
            logits = logits.float()
            if not torch.isfinite(logits).all().item():
                nonfinite_count = (~torch.isfinite(logits)).sum().item()
                raise FloatingPointError(
                    "Scheduled sampling received non-finite logits "
                    f"({nonfinite_count}/{logits.numel()}); aborting before "
                    "torch.multinomial. Check latent normalization and learning rate."
                )
            probs = F.softmax(logits, dim=-1) # (b, n - 1, codebook_size)
            indices = torch.multinomial(probs.view(-1, self.codebook_size), 1).view(*probs.size()[:-1]) # (b, n - 1)
        token_embedding = F.embedding(indices, self.get_emb()) # (b, n - 1, d=16)
        token_embedding = torch.where(mask, token_embedding, ar_input_staring_from_idx_1)

        return token_embedding
    
    def calculate_logits_and_ar_pred_cont(self, prior_model_output):
        ar_pred_cont = prior_model_output # (b, n, dim=16)
        logits = F.linear(prior_model_output, self.get_emb())[:, 1:]
        logits = logits / self.prior_latent_ce_temperature
        logits = logits.contiguous() # (b, n - 1, codebook_size)
        return logits, ar_pred_cont
    
    def prior_ar_predict_n_rounds_ss(self, ar_input, **kwargs):
        prior_model = self.transformer
        n_rounds = self.prior_n_rounds
        no_grad_before_last_round = self.prior_no_grad_before_last_round

        b, n, _ = ar_input.size()
        n_minus_1 = n - 1
        ss_ratio = self.scheduled_sampling_ratio(**kwargs)

        mask_ss = torch.rand(b, n_minus_1, 1, device=self.device) < ss_ratio
        mask_ss = mask_ss.expand(-1, -1, ar_input.shape[-1])

        logits_all_rounds = [] #(n_rounds, b, n - 1, codebook_size)
        next_ar_input = ar_input # (b, n, d=16)
        for i in range(n_rounds):
            if no_grad_before_last_round and i < n_rounds - 1:
                with torch.no_grad():
                    prior_model_output = prior_model.ar_predict(next_ar_input.detach())
            else:
                prior_model_output = prior_model.ar_predict(next_ar_input)
            logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)
            logits_all_rounds.append(logits)

            if i < n_rounds - 1:
                token_embedding = self.logits_to_token_embedding_with_ss(logits, ar_input[:, 1:], mask=mask_ss, **kwargs) # (b, n - 1, d=16)
                next_ar_input = torch.cat([ar_input[:, :1], token_embedding], dim=1) # (b, n, d=16)

        if self.prior_avg_loss_over_rounds:
            logits_all_rounds = torch.stack(logits_all_rounds, dim=0) # (n_rounds, b, n - 1, codebook_size)

        else:
            logits_all_rounds = torch.stack([logits_all_rounds[-1]], dim=0) # (1, b, n - 1, codebook_size)

        return logits_all_rounds, ar_pred_cont, next_ar_input # here the next_ar_input is actually the last round's ar_input

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        quant, diff, info = self.encode(x)
        dec = self.decode(quant)
        xrec = dec
        qloss =diff
        labels = info[-1].reshape(x.shape[0], -1)  # (b, n=h*w)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            #quant shape (b, c, h, w) 
            labels = labels[:, 1:].contiguous()
            quant = quant.flatten(2).permute(0, 2, 1).contiguous()
            gptloss = 0
            logits_all_rounds, ar_pred_cont, regularized_z_ss = self.prior_ar_predict_n_rounds_ss(quant) # regularized_z_ss: (b, n, d=16)
            labels_all_rounds = labels.unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1).contiguous() # (n_rounds or 1, b, n - 1)
            gptloss = F.cross_entropy(logits_all_rounds.view(-1, self.codebook_size), labels_all_rounds.view(-1))
            prior_metrics = self._prior_accuracy_metrics(logits_all_rounds[0], labels)
            codebook_metrics = self._codebook_usage_metrics(info[-1], self.codebook_size)
            aeloss += self.prior_loss_weight * gptloss
            self.log("train/gpt_loss",gptloss.float(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/prior_ce", gptloss.float(), prog_bar=False, logger=True,
                     on_step=True, on_epoch=True, sync_dist=True)
            for name, value in prior_metrics.items():
                self.log(f"train/{name}", value.float(), prog_bar=False, logger=True,
                         on_step=True, on_epoch=True, sync_dist=True)
            for name, value in codebook_metrics.items():
                self.log(f"train/{name}", value.float(), prog_bar=False, logger=True,
                         on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/ss_ratio", self.scheduled_sampling_ratio(), prog_bar=False, logger=True, on_step=True)
            self.log("train/aeloss", aeloss.float(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(self._metrics_fp32(log_dict_ae), prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss.float(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(self._metrics_fp32(log_dict_disc), prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        quant, diff, info = self.encode(x)
        dec = self.decode(quant)
        xrec = dec
        qloss =diff
        labels = info[-1].reshape(x.shape[0], -1)  # (b, n=h*w)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        labels = labels[:, 1:].contiguous()
        quant = quant.flatten(2).permute(0, 2, 1).contiguous()
        gptloss = 0
        logits_all_rounds, ar_pred_cont, regularized_z_ss = self.prior_ar_predict_n_rounds_ss(quant) # regularized_z_ss: (b, n, d=16)
        labels_all_rounds = labels.unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1).contiguous() # (n_rounds or 1, b, n - 1)
        gptloss = F.cross_entropy(logits_all_rounds.view(-1, self.codebook_size), labels_all_rounds.view(-1))
        prior_metrics = self._prior_accuracy_metrics(logits_all_rounds[0], labels)
        codebook_metrics = self._codebook_usage_metrics(info[-1], self.codebook_size)
        aeloss += self.prior_loss_weight * gptloss
        self.log("val/gpt_loss",gptloss.float(), prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("val/prior_ce", gptloss.float(), prog_bar=False, logger=True,
                 on_step=False, on_epoch=True, sync_dist=True)
        for name, value in prior_metrics.items():
            self.log(f"val/{name}", value.float(), prog_bar=False, logger=True,
                     on_step=False, on_epoch=True, sync_dist=True)
        for name, value in codebook_metrics.items():
            self.log(f"val/{name}", value.float(), prog_bar=False, logger=True,
                     on_step=False, on_epoch=True, sync_dist=True)
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        
        self.log("val/aeloss", aeloss.float(),
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(
            self._metrics_fp32(log_dict_ae),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            self._metrics_fp32(log_dict_disc),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return {"val/aeloss": aeloss.detach().float(), "val/gpt_loss": gptloss.detach().float()}

    def configure_optimizers(self):
        lr = self.learning_rate
        tr_lr = lr * float(self.transformer_lr_mult)

        # One optimizer with an explicit multiplier for the lightweight AR prior.
        opt_ae = torch.optim.Adam(
            [
                {
                    "params": list(self.encoder.parameters())
                              + list(self.decoder.parameters())
                              + list(self.quantize.parameters())
                              + list(self.quant_conv.parameters())
                              + list(self.post_quant_conv.parameters()),
                    "lr": lr,
                },
                {
                    "params": list(self.transformer.parameters()),
                    "lr": tr_lr,
                },
            ],
            betas=(0.5, 0.9),
        )

        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.9),
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


