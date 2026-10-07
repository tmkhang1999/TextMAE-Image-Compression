"""
TextMAE: score-guided masked-autoencoder image compression.

Pipeline (see README for a diagram):

1. Patch selection   - keep `num_keep_patches` of the image patches, guided by a per-patch
                       importance score (texture x structure, see `textmae.data`).
2. MAE encoder       - ViT encoder on the kept patches only.
3. Latent coder      - g_a -> hyperprior (h_a / h_s) -> channel-wise autoregressive entropy
                       model with latent residual prediction (LRP) -> g_s.
4. MAE decoder       - put mask tokens back at the dropped positions and reconstruct.

The patch indices (`ids_restore`) are side information; they are Huffman coded separately
in `textmae.coding`.

Needs timm==0.4.5 (PatchEmbed / Block API) and compressai.
"""
import warnings
from functools import partial

import torch
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import conv3x3, subpel_conv3x3
from compressai.models import CompressionModel
from compressai.ops import quantize_ste
from pytorch_msssim import SSIM
from timm.models.vision_transformer import Block, PatchEmbed

from textmae.losses.perceptual import feature_loss
from textmae.models import dims
from textmae.models.patch_selection import select_patches
from textmae.models.pos_embed import get_2d_sincos_pos_embed

warnings.filterwarnings("ignore")


def _with_gelu(layers):
    """Interleave GELU activations between layers (none after the last one)."""
    out = []
    for i, layer in enumerate(layers):
        out.append(layer)
        if i < len(layers) - 1:
            out.append(nn.GELU())
    return nn.Sequential(*out)


def _pointwise_stack(widths, layer_cls):
    """1x1 (transposed) convolutions over the channel widths, GELU in between."""
    return _with_gelu(
        [layer_cls(widths[i], widths[i + 1], kernel_size=1, stride=1, padding=0)
         for i in range(len(widths) - 1)]
    )


def _channel_context_transform(widths):
    """3x3 convolution stack used for the mean / scale / LRP predictions of one slice."""
    return _with_gelu(
        [nn.Conv2d(widths[i], widths[i + 1], kernel_size=3, stride=1, padding=1)
         for i in range(len(widths) - 1)]
    )


class TextMAE(CompressionModel):
    """MAE (ViT) backbone with a learned latent coder on the kept patch tokens."""

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            encoder_embed_dim=768,
            encoder_depth=12,
            encoder_num_heads=12,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            latent_depth=384,
            hyperprior_depth=192,
            num_slices=12,
            num_keep_patches=144,
            patch_selection="stratified",
    ):
        super().__init__()

        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.latent_depth = latent_depth
        self.hyperprior_depth = hyperprior_depth
        self.num_slices = num_slices
        self.num_keep_patches = num_keep_patches
        self.patch_selection = patch_selection

        # The kept tokens are laid out on a square grid for the convolutional latent coder.
        self.keep_grid = int(num_keep_patches ** 0.5)
        if self.keep_grid ** 2 != num_keep_patches:
            raise ValueError(f"num_keep_patches must be a perfect square, got {num_keep_patches}")

        self._build_latent_coder()
        self._build_mae(img_size, patch_size, in_chans, mlp_ratio, norm_layer)
        self.initialize_weights()

    # ------------------------------------------------------------------ construction

    def _build_latent_coder(self):
        """Entropy models plus the analysis / synthesis / hyper / context transforms."""
        enc, dec = self.encoder_embed_dim, self.decoder_embed_dim
        lat, hyp, slices = self.latent_depth, self.hyperprior_depth, self.num_slices

        self.entropy_bottleneck = EntropyBottleneck(hyp)
        self.gaussian_conditional = GaussianConditional(None)
        self.max_support_slices = slices // 2

        # Token <-> latent transforms (1x1 convolutions)
        self.g_a = _pointwise_stack(dims.analysis_dims(enc, dec, lat), nn.Conv2d)
        self.g_s = _pointwise_stack(dims.synthesis_dims(enc, dec, lat), nn.ConvTranspose2d)

        # Hyperprior
        d = dims.hyper_analysis_dims(lat, hyp)
        strides = [1, 1, 2, 1, 2]
        self.h_a = _with_gelu([conv3x3(d[i], d[i + 1], stride=strides[i]) for i in range(5)])

        d = dims.hyper_synthesis_dims(lat, hyp)

        def hyper_synthesis():
            return _with_gelu([
                conv3x3(d[0], d[1], stride=1),
                subpel_conv3x3(d[1], d[2], r=2),
                conv3x3(d[2], d[3], stride=1),
                subpel_conv3x3(d[3], d[4], r=2),
                conv3x3(d[4], d[5], stride=1),
            ])

        self.h_s_mean = hyper_synthesis()
        self.h_s_scale = hyper_synthesis()

        # Channel-wise context: slice i is predicted from the hyperprior and earlier slices
        self.cc_transform_mean = nn.ModuleList(
            _channel_context_transform(dims.channel_context_dims(lat, slices, i))
            for i in range(slices))
        self.cc_transform_scale = nn.ModuleList(
            _channel_context_transform(dims.channel_context_dims(lat, slices, i))
            for i in range(slices))
        self.lrp_transform = nn.ModuleList(
            _channel_context_transform(dims.channel_context_dims(lat, slices, i, in_extra=1))
            for i in range(slices))

    def _build_mae(self, img_size, patch_size, in_chans, mlp_ratio, norm_layer):
        """ViT encoder and decoder, following facebookresearch/mae."""
        # Encoder
        self.encoder_embed = PatchEmbed(img_size, patch_size, in_chans, self.encoder_embed_dim)
        num_patches = self.encoder_embed.num_patches
        if num_patches < self.num_keep_patches:
            raise ValueError(
                f"num_keep_patches ({self.num_keep_patches}) exceeds the {num_patches} patches "
                f"of a {img_size}px image"
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        # Fixed sin-cos embedding
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.encoder_embed_dim), requires_grad=False)
        self.encoder_blocks = nn.ModuleList([
            Block(dim=self.encoder_embed_dim, num_heads=self.encoder_num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(self.encoder_depth)
        ])
        self.encoder_norm = norm_layer(self.encoder_embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        # Fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([
            Block(dim=self.decoder_embed_dim, num_heads=self.decoder_num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(self.decoder_depth)
        ])
        self.decoder_norm = norm_layer(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        net = cls(**kwargs)
        net.load_state_dict(state_dict)
        return net

    def initialize_weights(self):
        # Fixed sin-cos position embeddings
        grid = int(self.encoder_embed.num_patches ** 0.5)
        for embed in (self.encoder_pos_embed, self.decoder_pos_embed):
            pos = get_2d_sincos_pos_embed(embed.shape[-1], grid, cls_token=True)
            embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.encoder_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # xavier_uniform, following the official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ------------------------------------------------------------------ patches

    def patchify(self, imgs):
        """(N, 3, H, W) -> (N, L, patch_size**2 * 3) with L = (H // patch_size) ** 2."""
        patch_size = self.encoder_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % patch_size == 0

        h = w = imgs.shape[2] // patch_size
        x = imgs.reshape(imgs.shape[0], 3, h, patch_size, w, patch_size)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, patch_size ** 2 * 3)

    def unpatchify(self, patched_feature):
        """(N, L, patch_size**2 * 3) -> (N, 3, H, W); inverse of `patchify`."""
        patch_size = self.encoder_embed.patch_size[0]

        h = w = int(patched_feature.shape[1] ** 0.5)
        assert h * w == patched_feature.shape[1]

        x = patched_feature.reshape(patched_feature.shape[0], h, w, patch_size, patch_size, 3)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], 3, h * patch_size, w * patch_size)

    def random_masking(self, x, total_scores):
        """
        Keep `num_keep_patches` patches per image, chosen by `self.patch_selection`.

        Args:
            x (torch.Tensor): Embedded patches (N, L, D).
            total_scores (torch.Tensor): Per-patch importance scores (N, L).

        Returns:
            x_keep (torch.Tensor): Kept patches (N, num_keep_patches, D).
            ids_restore (torch.Tensor): Indices that undo the shuffle (N, L).
        """
        ids_shuffle = select_patches(self.patch_selection, total_scores, self.num_keep_patches)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :self.num_keep_patches].to(x.device)
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        return x_keep, ids_restore

    # ------------------------------------------------------------------ MAE encoder / decoder

    def forward_encoder(self, imgs, total_scores):
        """
        Returns:
            x_keep (torch.Tensor): Encoded kept tokens (N, num_keep_patches, encoder_embed_dim).
            ids_restore (torch.Tensor): Indices restoring the original patch order (N, L).
        """
        x = self.encoder_embed(imgs)
        x = x + self.encoder_pos_embed[:, 1:, :]  # position embedding without cls token

        x_keep, ids_restore = self.random_masking(x, total_scores)

        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        x_keep = torch.cat((cls_token.expand(x_keep.shape[0], -1, -1), x_keep), dim=1)

        for blk in self.encoder_blocks:
            x_keep = blk(x_keep)
        x_keep = self.encoder_norm(x_keep)
        return x_keep[:, 1:, :], ids_restore

    def forward_decoder(self, x_keep, ids_restore):
        """
        Args:
            x_keep (torch.Tensor): Kept tokens (N, num_keep_patches, encoder_embed_dim).
            ids_restore (torch.Tensor): Indices restoring the original patch order (N, L).

        Returns:
            Patch predictions (N, L, patch_size**2 * 3).
        """
        ids_restore = ids_restore.to(x_keep.device)
        x = self.decoder_embed(x_keep)

        # Append mask tokens and unshuffle to the original patch order
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x = x + self.decoder_pos_embed[:, 1:, :]
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)

    def forward_loss(self, imgs, preds):
        """
        Args:
            imgs (torch.Tensor): Target images (N, 3, H, W).
            preds (torch.Tensor): Patch predictions (N, L, patch_size**2 * 3).

        Returns:
            (ssim_loss, l1_loss, feature_loss)
        """
        preds = self.unpatchify(preds)
        ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)
        ssim_loss = 1 - ssim(preds, imgs)
        l1_loss = nn.L1Loss()(preds, imgs)
        return ssim_loss, l1_loss, feature_loss(preds, imgs)

    # ------------------------------------------------------------------ latent coder helpers

    def _tokens_to_grid(self, tokens):
        """(N, num_keep_patches, D) -> (N, D, g, g) with g = sqrt(num_keep_patches)."""
        return (tokens.view(-1, self.keep_grid, self.keep_grid, tokens.shape[-1])
                .permute(0, 3, 1, 2).contiguous())

    def _grid_to_tokens(self, grid):
        """(N, D, g, g) -> (N, num_keep_patches, D); inverse of `_tokens_to_grid`."""
        return grid.permute(0, 2, 3, 1).contiguous().view(-1, self.num_keep_patches, grid.shape[1])

    def _slice_statistics(self, slice_index, latent_means, latent_scales, y_hat_slices, y_shape):
        """Predict mean and scale of slice `slice_index` from the hyperprior and earlier slices."""
        support = (y_hat_slices if self.max_support_slices < 0
                   else y_hat_slices[: self.max_support_slices])

        mean_support = torch.cat([latent_means] + support, dim=1)
        mu = self.cc_transform_mean[slice_index](mean_support)
        mu = mu[:, :, : y_shape[0], : y_shape[1]]

        scale_support = torch.cat([latent_scales] + support, dim=1)
        sigma = self.cc_transform_scale[slice_index](scale_support)
        sigma = sigma[:, :, : y_shape[0], : y_shape[1]]
        return mean_support, mu, sigma

    def _refine_slice(self, slice_index, mean_support, y_hat_slice):
        """Latent residual prediction: add a bounded correction to the decoded slice."""
        lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
        lrp = 0.5 * torch.tanh(self.lrp_transform[slice_index](lrp_support))
        return y_hat_slice + lrp

    def _ans_tables(self):
        """CDF tables of the Gaussian conditional, as plain lists for the rANS coder."""
        gc = self.gaussian_conditional
        return (gc.quantized_cdf.tolist(),
                gc.cdf_length.reshape(-1).int().tolist(),
                gc.offset.reshape(-1).int().tolist())

    # ------------------------------------------------------------------ training forward

    def forward(self, imgs, total_scores):
        """
        Args:
            imgs (torch.Tensor): Batch of images (N, 3, H, W).
            total_scores (torch.Tensor): Per-patch importance scores (N, L).

        Returns:
            dict with "loss" (ssim, l1, feature), "likelihoods" ({"y", "z"}) and "x_hat".
        """
        x_keep, ids_restore = self.forward_encoder(imgs, total_scores)

        y = self.g_a(self._tokens_to_grid(x_keep)).float()
        y_shape = y.shape[2:]

        # Hyperprior
        z = self.h_a(y)
        _, z_likelihood = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        # Slice-wise entropy model
        y_hat_slices = []
        y_likelihoods = []
        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, 1)):
            mean_support, mu, sigma = self._slice_statistics(
                slice_index, latent_means, latent_scales, y_hat_slices, y_shape)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, sigma, mu)
            y_likelihoods.append(y_slice_likelihood)

            y_hat_slice = quantize_ste(y_slice - mu) + mu
            y_hat_slices.append(self._refine_slice(slice_index, mean_support, y_hat_slice))

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihood = torch.cat(y_likelihoods, dim=1)

        tokens = self._grid_to_tokens(self.g_s(y_hat))
        preds = self.forward_decoder(tokens, ids_restore).float()

        return {
            "loss": self.forward_loss(imgs, preds),
            "likelihoods": {"y": y_likelihood, "z": z_likelihood},
            "x_hat": self.unpatchify(preds),
        }

    # ------------------------------------------------------------------ real bitstream

    def compress(self, imgs, total_scores):
        """
        Encode a batch to bitstreams.

        Returns:
            dict with "string" ([y_strings, z_strings]), "shape" (hyper-latent size) and
            "ids_restore" (patch order side information, coded separately by the caller).
        """
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for autoregressive models. The entropy "
                "coder is run sequentially on GPU "
            )

        x_keep, ids_restore = self.forward_encoder(imgs, total_scores)

        y = self.g_a(self._tokens_to_grid(x_keep)).float()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        cdfs, cdf_lengths, offsets = self._ans_tables()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        y_hat_slices = []
        for slice_index, y_slice in enumerate(y.chunk(self.num_slices, 1)):
            mean_support, mu, sigma = self._slice_statistics(
                slice_index, latent_means, latent_scales, y_hat_slices, y_shape)

            index = self.gaussian_conditional.build_indexes(sigma)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            y_hat_slices.append(self._refine_slice(slice_index, mean_support, y_hat_slice))

        encoder.encode_with_indexes(symbols_list, indexes_list, cdfs, cdf_lengths, offsets)
        y_strings = [encoder.flush()]

        return {
            "string": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "ids_restore": ids_restore,
        }

    def decompress(self, strings, shape, ids_restore):
        """
        Decode bitstreams from `compress` back to an image.

        Args:
            strings: [y_strings, z_strings] as returned by `compress`.
            shape: Hyper-latent spatial size as returned by `compress`.
            ids_restore: Patch order side information (N, L).
        """
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        # The hyper-synthesis upsamples the hyper-latent by 4 in each direction
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        cdfs, cdf_lengths, offsets = self._ans_tables()
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])

        y_hat_slices = []
        for slice_index in range(self.num_slices):
            mean_support, mu, sigma = self._slice_statistics(
                slice_index, latent_means, latent_scales, y_hat_slices, y_shape)

            index = self.gaussian_conditional.build_indexes(sigma)
            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdfs, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            y_hat_slices.append(self._refine_slice(slice_index, mean_support, y_hat_slice))

        y_hat = torch.cat(y_hat_slices, dim=1)
        tokens = self._grid_to_tokens(self.g_s(y_hat))
        x_hat = self.forward_decoder(tokens, ids_restore).float()
        return {"x_hat": self.unpatchify(x_hat)}


def textmae_base_patch16(**kwargs):
    """ViT-Base encoder (768-d, 12 layers), 512-d / 8-layer decoder."""
    return TextMAE(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        **kwargs,
    )


def textmae_large_patch16(**kwargs):
    """ViT-Large encoder (1024-d, 24 layers); matches the official MAE visualisation checkpoints."""
    return TextMAE(
        encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        **kwargs,
    )


MODELS = {
    "textmae_base_patch16": textmae_base_patch16,
    "textmae_large_patch16": textmae_large_patch16,
}


def build_model(name, **kwargs):
    """Create a model from the MODELS registry, e.g. build_model("textmae_base_patch16")."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)
