"""TextMAE: an MAE (ViT) backbone around a learned image compression (LIC) coder. Needs timm==0.4.5."""
import warnings
from functools import partial

import torch
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import conv3x3, subpel_conv3x3
from compressai.models import CompressionModel
from compressai.ops import quantize_ste
from timm.models.vision_transformer import Block, PatchEmbed

from models.patch_selection import select_patches
from models.utils.pos_embed import get_2d_sincos_pos_embed

warnings.filterwarnings("ignore")


def stack(layers):
    """Sequential with a GELU between consecutive layers."""
    out = []
    for i, layer in enumerate(layers):
        out += [layer] + ([nn.GELU()] if i < len(layers) - 1 else [])
    return nn.Sequential(*out)


def interp(a, b, quarters):
    """Width `quarters`/4 of the way from b to a."""
    return int(b + (a - b) * quarters / 4)


def conv1x1(widths, cls=nn.Conv2d):
    return stack([cls(i, o, kernel_size=1, stride=1, padding=0) for i, o in zip(widths, widths[1:])])


def conv3x3_stack(widths):
    return stack([nn.Conv2d(i, o, kernel_size=3, stride=1, padding=1) for i, o in zip(widths, widths[1:])])


def context_widths(lat, slices, i, extra=0):
    """Widths of the channel-wise context transform of slice i (extra=1 adds the current slice, for LRP)."""
    s, half = lat // slices, slices // 2
    return [int(lat + s * min(i + extra, half + extra)), int(s * (half + 1)), int(s * (half * 3 / 4 + 1)),
            int(s * (half * 2 / 4 + 1)), int(s * (half * 1 / 4 + 1)), int(s)]


class TextMAE(CompressionModel):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, enc_dim=768, enc_depth=12, enc_heads=12,
                 dec_dim=512, dec_depth=8, dec_heads=16, mlp_ratio=4.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 latent_depth=384, hyperprior_depth=192, num_slices=12, num_keep_patches=144,
                 patch_selection="stratified"):
        super().__init__()
        self.enc_dim, self.dec_dim = enc_dim, dec_dim
        self.num_slices, self.max_support_slices = num_slices, num_slices // 2
        self.num_keep_patches, self.patch_selection = num_keep_patches, patch_selection
        self.keep_grid = int(num_keep_patches ** 0.5)
        if self.keep_grid ** 2 != num_keep_patches:
            raise ValueError(f"num_keep_patches must be a perfect square, got {num_keep_patches}")

        # LIC: transforms, hyperprior and channel-wise entropy model
        lat, hyp, slices = latent_depth, hyperprior_depth, num_slices
        self.entropy_bottleneck = EntropyBottleneck(hyp)
        self.gaussian_conditional = GaussianConditional(None)
        self.g_a = conv1x1([enc_dim, interp(enc_dim, dec_dim, 3), interp(enc_dim, dec_dim, 2), dec_dim, lat])
        self.g_s = conv1x1([lat, dec_dim, interp(enc_dim, dec_dim, 2), interp(enc_dim, dec_dim, 3), enc_dim],
                           nn.ConvTranspose2d)
        q = [interp(lat, hyp, k) for k in (1, 2, 3)]  # widths a quarter, half and three quarters of the way
        d = [lat, lat, q[2], q[1], q[0], hyp]
        self.h_a = stack([conv3x3(d[i], d[i + 1], stride=s) for i, s in enumerate([1, 1, 2, 1, 2])])

        def h_s():
            return stack([conv3x3(hyp, q[0]), subpel_conv3x3(q[0], q[1], r=2), conv3x3(q[1], q[2]),
                          subpel_conv3x3(q[2], lat, r=2), conv3x3(lat, lat)])

        self.h_s_mean, self.h_s_scale = h_s(), h_s()
        self.cc_transform_mean = nn.ModuleList(conv3x3_stack(context_widths(lat, slices, i)) for i in range(slices))
        self.cc_transform_scale = nn.ModuleList(conv3x3_stack(context_widths(lat, slices, i)) for i in range(slices))
        self.lrp_transform = nn.ModuleList(conv3x3_stack(context_widths(lat, slices, i, 1)) for i in range(slices))

        # MAE (ViT) encoder and decoder
        self.encoder_embed = PatchEmbed(img_size, patch_size, in_chans, enc_dim)
        n = self.encoder_embed.num_patches
        if n < num_keep_patches:
            raise ValueError(f"num_keep_patches ({num_keep_patches}) exceeds the {n} patches of a {img_size}px image")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, n + 1, enc_dim), requires_grad=False)  # sin-cos
        self.encoder_blocks = nn.ModuleList(
            Block(enc_dim, enc_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(enc_depth))
        self.encoder_norm = norm_layer(enc_dim)
        self.decoder_embed = nn.Linear(enc_dim, dec_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n + 1, dec_dim), requires_grad=False)  # sin-cos
        self.decoder_blocks = nn.ModuleList(
            Block(dec_dim, dec_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(dec_depth))
        self.decoder_norm = norm_layer(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, patch_size ** 2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        grid = int(self.encoder_embed.num_patches ** 0.5)
        for embed in (self.encoder_pos_embed, self.decoder_pos_embed):
            pos = get_2d_sincos_pos_embed(embed.shape[-1], grid, cls_token=True)
            embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))
        w = self.encoder_embed.proj.weight.data  # patch embed like nn.Linear
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """(N, L, p*p*3) -> (N, 3, H, W)."""
        p = self.encoder_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        x = torch.einsum("nhwpqc->nchpwq", x.reshape(x.shape[0], h, w, p, p, 3))
        return x.reshape(x.shape[0], 3, h * p, w * p)

    # ---- MAE
    def forward_encoder(self, imgs, total_scores):
        """Kept tokens (N, num_keep_patches, enc_dim) and ids_restore (N, L) that undoes the selection."""
        x = self.encoder_embed(imgs) + self.encoder_pos_embed[:, 1:, :]
        ids_shuffle = select_patches(self.patch_selection, total_scores, self.num_keep_patches)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :self.num_keep_patches].to(x.device)
        x = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        cls = (self.cls_token + self.encoder_pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for blk in self.encoder_blocks:
            x = blk(x)
        return self.encoder_norm(x)[:, 1:, :], ids_restore

    def forward_decoder(self, x_keep, ids_restore):
        """Put mask tokens at the dropped positions and predict all patches (N, L, p*p*3)."""
        ids_restore = ids_restore.to(x_keep.device)
        x = self.decoder_embed(x_keep)
        x = torch.cat([x, self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) + self.decoder_pos_embed[:, 1:, :]
        for blk in self.decoder_blocks:
            x = blk(x)
        return self.decoder_pred(self.decoder_norm(x))

    # ---- LIC helpers
    def to_grid(self, tokens):
        """(N, K, D) -> (N, D, g, g) with g = sqrt(K)."""
        return tokens.view(-1, self.keep_grid, self.keep_grid, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()

    def to_tokens(self, grid):
        return grid.permute(0, 2, 3, 1).contiguous().view(-1, self.num_keep_patches, grid.shape[1])

    def slice_stats(self, i, means, scales, decoded, y_shape):
        """Mean and scale of slice i from the hyperprior and the first max_support_slices decoded slices."""
        support = decoded if self.max_support_slices < 0 else decoded[:self.max_support_slices]
        mean_support = torch.cat([means] + support, dim=1)
        mu = self.cc_transform_mean[i](mean_support)[:, :, :y_shape[0], :y_shape[1]]
        sigma = self.cc_transform_scale[i](torch.cat([scales] + support, dim=1))[:, :, :y_shape[0], :y_shape[1]]
        return mean_support, mu, sigma

    def refine(self, i, mean_support, y_hat_slice):
        """Latent residual prediction: add a bounded correction to the decoded slice."""
        lrp = 0.5 * torch.tanh(self.lrp_transform[i](torch.cat([mean_support, y_hat_slice], dim=1)))
        return y_hat_slice + lrp

    def ans_tables(self):
        gc = self.gaussian_conditional
        return gc.quantized_cdf.tolist(), gc.cdf_length.reshape(-1).int().tolist(), gc.offset.reshape(-1).int().tolist()

    # ---- training forward
    def forward(self, imgs, total_scores):
        """Returns {"x_hat", "likelihoods": {"y", "z"}}; distortion terms live in RateDistortionLoss."""
        x_keep, ids_restore = self.forward_encoder(imgs, total_scores)
        y = self.g_a(self.to_grid(x_keep)).float()
        z = self.h_a(y)
        _, z_likelihood = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset
        scales, means = self.h_s_scale(z_hat), self.h_s_mean(z_hat)

        decoded, likelihoods = [], []
        for i, y_slice in enumerate(y.chunk(self.num_slices, 1)):
            mean_support, mu, sigma = self.slice_stats(i, means, scales, decoded, y.shape[2:])
            likelihoods.append(self.gaussian_conditional(y_slice, sigma, mu)[1])
            decoded.append(self.refine(i, mean_support, quantize_ste(y_slice - mu) + mu))

        tokens = self.to_tokens(self.g_s(torch.cat(decoded, dim=1)))
        preds = self.forward_decoder(tokens, ids_restore).float()
        return {"x_hat": self.unpatchify(preds), "likelihoods": {"y": torch.cat(likelihoods, dim=1), "z": z_likelihood}}

    # ---- real bitstream
    def compress(self, imgs, total_scores):
        """Returns {"string": [y_strings, z_strings], "shape": hyper-latent size, "ids_restore"}."""
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn("Inference on GPU is not recommended for autoregressive models: the entropy coder "
                          "runs sequentially.")
        x_keep, ids_restore = self.forward_encoder(imgs, total_scores)
        y = self.g_a(self.to_grid(x_keep)).float()
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        scales, means = self.h_s_scale(z_hat), self.h_s_mean(z_hat)

        cdfs, cdf_lengths, offsets = self.ans_tables()
        symbols, indexes, decoded = [], [], []
        for i, y_slice in enumerate(y.chunk(self.num_slices, 1)):
            mean_support, mu, sigma = self.slice_stats(i, means, scales, decoded, y.shape[2:])
            indexes.extend(self.gaussian_conditional.build_indexes(sigma).reshape(-1).tolist())
            y_q = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            symbols.extend(y_q.reshape(-1).tolist())
            decoded.append(self.refine(i, mean_support, y_q + mu))

        encoder = BufferedRansEncoder()
        encoder.encode_with_indexes(symbols, indexes, cdfs, cdf_lengths, offsets)
        return {"string": [[encoder.flush()], z_strings], "shape": z.size()[-2:], "ids_restore": ids_restore}

    def decompress(self, strings, shape, ids_restore):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales, means = self.h_s_scale(z_hat), self.h_s_mean(z_hat)
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]  # h_s upsamples the hyper-latent by 4

        cdfs, cdf_lengths, offsets = self.ans_tables()
        decoder = RansDecoder()
        decoder.set_stream(strings[0][0])
        decoded = []
        for i in range(self.num_slices):
            mean_support, mu, sigma = self.slice_stats(i, means, scales, decoded, y_shape)
            index = self.gaussian_conditional.build_indexes(sigma)
            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdfs, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            decoded.append(self.refine(i, mean_support, self.gaussian_conditional.dequantize(rv, mu)))

        tokens = self.to_tokens(self.g_s(torch.cat(decoded, dim=1)))
        return {"x_hat": self.unpatchify(self.forward_decoder(tokens, ids_restore).float())}


def textmae_base_patch16(**kw):
    """ViT-B encoder (768-d, 12 layers), 512-d / 8-layer decoder."""
    return TextMAE(enc_dim=768, enc_depth=12, enc_heads=12, dec_dim=512, dec_depth=8, dec_heads=16, **kw)


def textmae_large_patch16(**kw):
    """ViT-L encoder (1024-d, 24 layers): fits the official MAE visualisation checkpoints."""
    return TextMAE(enc_dim=1024, enc_depth=24, enc_heads=16, dec_dim=512, dec_depth=8, dec_heads=16, **kw)


MODELS = {"textmae_base_patch16": textmae_base_patch16, "textmae_large_patch16": textmae_large_patch16}


def build_model(name, **kw):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}', choose from {sorted(MODELS)}")
    return MODELS[name](**kw)
