import warnings

warnings.filterwarnings("ignore")

from functools import partial
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
from pytorch_msssim import SSIM

# Timm.__version__ == 0.4.5
from timm.models.vision_transformer import PatchEmbed, Block

from compressai.models import CompressionModel
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.layers import conv3x3, subpel_conv3x3
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ops import quantize_ste

from models.Compression.loss.vgg import cal_features_loss
from models.Compression.common.pos_embed import get_2d_sincos_pos_embed


class MCM(CompressionModel):
    """
    Masked Autoencoder with Vision Transformer backbone

    This class inherits from MaskedAutoencoder in *facebookresearch/mae* class.
    See the original paper and the `MAE' documentation
    <https://github.com/facebookresearch/mae/blob/main/README.md> for an introduction.
    """

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
        norm_pix_loss=False,
        latent_depth=384,
        hyperprior_depth=192,
        num_slices=12,
        num_keep_patches=144,
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

        # Initialize entropy model
        self.entropy_bottleneck = EntropyBottleneck(hyperprior_depth)
        self.gaussian_conditional = GaussianConditional(None)
        self.max_support_slices = self.num_slices // 2

        # Initialize frozen_stage
        self.frozen_stages = -1

        # Define Compresion Modules

        # G_a Module
        self.g_a = nn.Sequential(
            nn.Conv2d(
                self.encoder_embed_dim,
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4
                ),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4
                ),
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4
                ),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4
                ),
                self.decoder_embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.decoder_embed_dim,
                self.latent_depth,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # G_s Module
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(
                self.latent_depth,
                self.decoder_embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.decoder_embed_dim,
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4
                ),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4
                ),
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4
                ),
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                int(
                    self.decoder_embed_dim
                    + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4
                ),
                self.encoder_embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # H_a Module
        self.h_a = nn.Sequential(
            conv3x3(self.latent_depth, self.latent_depth, stride=1),
            nn.GELU(),
            conv3x3(
                self.latent_depth,
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                stride=2,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                self.hyperprior_depth,
                stride=2,
            ),
        )

        # H_s Module
        self.h_s_mean = nn.Sequential(
            conv3x3(
                self.hyperprior_depth,
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                r=2,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                self.latent_depth,
                r=2,
            ),
            nn.GELU(),
            conv3x3(self.latent_depth, self.latent_depth, stride=1),
        )

        self.h_s_scale = nn.Sequential(
            conv3x3(
                self.hyperprior_depth,
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                r=2,
            ),
            nn.GELU(),
            conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 2 / 4
                ),
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                int(
                    self.hyperprior_depth
                    + (self.latent_depth - self.hyperprior_depth) * 3 / 4
                ),
                self.latent_depth,
                r=2,
            ),
            nn.GELU(),
            conv3x3(self.latent_depth, self.latent_depth, stride=1),
        )

        # CC_Transform Module
        self.cc_transform_mean = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    int(
                        self.latent_depth
                        + (self.latent_depth // self.num_slices)
                        * min(i, self.num_slices // 2)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 3 / 4 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 3 / 4 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 2 / 4 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 2 / 4 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 1 / 4 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 1 / 4 + 1)
                    ),
                    int(self.latent_depth // self.num_slices),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
            for i in range(self.num_slices)
        )

        self.cc_transform_scale = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    int(
                        self.latent_depth
                        + (self.latent_depth // self.num_slices)
                        * min(i, self.num_slices // 2)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 3 / 4 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 3 / 4 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 2 / 4 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 2 / 4 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 1 / 4 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 1 / 4 + 1)
                    ),
                    int(self.latent_depth // self.num_slices),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
            for i in range(self.num_slices)
        )

        # LRP Transform Module
        self.lrp_transform = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    int(
                        self.latent_depth
                        + (self.latent_depth // self.num_slices)
                        * min(i + 1, self.num_slices // 2 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 3 / 4 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 3 / 4 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 2 / 4 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 2 / 4 + 1)
                    ),
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 1 / 4 + 1)
                    ),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    int(
                        self.latent_depth
                        // self.num_slices
                        * (self.num_slices // 2 * 1 / 4 + 1)
                    ),
                    int(self.latent_depth // self.num_slices),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
            for i in range(self.num_slices)
        )

        # Initialize frezze stage status
        self._freeze_stages()

        # Initialize MAE layers

        # ----------------------------------------------------
        # Encoder

        self.encoder_embed = PatchEmbed(
            img_size, patch_size, in_chans, self.encoder_embed_dim
        )
        num_patches = self.encoder_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))
        self.encoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.encoder_embed_dim), requires_grad=False
        )  # Fixed sin-cos embeding

        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.encoder_embed_dim,
                    num_heads=self.encoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(self.encoder_depth)
            ]
        )

        self.encoder_norm = norm_layer(self.encoder_embed_dim)

        # ----------------------------------------------------

        # ----------------------------------------------------
        # Decoder

        self.decoder_embed = nn.Linear(
            self.encoder_embed_dim, self.decoder_embed_dim, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.decoder_embed_dim), requires_grad=False
        )  # Fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.decoder_embed_dim,
                    num_heads=self.decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(self.decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            self.decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )

        # ----------------------------------------------------

        # Initialize norm_pix_loss
        self.norm_pix_loss = norm_pix_loss

        # Initialize weight
        self.initialize_weights()

    # ----------------------------------------------------
    # Initialize the ids shuffle for input images
    def get_ids_shuffle(self, total_score):
        """
        Get the ids_shuffle of the the feature map (patched image) by using total_score and visual_token
        """
        if self.num_keep_patches > len(total_score):
            raise ValueError(
                "Number of patches should not be greater than the length of scores"
            )

        sorted_scores = np.sort(total_score)

        # Calculate percentiles and thresholds
        percentiles = np.arange(10, 91, 10)
        thresholds = np.percentile(np.unique(sorted_scores), percentiles)

        # Categorize data into groups
        categories = np.digitize(sorted_scores, thresholds)

        # Calculate group means
        group_means = np.array(
            [
                np.mean(sorted_scores[categories == group])
                for group in range(len(percentiles) + 1)
            ]
        )

        # Keep values from the group with the highest category (categorized_data == 9)
        keep_values = list(sorted_scores[categories == 9])

        # Apply softmax to group means for other groups
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        softmaxed_means = softmax(group_means[:-1])  # Exclude the last group
        new_target = self.num_keep_patches - len(keep_values)
        scaled_means = np.round(softmaxed_means * new_target)

        # Populate high_category_values
        for i, num_to_keep in enumerate(scaled_means):
            start_index = len(sorted_scores[categories == i]) - num_to_keep
            keep_values.extend(list(sorted_scores[categories == i][int(start_index) :]))

        keep_values.append(sorted_scores[0])  # Append the least important patch
        keep_values_frequency = Counter(keep_values)
        ids_shuffle = []

        # Create a list of indices
        for value, freq in keep_values_frequency.items():
            ids_shuffle.extend(list(np.where(total_score == value)[0][:freq]))

        remaining_indices = [i for i in range(len(total_score)) if i not in ids_shuffle]
        ids_shuffle.extend(remaining_indices)

        return ids_shuffle

    # ----------------------------------------------------

    def _freeze_stages(self):
        """
        Freeze the stages
        """
        if self.frozen_stages >= 0:
            self.encoder_embed.eval()
            for param in self.encoder_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, vis_num, state_dict):
        net = cls(visual_tokens=vis_num)
        net.load_state_dict(state_dict)
        return net

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.encoder_pos_embed.shape[-1],
            int(self.encoder_embed.num_patches**0.5),
            cls_token=True,
        )
        self.encoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.encoder_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.encoder_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Patchify function convert from a image to a patched feature.

        Args:
            imgs (torch.Tensor): Batch of images (N, 3, H, W)

        Returns:
            patched_feature (torch.Tensor): Patched feature (N, L, D) with
                          N: Batch numbers
                          L: (H // patch_size) ** 2
                          D: patch_size ** 2 * 3
        """
        p = self.encoder_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        patched_feature = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return patched_feature

    def unpatchify(self, patched_feature):
        """
        Unpatchify function to convert from a patched feature to image (H, W, 3)
        This function is only used for results after passing the decoder MAE.

        Args:
            patched_feature (torch.Tensor): Patched feature after decode (N, L, patch_size**2 *3)

        Returns:
            imgs (torch.Tensor): Image (N, 3, H, W)
        """
        p = self.encoder_embed.patch_size[0]
        h = w = int(patched_feature.shape[1] ** 0.5)
        assert h * w == patched_feature.shape[1]

        x = patched_feature.reshape(shape=(patched_feature.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, total_scores):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.

        Args:
            x (torch.Tensor): Input patched feature [N, L, D] with
                            N: Batch numbers
                            L: (H // patch_size) ** 2
                            D: patch_size ** 2 * 3

            total_scores (torch.Tensor): Probability map [N, L]

            self.num_keep_patches (int): Number of patches to keep.

        Returns:
            x_remain (torch.Tensor): Remain patched images after masking [N, L_new, D] with
                                    N: Batch numbers
                                    L_new: self.num_keep_patches
                                    D: patch_size ** 2 * 3

            ids_restore (torch.Tensor):  Ids tensor for restore original patched feature.
        """

        # We will generate a tensor IDs shuffle that represents for probability map and produces attention to meaningful patches

        ids_shuffle = self.get_ids_shuffle(total_scores)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(self.num_keep_patches)

        # Sort noise for each sample
        ids_shuffle = torch.tensor(ids_shuffle).view(1, -1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_remain = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_remain, ids_restore

    def foward_encoder(self, imgs, total_scores):
        """
        Encoder module of MCM model

        Args:
            x (torch.Tensor): Batch-like images (N, H, W, 3)
            total_scores (torch.Tensor): Probability map [N, L]

        Returns:
            x_remain (torch.Tensor): Remain patched image [N, L, D] with
                                    N: Batch numbers
                                    L_new: self.num_keep_patches
                                    D: patch_size ** 2 * 3

            ids_restore (torch.Tensor):  Ids tensor for restore original patched feature [N, L]
        """
        # Embeding the input images to patched images
        encoder_imgs = self.encoder_embed(imgs)

        # Add pos_embed w/o cls_token
        encoder_imgs = encoder_imgs + self.encoder_pos_embed[:, 1:, :]

        # Masking: full_length -> num_keep_patches
        x_remain, ids_restore = self.random_masking(encoder_imgs, total_scores)

        # Append cls_token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_token = cls_token.expand(x_remain.shape[0], -1, -1)
        x_remain = torch.cat((cls_token, x_remain), dim=1)

        # Apply Transformer blocks
        for blk in self.encoder_blocks:
            x_remain = blk(x_remain)
        x_remain = self.encoder_norm(x_remain)
        x_remain = x_remain[:, 1:, :]

        return x_remain, ids_restore

    def forward_decoder(self, x_remain, ids_restore):
        """
        Decoder module of MCM model

        Args:
            x_remain (torch.Tensor): Remain patched image [N, L_new, D] with
                                    N: Batch numbers
                                    L_new: self.num_keep_patches
                                    D: patch_size ** 2 * 3

            ids_restore (torch.Tensor):  Ids tensor for restore original patched feature [N, L] with
                                    N: Batch numbers
                                    L: Full patches L=(img_size // patch_size) ** 2


        Returns:
            patched_imgs (torch.Tensor): Patched reconstruction image
        """
        # Decoder embed tokens
        x_decode = self.decoder_embed(
            x_remain
        )  # Covnert [N, 1 + self.num_keep_patches, D] -> [N, 1 + self.num_keep_patches, self.decoder_embed_dim]

        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x_decode.shape[0], ids_restore.shape[1] + 1 - x_decode.shape[1], 1
        )  # Convert [N, L - self.num_keep_patches, self.decoder_embed_dim] into [N, L - self.num_keep_patches]

        x_ = torch.cat([x_decode[:, 1:, :], mask_tokens], dim=1)  # Remove cls_tokens

        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_decode.shape[2])
        )  # Unshuffle corresponding to ids_restore

        x = torch.cat([x_decode[:, :1, :], x_], dim=1)  # Append cls token [N, L + 1, D]

        # Add position embedding
        x = x + self.decoder_pos_embed

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Prediction projection
        x = self.decoder_pred(x)

        # Remove cls token
        patched_imgs = x[:, 1:, :]

        return patched_imgs

    def forward_loss(self, imgs, preds):
        """
        Loss function

        Args:
            imgs (torch.Tensor): Batch-like images (N, H, W, 3)
            preds (torch.Tensor): Batch-like patches (N, L, D) with
                                N: Batch numbers
                                L: (H // patch_size) ** 2
                                D: patch_size ** 2 * 3

        Returns:
            ssim loss, l1_loss, feature_loss
        """
        preds = self.unpatchify(preds)
        ssim = SSIM(
            win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3
        )
        ssim_loss = 1 - ssim(preds, imgs)

        l1_loss = nn.L1Loss()(preds, imgs)
        feature_loss = cal_features_loss(preds, imgs)
        return ssim_loss, l1_loss, feature_loss

    def forward(self, imgs, total_scores):
        """
        MCM model

        Args:
            imgs (torch.Tensor): Batch-like images (N, H, W, 3)
            total_scores (torch.Tensor): Probability map [N, L]

        Returns:
            patched_imgs (torch.Tensor): Patched reconstruction image
        """

        # Encoder
        x_remain, ids_restore = self.foward_encoder(imgs, total_scores)

        # LIC
        y = (
            x_remain.view(
                -1,
                int(self.num_keep_patches**0.5),
                int(self.num_keep_patches**0.5),
                self.encoder_embed_dim,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Apply G_a module
        y = self.g_a(y).float()
        y_shape = y.shape[2:]

        # Apply H_a module
        z = self.h_a(y)
        _, z_likelihood = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()

        z_tmp = z - z_offset
        z_hat = quantize_ste(z_tmp) + z_offset

        # Apply h_s module
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        # Compress using slices
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihoods = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # Calculate mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_transform_mean[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # Calculate scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            sigma = self.cc_transform_scale[slice_index](scale_support)
            sigma = sigma[:, :, : y_shape[0], : y_shape[1]]

            # Calculate y_slice_likelihood
            _, y_slice_likelihood = self.gaussian_conditional(y_slice, sigma, mu)
            y_likelihoods.append(y_slice_likelihood)

            # Calculate y_hat_slice
            y_hat_slice = quantize_ste(y_slice - mu) + mu

            # Calculate lrp transform
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transform[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihood = torch.cat(y_likelihoods, dim=1)

        y_hat = self.g_s(y_hat)
        y_hat = (
            y_hat.permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, self.num_keep_patches, self.encoder_embed_dim)
        )

        # Decoder
        preds = self.forward_decoder(y_hat, ids_restore).float()
        loss = self.forward_loss(imgs, preds)
        x_hat = self.unpatchify(preds)

        return {
            "loss": loss,
            "likelihood": {"y": y_likelihood, "z": z_likelihood},
            "x_hat": x_hat,
        }

    def compress(self, imgs, total_scores):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for autoregressive models. The entropy coder is run sequentially on GPU"
            )

        # Encoder MCM
        x_remain, ids_restore = self.foward_encoder(imgs, total_scores)

        # LIC
        y = (
            x_remain.view(
                -1,
                int(self.num_keep_patches**0.5),
                int(self.num_keep_patches**0.5),
                self.encoder_embed_dim,
            )
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        # Apply G_a module
        y = self.g_a(y).float()
        y_shape = y.shape[2:]

        # Apply H_a module
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # Apply h_s module
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        # Compress using slices
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        # CDF
        cdfs = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        # BufferedRansEncoder Module
        encoder = BufferedRansEncoder()

        # Compress using slices
        y_strings = []
        symbols_list = []
        indexes_list = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # Calculate mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_transform_mean[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # Calculate scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            sigma = self.cc_transform_scale[slice_index](scale_support)
            sigma = sigma[:, :, : y_shape[0], : y_shape[1]]

            index = self.gaussian_conditional.build_indexes(sigma)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            # Calculate lrp transform
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transform[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdfs, cdf_lengths, offsets
        )

        # Get y_string
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {
            "string": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "ids_restore": ids_restore,
        }

    def decompress(self, strings, shape, ids_restore=None):
        assert isinstance(strings, list) and len(strings) == 2

        # Decompress
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        # Apply h_s module
        latent_scales = self.h_s_scale(z_hat)
        latent_means = self.h_s_mean(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0][0]
        y_hat_slices = []

        # Cdf
        cdfs = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        # Decoder bit stream
        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Decompress using slices
        for slice_index in range(self.num_slices):
        # for slice_index in range(3):

            
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[: self.max_support_slices]
            )

            # Calculate mean
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_transform_mean[slice_index](mean_support)
            mu = mu[:, :, : y_shape[0], : y_shape[1]]

            # Calculate scale
            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            sigma = self.cc_transform_scale[slice_index](scale_support)
            sigma = sigma[:, :, : y_shape[0], : y_shape[1]]

            # Index
            index = self.gaussian_conditional.build_indexes(sigma)


            # Revert string indices
            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdfs, cdf_lengths, offsets
            )
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)


            # Lrp transform
            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transform[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)


        y_hat = torch.cat(y_hat_slices, dim=1)

        # Apply G_s module
        y_hat = self.g_s(y_hat)
        y_hat = (
            y_hat.permute(0, 2, 3, 1)
            .contiguous()
            .view(-1, self.num_keep_patches, self.encoder_embed_dim)
        )

        # Decoder MCM
        x_hat = self.forward_decoder(y_hat, ids_restore).float()
        x_hat = self.unpatchify(x_hat)

        return {"x_hat": x_hat}
