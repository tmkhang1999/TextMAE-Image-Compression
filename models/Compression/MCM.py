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

from .loss import vgg
from common.pos_embed import get_2d_sincos_pos_embed


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
        total_score=None,
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

        self.total_score = total_score

        """
        We will generate a tensor IDs shuffle that represents for probability map and produces attention to meaningful patches before passing into the MCM network
        """
        self.ids_shuffle = self.get_ids_shuffle()

        # Define featuremap loss
        self.vgg_loss = vgg.feature_network(
            net_type="vgg16", gpu_ids=[0], requires_grad=False
        )

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
                self.decoder_embed_dim
                + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.decoder_embed_dim
                + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4,
                self.decoder_embed_dim
                + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.Conv2d(
                self.decoder_embed_dim
                + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4,
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
                self.decoder_embed_dim
                + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.decoder_embed_dim
                + (self.encoder_embed_dim - self.decoder_embed_dim) * 2 / 4,
                self.decoder_embed_dim
                + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.decoder_embed_dim
                + (self.encoder_embed_dim - self.decoder_embed_dim) * 3 / 4,
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
                self.hyperprior_depth
                + (self.latent_depth - self.hyperprior_depth) * 3 / 4,
                stride=1,
            ),
            nn.GELU(),
            conv3x3(
                self.hyperprior_depth
                + (self.latent_depth - self.hyperprior_depth) * 3 / 4,
                self.hyperprior_depth
                + (self.latent_depth - self.hyperprior_depth) * 2 / 4,
                stride=2,
            ),
            nn.GELU(),
            conv3x3(
                self.hyperprior_depth
                + (self.latent_depth - self.hyperprior_depth) * 2 / 4,
                self.hyperprior_depth + (self.latent_depth - self.hyperprior_depth) / 4,
                stride=1,
            ),
            nn.GELU(),
            conv3x3(
                self.hyperprior_depth + (self.latent_depth - self.hyperprior_depth) / 4,
                self.hyperprior_depth,
                stride=2,
            ),
        )

        # H_s Module
        self.h_s = nn.Sequential(
            conv3x3(
                self.hyperprior_depth,
                self.hyperprior_depth + (self.latent_depth - self.hyperprior_depth) / 4,
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                self.hyperprior_depth + (self.latent_depth - self.hyperprior_depth) / 4,
                self.hyperprior_depth
                + (self.latent_depth - self.hyperprior_depth) * 2 / 4,
                r=2,
            ),
            nn.GELU(),
            conv3x3(
                self.hyperprior_depth
                + (self.latent_depth - self.hyperprior_depth) * 2 / 4,
                self.hyperprior_depth
                + (self.latent_depth - self.hyperprior_depth) * 3 / 4,
                stride=1,
            ),
            nn.GELU(),
            subpel_conv3x3(
                self.hyperprior_depth
                + (self.latent_depth - self.hyperprior_depth) * 3 / 4,
                self.latent_depth,
                r=2,
            ),
            nn.GELU(),
            conv3x3(self.latent_depth, self.latent_depth, stride=1),
        )

        # CC_Transform Module
        self.cc_transform = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    self.latent_depth
                    + (self.latent_depth // self.num_slices)
                    * min(i, self.num_slices // 2),
                    self.latent_depth // self.num_slices * (self.num_slices // 2 + 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.latent_depth // self.num_slices * (self.num_slices // 2 + 1),
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 3 / 4 + 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 3 / 4 + 1),
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 2 / 4 + 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 2 / 4 + 1),
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 1 / 4 + 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 1 / 4 + 1),
                    self.latent_depth // self.num_slices,
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
                    self.latent_depth
                    + (self.latent_depth // self.num_slices)
                    * min(i + 1, self.num_slices // 2 + 1),
                    self.latent_depth // self.num_slices * (self.num_slices // 2 + 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.latent_depth // self.num_slices * (self.num_slices // 2 + 1),
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 3 / 4 + 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 3 / 4 + 1),
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 2 / 4 + 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 2 / 4 + 1),
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 1 / 4 + 1),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.latent_depth
                    // self.num_slices
                    * (self.num_slices // 2 * 1 / 4 + 1),
                    self.latent_depth // self.num_slices,
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
            torch.zeros(1, num_patches + 1, self.decoder_embed_dim), required_grad=False
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
    def get_ids_shuffle(self):
        """
        Get the ids_shuffle of the the feature map (patched image) by using total_score and visual_token
        """
        if self.num_keep_patches > len(self.total_score):
            raise ValueError(
                "Number of patches should not be greater than the length of scores"
            )

        sorted_scores = np.sort(self.total_score)

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
            ids_shuffle.extend(list(np.where(self.total_score == value)[0][:freq]))

        remaining_indices = [
            i for i in range(len(self.total_score)) if i not in ids_shuffle
        ]
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

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.

        Args:
            x (torch.Tensor): Input patched feature [N, L, D] with
                            N: Batch numbers
                            L: (H // patch_size) ** 2
                            D: patch_size ** 2 * 3

            self.ids_shuffle (torch.Tensor): Shuffle ids tensor
            self.num_keep_patches (int): Number of patches to keep.

        Returns:
            x_remain (torch.Tensor): Remain patched images after masking [N, L_new, D] with
                                  N: Batch numbers
                                  L_new: self.num_keep_patches
                                  D: patch_size ** 2 * 3

            ids_restore (torch.Tensor):  Ids tensor for restore original patched feature.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(self.num_keep_patches)

        # Sort noise for each sample
        ids_shuffle = torch.tensor(self.ids_shuffle).view(1, -1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_remain = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        return x_remain, ids_restore

    def foward_encoder(self, x):
        """
        Encoder module of MCM model
        """
        # Embeding the input images to patched images
        x = self.encoder_embed(x)

        # Add pos_embed w/o cls_token
        x = x + self.encoder_pos_embed[:, 1:, :]

        # Masking: full_length -> num_keep_patches
        x_remain, ids_restore = self.random_masking(x)

        # Append cls_token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :]
        cls_token = cls_token.expand(x_remain.shape[0], -1, -1)
        x_remain = torch.cat((cls_token, x_remain), dim=1)

        # Apply Transformer blocks
        for blk in self.encoder_blocks:
            x_remain = blk(x_remain)
        x_remain = self.encoder_norm(x)
        x_remain = x_remain[:, 1:, :]

        return x_remain, ids_restore
    
    def forward_decoder(self, x_remain, ids_restore):
        """
        Decoder module of MCM model

        Args:
            x_remain (torch.Tensor): 
        """


if __name__ == "__main__":
    print(1)
