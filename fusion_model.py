import torch.nn as nn
import torch
from einops import rearrange, reduce, repeat
from calflops import calculate_flops
import torch.nn.functional as F
from thop import profile
import clip
from PIL import Image


class AverageFusionModel(nn.Module):
    """
    Average fusion model
    """
    def __init__(self, num_patch, emb_size):
        super().__init__()

    def forward(self, patched_feature):
        """
        patched_feature: (bs, num_patch, emb_size)
        """
        new_feature = patched_feature.mean(1)
        return new_feature


class WeightedAverageFusionModel(nn.Module):
    """
    Weighted average fusion model
    """
    def __init__(self, num_patch, emb_size):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_patch, emb_size))

    def forward(self, patched_feature, full_feature, mask=None):
        """
        patched_feature: (bs, num_patch, emb_size)
        """
        new_feature = patched_feature * self.weight
        new_feature = new_feature.sum(1)
        return new_feature


class FCFusionModel(nn.Module):
    """
    FC fusion model
    """
    def __init__(self, num_patch, emb_size):
        super().__init__()
        self.fc = nn.Linear(emb_size * num_patch, emb_size, bias=False)

    def forward(self, patched_feature, full_feature, mask=None):
        """
        patched_feature: (bs, num_patch, emb_size)
        """
        b, p, f = patched_feature.shape
        concat_patch_feature = rearrange(patched_feature, 'b p f -> b (p f)')
        new_feature = self.fc(concat_patch_feature)
        return new_feature


class MLPLocalFusionModel(nn.Module):
    """
    MLP fusing model, local version (reference: PointNet)
    """
    def __init__(self, num_patch, emb_size):
        super(MLPLocalFusionModel, self).__init__()
        init_emb_size = num_patch * emb_size
        # (bs * 50, 512) -> (bs * 50, 256)
        self.shrink_1 = nn.Linear(emb_size, emb_size//2)
        # (bs * 50, 256) -> (bs * 50, 128)
        self.shrink_2 = nn.Linear(emb_size//2, emb_size//4)
        # (bs, 50 * 128) -> (bs, 50 * 64)
        self.fc_3200 = nn.Linear(num_patch * emb_size//4, num_patch * emb_size//8)
        # (bs, 50 * 64) -> (bs, 1024)
        self.fc_1024 = nn.Linear(num_patch * 64, 1024)
        # (bs, 1024) -> (bs, 512)
        self.fc_512 = nn.Linear(1024, emb_size)

        self.bn_3200 = nn.BatchNorm1d(num_patch * emb_size//8)
        self.relu_3200 = nn.LeakyReLU()

        self.bn_1024 = nn.BatchNorm1d(1024)
        self.relu_1024 = nn.LeakyReLU()

        self.bn_512 = nn.BatchNorm1d(512)
        self.relu_512 = nn.LeakyReLU()

    def forward(self, x, full_x, mask=None):
        """
        x: (bs, 50, 512)
        """
        b, patch_num, emb_size = x.shape
        x = rearrange(x, 'b p e -> (b p) e')
        x1 = self.shrink_1(x)
        # (bs, 50, 128)
        x2 = self.shrink_2(x1)
        # (bs, 50 * 128)
        x_shrink = rearrange(x2, '(b p) e -> b (p e)', b=b)

        x_fc_3200 = self.fc_3200(x_shrink)
        x_fc_3200 = self.bn_3200(x_fc_3200)
        x_fc_3200 = self.relu_3200(x_fc_3200)

        x_fc_1024 = self.fc_1024(x_fc_3200)
        x_fc_1024 = self.bn_1024(x_fc_1024)
        x_fc_1024 = self.relu_1024(x_fc_1024)

        x_fc_512 = self.fc_512(x_fc_1024)
        x_fc_512 = self.bn_512(x_fc_512)
        x_fc_512 = self.relu_512(x_fc_512)

        out = x_fc_512

        return out


class MLP3FusionModel(nn.Module):
    """
    MLP fusing model
    """
    def __init__(self, num_patch, emb_size):
        super(MLP3FusionModel, self).__init__()
        init_emb_size = num_patch * emb_size

        # (bs, 50 * 512) -> (bs, 50 * 64)
        # self.fc_3200 = nn.Linear(init_emb_size, num_patch * emb_size//8, bias=False)
        self.fc_3200 = nn.Linear(init_emb_size, num_patch * emb_size//8)
        # (bs, 50 * 64) -> (bs, 1024)
        # self.fc_1024 = nn.Linear(num_patch * 64, 1024, bias=False)
        self.fc_1024 = nn.Linear(num_patch * emb_size//8, 1024)
        # (bs, 1024) -> (bs, 512)
        # self.fc_512 = nn.Linear(1024, emb_size, bias=False)
        self.fc_512 = nn.Linear(1024, emb_size)

        self.bn_3200 = nn.BatchNorm1d(num_patch * emb_size//8)
        self.relu_3200 = nn.LeakyReLU()

        self.bn_1024 = nn.BatchNorm1d(1024)
        self.relu_1024 = nn.LeakyReLU()

        self.bn_512 = nn.BatchNorm1d(512)
        self.relu_512 = nn.LeakyReLU()

    def forward(self, x, full_x, mask=None):
        """
        x: (bs, 50, 512)
        """
        b, patch_num, emb_size = x.shape
        x = rearrange(x, 'b p e -> (b p) e')
        # (bs, 50 * 512)
        x_shrink = rearrange(x, '(b p) e -> b (p e)', b=b)
        # (bs, 3200)
        x_fc_3200 = self.fc_3200(x_shrink)
        x_fc_3200 = self.bn_3200(x_fc_3200)
        x_fc_3200 = self.relu_3200(x_fc_3200)
        # (bs, 1024)
        x_fc_1024 = self.fc_1024(x_fc_3200)
        x_fc_1024 = self.bn_1024(x_fc_1024)
        x_fc_1024 = self.relu_1024(x_fc_1024)
        # (bs, 512)
        x_fc_512 = self.fc_512(x_fc_1024)
        x_fc_512 = self.bn_512(x_fc_512)
        x_fc_512 = self.relu_512(x_fc_512)

        out = x_fc_512

        return out


class TransformerFusionModel(nn.Module):
    """
    Transformer fusing model
    """
    def __init__(self, emb_size, size, ln_eps):
        super(TransformerFusionModel, self).__init__()

        if size == "tiny":
            self.transformer_model = nn.Transformer(
                d_model=emb_size,
                nhead=2 ** 1,
                num_encoder_layers=3,
                num_decoder_layers=3,
                activation="gelu",
                batch_first=True,
                layer_norm_eps=ln_eps
            )

        elif size == "small":
            self.transformer_model = nn.Transformer(
                d_model=emb_size,
                nhead=2 ** 2,
                num_encoder_layers=6,
                num_decoder_layers=6,
                activation="gelu",
                batch_first=True,
                layer_norm_eps=ln_eps
            )
        elif size == "base":
            self.transformer_model = nn.Transformer(
                d_model=emb_size,
                nhead=2 ** 3,
                num_encoder_layers=12,
                num_decoder_layers=12,
                activation="gelu",
                batch_first=True,
                layer_norm_eps=ln_eps
            )

    def forward(self, patch_feat, full_feat, obj_mask=None):
        """

        :param patch_feat: (bs, num_patch, 512)
        :param full_feat: (bs, 1, 512)
        :param obj_mask: (bs, num_patch, 512)
        :return: (bs, 512)
        """

        # patch_cc or patch_grid, no padding is needed
        if obj_mask is None:
            out = self.transformer_model(patch_feat, full_feat)
        # patch obj, need padding and mask
        else:
            out = self.transformer_model(patch_feat, full_feat, src_key_padding_mask=obj_mask)

        out = out.squeeze(1)
        return out


class TransformerLocalLizationModel(nn.Module):
    """
    Transformer fusing model with localization ability
    """
    def __init__(self, num_patch, emb_size):
        super(TransformerLocalLizationModel, self).__init__()
        self.num_patch = num_patch
        self.fusion_model = nn.Transformer(
            d_model=emb_size,
            nhead=16,
            num_encoder_layers=6,
            num_decoder_layers=6,
            # dim_feedforward=emb_size * 10,
            activation="gelu",
            batch_first=True
        )

        self.g = nn.Sequential(
            # (b, p, 2 * d) -> (b, p, p)
            nn.Linear(2 * emb_size, num_patch)
        )

        self.softmax = nn.Softmax(-1)
        self.mse = nn.MSELoss()

    def forward(self, patch_feat, full_feat, mask=None):
        """

        :param patch_feat: (bs, num_patch, 512)
        :param full_feat: (bs, 1, 512)
        :return: (bs, 512)
        """
        b, p, d = patch_feat.shape
        # (b, p, 512) -> (b, 1, 512)
        v = self.fusion_model(patch_feat, full_feat)

        # (b, 1, 512) -> (b, p, 512)
        v_repeat = v.repeat(1, p, 1)
        # (b, p, 2 * 512)
        v_b_concat = torch.cat([v_repeat, patch_feat], -1)
        # (b, p, p)
        localization_pred = self.softmax(self.g(v_b_concat))
        # (b, p, p)
        gt = torch.eye(p).repeat(b, 1, 1)
        self_supervised_loss = self.mse(localization_pred, gt)

        return out, self_supervised_loss


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


# def profile():
#     src = torch.rand((1, 3, 224, 224))
#     model = TransformerFusionModel(512, "small", 1e-5)
#
#     batch_size = 1
#     input_shape = ((166, 3, 224, 224), (batch_size, 3, 224, 224))
#     flops, macs, params = calculate_flops(model=model,
#                                           input_shape=input_shape,
#                                           output_as_string=True,
#                                           output_precision=4)
#     print("Transformer Fusion Model FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


def profile_fusion_model():
    fusion_model = TransformerFusionModel(512, "small", 1e-5)
    fusion_model.eval()
    fusion_model_flops, fusion_model_params = profile(fusion_model, inputs=(torch.randn(1, 166, 512), torch.randn(1, 1, 512)))
    print('fusion_model FLOPs = ' + str(fusion_model_flops/1000**3) + ' G')
    print('fusion_model Params = ' + str(fusion_model_params/1000**2) + ' M')


def profile_attn_pool(model_size="small"):
    if model_size == "small":
        attnpool_model = AttentionPool2d(224 // 32, 2048, 32, 1024)
        attnpool_model.eval()
        attnpool_model_flops, attnpool_model_macs, attnpool_model_params = calculate_flops(model=attnpool_model,
                                              input_shape=(1, 2048, 7, 7),
                                              output_as_string=True,
                                              output_precision=4)

    elif model_size == "large":
        attnpool_model = AttentionPool2d(448 // 32, 4096, 64, 1024)
        attnpool_model_flops, attnpool_model_macs, attnpool_model_params = calculate_flops(model=attnpool_model,
                                              input_shape=(1, 4096, 14, 14),
                                              output_as_string=True,
                                              output_precision=4)

    print('AttentionPool2d FLOPs = ' + str(attnpool_model_flops))
    print('AttentionPool2d MACS = ' + str(attnpool_model_macs))
    print('AttentionPool2d Params = ' + str(attnpool_model_params))


def profile_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model = model.visual

    # image = preprocess(Image.open("/home/zilun/Pictures/Screenshots/Screenshot from 2022-10-16 22-38-33.png")).unsqueeze(0).to(device)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

    # image = torch.randn(1, 166, 3, 224, 224).half().to(device)
    # clip_model_flops, fusion_model_params = profile(model, inputs=(image))

    clip_model_flops, clip_model_macs, clip_model_params = calculate_flops(model=model,
                                          input_shape=(1, 3, 224, 224),
                                          output_as_string=True,
                                          output_precision=4)

    print('CLIP FLOPs = ' + str(clip_model_flops))
    print('CLIP MACS = ' + str(clip_model_macs))
    print('CLIP Params = ' + str(clip_model_params))

    # with torch.no_grad():
    #     image_features = model.encode_image(image)
    #     text_features = model.encode_text(text)
    #     logits_per_image, logits_per_text = model(image, text)
    #     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    # print(probs)

    # print('clip_model FLOPs = ' + str(fusion_model_flops/1000**3) + ' G')
    # print('clip_model Params = ' + str(fusion_model_params/1000**2) + ' M')


def main():
    profile_fusion_model()
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    profile_attn_pool("large")
    print("--------------------------------------------------------------------------------------------------------------------------------------------")
    profile_clip()


if __name__ == "__main__":
    main()







