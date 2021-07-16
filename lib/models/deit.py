import torch
import torch.nn as nn
from functools import partial


from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = [
    'deit_cam_tiny_patch16_224', 'deit_cam_small_patch16_224', 'deit_cam_base_patch16_224',
]

class TSCAM(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(self.embed_dim, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.extra_cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        #self.extra_pos_embed = nn.Parameter(torch.zeros(1,  self.num_classes, self.embed_dim))

        trunc_normal_(self.extra_cls_token, std=.02)
        #trunc_normal_(self.extra_pos_embed, std=.02)
        self.head.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        extra_cls_token = self.extra_cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #pos_embed = torch.cat((self.pos_embed, self.extra_pos_embed), dim=1)
        x = x + self.pos_embed
        x = torch.cat((x, extra_cls_token), dim=1)
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, self.patch_embed.num_patches+1:], attn_weights

    def forward(self, x, return_cam=False):
        x_patch, attn_weights = self.forward_features(x)
        n, c, dim = x_patch.shape
        x_patch = self.head(x_patch)       # B * C * 1
        x_logits = x_patch.squeeze(-1)

        if self.training:
            return x_logits
        else:
            attn_weights = torch.stack(attn_weights)
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N
            residual_att = torch.eye(attn_weights.size(2)).unsqueeze(0).unsqueeze(1).to(attn_weights.get_device())    # 12 * B * N * N
            aug_att_mat = attn_weights + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            joint_attns = torch.zeros(aug_att_mat.size()).to(attn_weights.get_device())
            joint_attns_skip = torch.zeros(attn_weights.size()).to(attn_weights.get_device())
            joint_attns[0] = aug_att_mat[0]
            joint_attns_skip[0] = attn_weights[0]
            for n in range(1, aug_att_mat.size()[0]):
                joint_attns[n] = torch.matmul(aug_att_mat[n], joint_attns[n-1])
                joint_attns_skip[n] = torch.matmul(attn_weights[n], joint_attns_skip[n-1])

            #feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            #n, c, h, w = feature_map.shape
            #cams = joint_attns[-1][:, 0, 1:].reshape([n, h, w]).unsqueeze(1)       # B * 1 * 14 * 14
            #cams = attn_weights.mean(0).mean(1)[:, 1:].reshape([n, h, w]).unsqueeze(1)
            #cams = attn_weights.sum(0)[:,0,1:].reshape([n, h, w]).unsqueeze(1)
            #cams = cams * feature_map                           # B * C * 14 * 14
            _, B, N, N = attn_weights.shape
            h = w = int(self.patch_embed.num_patches**0.5)
            cams_cls = attn_weights.sum(0)[:, self.patch_embed.num_patches+1:, 1:self.patch_embed.num_patches+1]
            cams_cls = cams_cls.reshape([B,self.num_classes, h, w])

            return x_logits, cams_cls


@register_model
def deit_cam_tiny_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@register_model
def deit_cam_small_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@register_model
def deit_cam_base_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@register_model
def deit_cam_base_patch16_384(pretrained=False, **kwargs):
    model = TSCAM(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model





