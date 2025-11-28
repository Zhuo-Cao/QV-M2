# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
FlashMMR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from FlashMMR.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
from FlashMMR.position_encoding import build_position_encoding, PositionEmbeddingSine
from nncore.nn import build_model as build_adapter
from blocks.generator import PointGenerator



def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def find_nth(vid, underline, n):
    max_len = len(vid)
    start = vid.find(underline)
    while start >= 0 and n > 1:
        start = vid.find(underline, start+len(underline))
        n -= 1
    if start == -1:
        start = max_len
    return start

def element_wise_list_equal(listA, listB):
    res = []
    for a, b in zip(listA, listB):
        if a==b:
            res.append(True)
        else:
            res.append(False)
    return res

class ConfidenceScorer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_conv_layers=1, num_mlp_layers=3):
        super(ConfidenceScorer, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        for i in range(num_conv_layers):
            if i == 0:
                self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            else:
                self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=(0, kernel_size[1] // 2)))
            self.activations.append(nn.ReLU(inplace=True))
        
        self.fc = MLP(out_channels, out_channels // 2, 1, num_layers=num_mlp_layers)
    
    def forward(self, x):
        x = x.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        
        for conv, activation in zip(self.convs, self.activations):
            x = conv(x)
            x = activation(x)
        
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.fc(x)
        
        return x

class FlashMMR(nn.Module):
    """ FlashMMR. """

    def __init__(self, transformer, position_embed, txt_position_embed, n_input_proj, input_dropout, txt_dim, vid_dim, aud_dim=0, use_txt_pos=False,
                strides=(1, 2, 4, 8),
                buffer_size=2048,
                max_num_moment=50,
                merge_cls_sal=True,
                pyramid_cfg=None,
                coord_head_cfg=None,
                args=None):
        """ Initializes the model."""
        super().__init__()
        self.args=args
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.PositionEmbeddingSine = PositionEmbeddingSine(hidden_dim, normalize=True)
        
        # input projection
        self.n_input_proj = n_input_proj
        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        # set up dummy token
        self.token_type_embeddings = nn.Embedding(2, hidden_dim)
        self.token_type_embeddings.apply(init_weights)
        self.use_txt_pos = use_txt_pos
        self.dummy_rep_token = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        self.dummy_rep_pos = torch.nn.Parameter(torch.randn(args.num_dummies, hidden_dim))
        normalize_before = False
        input_txt_sa_proj = TransformerEncoderLayer(hidden_dim, 8, self.args.dim_feedforward, 0.1, "prelu", normalize_before)
        txtproj_encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.txtproj_encoder = TransformerEncoder(input_txt_sa_proj, args.dummy_layers, txtproj_encoder_norm)

        # build muti-scale pyramid
        self.pyramid = build_adapter(pyramid_cfg, hidden_dim, strides)
        self.conf_head = ConfidenceScorer(in_channels=256, out_channels=256, kernel_size=(1, args.kernel_size), num_conv_layers=args.num_conv_layers, num_mlp_layers = args.num_mlp_layers)
        self.class_head = ConfidenceScorer(in_channels=256, out_channels=256, kernel_size=(1, args.kernel_size), num_conv_layers=args.num_conv_layers, num_mlp_layers = args.num_mlp_layers)
        self.coef = nn.Parameter(torch.ones(len(strides)))
        self.coord_head = build_adapter(coord_head_cfg, hidden_dim, 2)
        self.generator = PointGenerator(strides, buffer_size)
        self.max_num_moment = max_num_moment
        self.merge_cls_sal = merge_cls_sal
        self.args = args
        self.x = nn.Parameter(torch.tensor(0.5))


    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask, vid, qid, targets=None):
        if self.training:
            raise RuntimeError("Only supports inference.")

        # Project inputs to the same hidden dimension
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        # Add type embeddings
        src_vid = src_vid + self.token_type_embeddings(torch.full_like(src_vid_mask.long(), 1))
        src_txt = src_txt + self.token_type_embeddings(torch.zeros_like(src_txt_mask.long()))
        # Add position embeddings
        pos_vid = self.position_embed(src_vid, src_vid_mask)
        if self.use_txt_pos:
            pos_txt = self.txt_position_embed(src_txt)
        else:
            pos_txt = torch.zeros_like(src_txt, device=src_txt.device)

        # Insert dummy tokens in front of text
        batch_size = src_txt.shape[0]
        txt_dummy = self.dummy_rep_token.unsqueeze(0).expand(batch_size, -1, -1)
        pos_dummy = self.dummy_rep_pos.unsqueeze(0).expand(batch_size, -1, -1)
        mask_txt = torch.ones(
            (batch_size, self.args.num_dummies),
            dtype=torch.bool,
            device=src_txt_mask.device,
        )

        src_txt_dummy = torch.cat([txt_dummy, src_txt], dim=1)
        src_txt_mask_dummy = torch.cat([mask_txt, src_txt_mask], dim=1)
        pos_txt_dummy = torch.cat([pos_dummy, pos_txt], dim=1)

        src_txt_dummy = src_txt_dummy.permute(1, 0, 2)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)

        memory = self.txtproj_encoder(
            src_txt_dummy,
            src_key_padding_mask=~(src_txt_mask_dummy.bool()),
            pos=pos_txt_dummy,
        )
        dummy_token = memory[: self.args.num_dummies].permute(1, 0, 2)
        pos_txt_dummy = pos_txt_dummy.permute(1, 0, 2)

        src_txt_dummy = torch.cat([dummy_token, src_txt], dim=1)
        mask_txt_dummy = torch.tensor([[True] * self.args.num_dummies], device=src_txt_mask.device).repeat(
            src_txt_mask.shape[0], 1
        )
        src_txt_mask_dummy = torch.cat([mask_txt_dummy, src_txt_mask], dim=1)

        src = torch.cat([src_vid, src_txt_dummy], dim=1)
        mask = torch.cat([src_vid_mask, src_txt_mask_dummy], dim=1).bool()
        pos = torch.cat([pos_vid, pos_txt_dummy], dim=1)

        video_length = src_vid.shape[1]

        video_emb, video_msk, _, _, saliency_scores = self.transformer(
            src,
            ~mask,
            pos,
            video_length=video_length,
            saliency_proj1=self.saliency_proj1,
            saliency_proj2=self.saliency_proj2,
        )

        video_emb = video_emb.permute(1, 0, 2)
        video_msk = (~video_msk).int()
        pymid, _ = self.pyramid(video_emb, video_msk, return_mask=False)
        point = self.generator(pymid)

        with torch.autocast("cuda", enabled=False):
            video_emb = video_emb.float()
            out_class = [self.class_head(e.float()) for e in pymid]
            out_class = torch.cat(out_class, dim=1)
            out_conf = torch.cat(pymid, dim=1)
            out_conf = self.conf_head(out_conf)
            out_class = self.x * out_class + (1 - self.x) * out_conf

            out_coord = None
            if self.coord_head is not None and len(pymid) > 0:
                out_coord = [
                    self.coord_head(e.float()).exp() * self.coef[i]
                    for i, e in enumerate(pymid)
                ]
                out_coord = torch.cat(out_coord, dim=1)

        if out_coord is None:
            raise RuntimeError("coord_head did not produce localization results; inference cannot proceed.")

        bs = src_vid.shape[0]
        if bs != 1:
            raise AssertionError("batch size larger than 1 is not supported for inference")

        out_class = out_class.sigmoid()

        boundary = out_coord[0]
        boundary[:, 0] *= -1
        boundary *= point[:, 3, None].repeat(1, 2)
        boundary += point[:, 0, None].repeat(1, 2)
        boundary /= 1 / self.args.clip_length
        boundary = torch.cat((boundary, out_class[0]), dim=-1)

        _, inds = out_class[0, :, 0].sort(descending=True)
        boundary = boundary[inds[: self.max_num_moment]]

        output = dict(
            saliency_scores=saliency_scores,
            _out=dict(
                label=None if targets is None else targets.get("label", [None])[0],
                video_msk=video_msk,
                saliency=saliency_scores[0],
                boundary=boundary,
            ),
        )

        return output



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, input_dim, output_dim, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(input_dim)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):
    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = FlashMMR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        input_dropout=args.input_dropout,
        n_input_proj=args.n_input_proj,
        strides=args.cfg.model.strides,
        buffer_size=args.cfg.model.buffer_size,
        max_num_moment=args.cfg.model.max_num_moment,
        pyramid_cfg=args.cfg.model.pyramid_cfg,
        coord_head_cfg=args.cfg.model.coord_head_cfg,
        args=args
    )

    return model
