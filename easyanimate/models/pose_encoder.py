import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from einops import rearrange

# from cameractrl.models.motion_module import TemporalTransformerBlock


def get_parameter_dtype(parameter: torch.nn.Module):
    try:
        params = tuple(parameter.parameters())
        if len(params) > 0:
            return params[0].dtype

        buffers = tuple(parameter.buffers())
        if len(buffers) > 0:
            return buffers[0].dtype

    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, torch.Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class PoseAdaptor(nn.Module):
    def __init__(self, unet, pose_encoder):
        super().__init__()
        self.unet = unet
        # if pose_encoder:
        self.pose_encoder = pose_encoder

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, pose_embedding):
        # if pose_embedding and self.pose_encoder:
        assert pose_embedding.ndim == 5
        bs = pose_embedding.shape[0]  # b c f h w
        pose_embedding_features = self.pose_encoder(pose_embedding)  # bf c h w
        pose_embedding_features = [rearrange(x, '(b f) c h w -> b c f h w', b=bs) for x in pose_embedding_features]
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, pose_embedding_features=pose_embedding_features).sample
        return noise_pred


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):

    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class PoseAdaptorAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size,  # dimension of hidden state
        pose_feature_dim=None,  # dimension of the pose feature
        cross_attention_dim=None,  # dimension of the text embedding
        query_condition=False,
        key_value_condition=False,
        scale=1.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.pose_feature_dim = pose_feature_dim
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.query_condition = query_condition
        self.key_value_condition = key_value_condition
        assert hidden_size == pose_feature_dim
        if self.query_condition and self.key_value_condition:
            self.qkv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.qkv_merge.weight)
            init.zeros_(self.qkv_merge.bias)
        elif self.query_condition:
            self.q_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.q_merge.weight)
            init.zeros_(self.q_merge.bias)
        else:
            self.kv_merge = nn.Linear(hidden_size, hidden_size)
            init.zeros_(self.kv_merge.weight)
            init.zeros_(self.kv_merge.bias)

    def forward(
        self,
        attn,
        hidden_states,
        pose_feature,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=None,
    ):
        assert pose_feature is not None
        pose_embedding_scale = scale or self.scale

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if hidden_states.dim == 5:
            hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) (h w) c')
        elif hidden_states.ndim == 4:
            hidden_states = rearrange(hidden_states, 'b c h w -> b (h w) c')
        else:
            assert hidden_states.ndim == 3

        if self.query_condition and self.key_value_condition:
            assert encoder_hidden_states is None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if encoder_hidden_states.ndim == 5:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c f h w -> (b f) (h w) c')
        elif encoder_hidden_states.ndim == 4:
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')
        else:
            assert encoder_hidden_states.ndim == 3
        if pose_feature.ndim == 5:
            pose_feature = rearrange(pose_feature, "b c f h w -> (b f) (h w) c")
        elif pose_feature.ndim == 4:
            pose_feature = rearrange(pose_feature, "b c h w -> b (h w) c")
        else:
            assert pose_feature.ndim == 3

        batch_size, ehs_sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, ehs_sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.query_condition and self.key_value_condition:  # only self attention
            query_hidden_state = self.qkv_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
            key_value_hidden_state = query_hidden_state
        elif self.query_condition:
            query_hidden_state = self.q_merge(hidden_states + pose_feature) * pose_embedding_scale + hidden_states
            key_value_hidden_state = encoder_hidden_states
        else:
            key_value_hidden_state = self.kv_merge(encoder_hidden_states + pose_feature) * pose_embedding_scale + encoder_hidden_states
            query_hidden_state = hidden_states

        # original attention
        query = attn.to_q(query_hidden_state)
        key = attn.to_k(key_value_hidden_state)
        value = attn.to_v(key_value_hidden_state)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class TemporalSelfAttention(Attention):
    def __init__(self, attention_mode=None, temporal_position_encoding=False, temporal_position_encoding_max_len=32, rescale_output_factor=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal_Self"

        self.pos_encoder = PositionalEncoding(kwargs["query_dim"], max_len=temporal_position_encoding_max_len) if temporal_position_encoding else None
        self.rescale_output_factor = rescale_output_factor

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None):
        # disable motion module efficient xformers to avoid bad results, don't know why
        # TODO: fix this bug
        pass

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        # add position encoding
        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)
        if "pose_feature" in cross_attention_kwargs:
            pose_feature = cross_attention_kwargs["pose_feature"]
            if pose_feature.ndim == 5:
                pose_feature = rearrange(pose_feature, "b c f h w -> (b h w) f c")
            else:
                assert pose_feature.ndim == 3
            cross_attention_kwargs["pose_feature"] = pose_feature

        if isinstance(self.processor, PoseAdaptorAttnProcessor):
            return self.processor(
                self,
                hidden_states,
                cross_attention_kwargs.pop('pose_feature'),
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        elif hasattr(self.processor, "__call__"):
            return self.processor.__call__(
                self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        else:
            return self.processor(
                self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types=(
            "Temporal_Self",
            "Temporal_Self",
        ),
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=768,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=32,
        encoder_hidden_states_query=(False, False),
        attention_activation_scale=1.0,
        attention_processor_kwargs: Dict = {},
        rescale_output_factor=1.0,
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        self.attention_block_types = attention_block_types

        for block_idx, block_name in enumerate(attention_block_types):
            attention_blocks.append(
                TemporalSelfAttention(
                    attention_mode=block_name,
                    cross_attention_dim=cross_attention_dim if block_name in ['Temporal_Cross', 'Temporal_Pose_Adaptor'] else None,
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rescale_output_factor=rescale_output_factor,
                )
            )
            norms.append(nn.LayerNorm(dim))

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs: Dict[str, Any] = {}):
        for attention_block, norm, attention_block_type in zip(self.attention_blocks, self.norms, self.attention_block_types):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                attention_block(
                    norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states if attention_block_type == 'Temporal_Self' else encoder_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
                + hidden_states
            )

        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states

        output = hidden_states
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        dropout=0.0,
        max_len=32,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, : x.size(1), ...]
        x = x + pe
        return self.dropout(x)


class CameraPoseEncoderCameraCtrl(nn.Module):

    def __init__(
        self,
        downscale_factor,
        channels=[320, 640, 1280, 1280],
        nums_rb=3,
        cin=64,
        ksize=3,
        sk=False,
        use_conv=True,
        compression_factor=1,
        temporal_attention_nhead=8,
        attention_block_types=("Temporal_Self",),
        temporal_position_encoding=False,
        temporal_position_encoding_max_len=16,
        rescale_output_factor=1.0,
    ):
        super(CameraPoseEncoderCameraCtrl, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.channels = channels
        self.nums_rb = nums_rb
        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()
        for i in range(len(channels)):
            conv_layers = nn.ModuleList()
            temporal_attention_layers = nn.ModuleList()
            for j in range(nums_rb):
                if j == 0 and i != 0:
                    in_dim = channels[i - 1]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=True, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == 0:
                    in_dim = channels[0]
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                elif j == nums_rb - 1:
                    in_dim = channels[i] / compression_factor
                    out_dim = channels[i]
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                else:
                    in_dim = int(channels[i] / compression_factor)
                    out_dim = int(channels[i] / compression_factor)
                    conv_layer = ResnetBlock(in_dim, out_dim, down=False, ksize=ksize, sk=sk, use_conv=use_conv)
                temporal_attention_layer = TemporalTransformerBlock(
                    dim=out_dim,
                    num_attention_heads=temporal_attention_nhead,
                    attention_head_dim=int(out_dim / temporal_attention_nhead),
                    attention_block_types=attention_block_types,
                    dropout=0.0,
                    cross_attention_dim=None,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    rescale_output_factor=rescale_output_factor,
                )
                conv_layers.append(conv_layer)
                temporal_attention_layers.append(temporal_attention_layer)
            self.encoder_down_conv_blocks.append(conv_layers)
            self.encoder_down_attention_blocks.append(temporal_attention_layers)

        self.encoder_conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def forward(self, x):
        # unshuffle
        bs = x.shape[0]
        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.encoder_conv_in(x)
        for res_block, attention_block in zip(self.encoder_down_conv_blocks, self.encoder_down_attention_blocks):
            for res_layer, attention_layer in zip(res_block, attention_block):
                x = res_layer(x)
                h, w = x.shape[-2:]
                x = rearrange(x, '(b f) c h w -> (b h w) f c', b=bs)
                x = attention_layer(x)
                x = rearrange(x, '(b h w) f c -> (b f) c h w', h=h, w=w)
            features.append(x)

        return features


class VideoFrameTokenization(nn.Module):
    def __init__(self, output_dim=3072):
        super(VideoFrameTokenization, self).__init__()
        # 通道降维层
        self.conv1 = nn.Conv2d(320, 512, kernel_size=1)
        self.conv2 = nn.Conv2d(640, 512, kernel_size=1)
        self.conv3 = nn.Conv2d(1280, 512, kernel_size=1)
        self.conv4 = nn.Conv2d(1280, 512, kernel_size=1)

        # 自适应平均池化，将所有特征图调整到相同的空间尺寸 (6x10)
        self.avgpool1 = nn.AdaptiveAvgPool2d((6, 10))  # 从 [48, 84] 到 [6, 10]
        self.avgpool2 = nn.AdaptiveAvgPool2d((6, 10))  # 从 [24, 42] 到 [6, 10]
        self.avgpool3 = nn.AdaptiveAvgPool2d((6, 10))  # 从 [12, 21] 到 [6, 10]
        self.avgpool4 = nn.Identity()  # [6, 10] 已经是目标尺寸，无需处理

        # 全连接层，将融合后的特征投影到3072维
        self.fc = nn.Linear(512 * 4, output_dim)

    def forward(self, feature_maps, batch_size):
        """
        参数:
            feature_maps: 一个包含4个特征图的列表，每个特征图的尺寸分别为:
                          [b, 320, 48, 84],
                          [b, 640, 24, 42],
                          [b, 1280, 12, 21],
                          [b, 1280, 6, 10]

        返回:
            tokens: 尺寸为 [b, frames, 3072] 的张量
        """
        # 假设 feature_maps 是一个包含4个特征图的列表
        x1, x2, x3, x4 = feature_maps  # 各特征图的尺寸如上所述

        # 通道降维
        x1 = self.conv1(x1)  # [b, 512, 48, 84]
        x2 = self.conv2(x2)  # [b, 512, 24, 42]
        x3 = self.conv3(x3)  # [b, 512, 12, 21]
        x4 = self.conv4(x4)  # [b, 512, 6, 10]

        # 空间尺寸统一
        x1 = self.avgpool1(x1)  # [b, 512, 6, 10]
        x2 = self.avgpool2(x2)  # [b, 512, 6, 10]
        x3 = self.avgpool3(x3)  # [b, 512, 6, 10]
        x4 = self.avgpool4(x4)  # [b, 512, 6, 10]

        # 特征融合（在通道维拼接）
        x = torch.cat([x1, x2, x3, x4], dim=1)  # [b, 2048, 6, 10]

        # 展平空间维度
        x = x.view(x.size(0), x.size(1), -1)  # [b, 2048, 60]

        # 对空间维度进行池化（平均池化），得到每帧的全局特征
        x = x.mean(dim=2)  # [b, 2048]

        # 投影到3072维
        tokens = self.fc(x)  # [b, 3072]

        # tokens = tokens.unsqueeze(0)

        tokens = rearrange(tokens, "(b f) t -> b f t", b=batch_size)

        # 如果有多个帧，可以对每个帧独立处理，以下假设输入包含多帧
        # 但根据您的描述，49是帧数，可能需要对每帧独立处理
        # 具体实现需要根据输入数据的具体格式进行调整

        return tokens


class CameraPoseEncoder(nn.Module):

    def __init__(self, kernel_size=2):
        super(CameraPoseEncoder, self).__init__()
        self.patch_encoder = nn.Conv2d(32, 3072, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        bs = x.shape[0]  # 1

        x = rearrange(x, 'n c t h w -> (n t) c h w')  # torch.Size([13, 32, 48, 84])

        camera_pose_embedding = self.patch_encoder(x)  # torch.Size([13, 3072, 24, 42])

        camera_pose_embedding = rearrange(camera_pose_embedding, '(b n) l h w -> b (n h w) l', b=bs)  # torch.Size([1, 13104, 3072])

        return camera_pose_embedding
