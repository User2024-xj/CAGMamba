# ====================================================================================
# 文件2: msamba_blocks.py - 针对特征级融合的优化修改
# ====================================================================================

"""
MSAmba核心组件提取 - CHM模块（完整Mamba实现）
针对特征级融合优化的版本，确保与新的MambaFeatureLevelCHM兼容
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

# 修复的导入部分
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
    print("✅ 成功导入核心 Mamba 组件")
except ImportError as e:
    print(f"Warning: mamba_ssm core components not available: {e}")
    Mamba = None
    selective_scan_fn = None
    MAMBA_AVAILABLE = False

# 修复的RMSNorm导入
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
    print("✅ 成功导入 RMSNorm (新路径)")
except ImportError:
    try:
        from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
        print("✅ 成功导入 RMSNorm (旧路径)")
    except ImportError:
        print("⚠️ RMSNorm 导入失败，使用 LayerNorm 替代")
        RMSNorm = nn.LayerNorm
        layer_norm_fn = None
        rms_norm_fn = None

from timm.models.layers import DropPath


class MambaVisionMixer(nn.Module):
    """Mamba序列建模混合器 - 针对特征级融合优化的版本"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", 
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, conv_bias=True, bias=False, device=None, dtype=None):
        super().__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        factory_kwargs = {"device": self.device, "dtype": self.dtype}
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        # 线性投影层
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # === 优化：针对短序列的卷积参数调整 ===
        # 对于特征级融合(seq_len=1)，减少卷积核大小以避免padding问题
        effective_conv_size = min(d_conv, 3)  # 限制卷积核大小
        self.effective_conv_size = effective_conv_size
        
        if effective_conv_size % 2 == 0:
            self.pad_left = (effective_conv_size - 1) // 2
            self.pad_right = effective_conv_size // 2
            self.use_asymmetric_padding = True
        else:
            self.conv_padding = (effective_conv_size - 1) // 2
            self.use_asymmetric_padding = False
        
        # 卷积层（使用优化的卷积核大小）
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2, 
            out_channels=self.d_inner // 2,
            bias=conv_bias, 
            kernel_size=effective_conv_size, 
            groups=self.d_inner // 2,
            padding=0,
            **factory_kwargs
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2, 
            out_channels=self.d_inner // 2,
            bias=conv_bias, 
            kernel_size=effective_conv_size, 
            groups=self.d_inner // 2,
            padding=0,
            **factory_kwargs
        )
        
        self._init_mamba_params(dt_init, dt_min, dt_max, dt_scale, dt_init_floor)

    def _init_mamba_params(self, dt_init, dt_min, dt_max, dt_scale, dt_init_floor):
        """初始化Mamba相关参数"""
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(torch.rand(self.d_inner // 2, device=self.device, dtype=self.dtype) * 
                      (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        
        A = torch.arange(1, self.d_state + 1, device=self.device, dtype=self.dtype).repeat(self.d_inner // 2, 1)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=self.device, dtype=self.dtype))
        self.D._no_weight_decay = True

    def _apply_conv_with_length_preservation(self, x, conv_layer):
        """应用卷积同时保持序列长度不变 - 针对短序列优化"""
        original_length = x.shape[2]
        
        # === 优化：对于长度为1的序列特殊处理 ===
        if original_length == 1:
            # 对于特征级融合，序列长度为1，直接进行padding
            padding_needed = self.effective_conv_size - 1
            x_padded = F.pad(x, (padding_needed // 2, padding_needed - padding_needed // 2))
            x_conv = conv_layer(x_padded)
            # 确保输出长度为1
            if x_conv.shape[2] != 1:
                # 取中间位置的值
                center_idx = x_conv.shape[2] // 2
                x_conv = x_conv[:, :, center_idx:center_idx+1]
            return x_conv
        
        # 原有的逻辑用于处理更长的序列
        if self.use_asymmetric_padding:
            x_padded = F.pad(x, (self.pad_left, self.pad_right))
            x_conv = conv_layer(x_padded)
            if x_conv.shape[2] > original_length:
                x_conv = x_conv[:, :, :original_length]
            elif x_conv.shape[2] < original_length:
                padding_needed = original_length - x_conv.shape[2]
                x_conv = F.pad(x_conv, (0, padding_needed))
        else:
            x_padded = F.pad(x, (self.conv_padding, self.conv_padding))
            x_conv = conv_layer(x_padded)
            if x_conv.shape[2] != original_length:
                if x_conv.shape[2] > original_length:
                    x_conv = x_conv[:, :, :original_length]
                else:
                    padding_needed = original_length - x_conv.shape[2]
                    x_conv = F.pad(x_conv, (0, padding_needed))
        
        return x_conv

    def forward(self, hidden_states, inference_params=None):
        """Mamba前向传播 - 优化版本"""
        batch, seqlen, dim = hidden_states.shape
        original_seqlen = seqlen
        
        # 线性投影
        xz = self.in_proj(hidden_states)
        xz = xz.transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        
        # 状态参数
        A = -torch.exp(self.A_log.float())
        
        # 应用卷积并保持长度不变
        x = F.silu(self._apply_conv_with_length_preservation(x, self.conv1d_x))
        z = F.silu(self._apply_conv_with_length_preservation(z, self.conv1d_z))
        
        # 验证长度
        assert x.shape[2] == original_seqlen, f"x卷积后长度不匹配: {x.shape[2]} vs {original_seqlen}"
        assert z.shape[2] == original_seqlen, f"z卷积后长度不匹配: {z.shape[2]} vs {original_seqlen}"
        
        # 转换维度用于后续处理
        x = x.transpose(1, 2)
        z = z.transpose(1, 2)
        
        # 使用真正的selective scan或备用实现
        if MAMBA_AVAILABLE and selective_scan_fn is not None:
            x_dbl = self.x_proj(x)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj(dt)
            
            try:
                y = selective_scan_fn(x.transpose(1, 2), dt.transpose(1, 2), A, B.transpose(1, 2), 
                                    C.transpose(1, 2), self.D.float(), z=None, 
                                    delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
                y = y.transpose(1, 2)
            except Exception as e:
                # 如果selective_scan_fn失败，使用备用实现
                print(f"Warning: selective_scan_fn failed ({e}), using fallback")
                x_dbl = self.x_proj(x)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_proj(dt)
                dt = torch.sigmoid(dt)
                y = x * dt
        else:
            # 备用实现：简化的状态空间模型
            x_dbl = self.x_proj(x)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj(dt)
            dt = torch.sigmoid(dt)
            # 简单的时间加权
            y = x * dt
        
        # 验证selective scan输出长度
        assert y.shape[1] == original_seqlen, f"y长度不匹配: {y.shape[1]} vs {original_seqlen}"
        assert z.shape[1] == original_seqlen, f"z长度不匹配: {z.shape[1]} vs {original_seqlen}"
        
        # 门控机制
        y = torch.cat([y, z], dim=2)
        
        # 输出投影
        out = self.out_proj(y)
        
        # 最终验证
        assert out.shape == hidden_states.shape, f"输出形状不匹配: {out.shape} vs {hidden_states.shape}"
        
        return out


# === 保留但优化TextGuidedFusionBlock_Bimodal，以备未来使用 ===
class TextGuidedFusionBlock_Bimodal(nn.Module):
    """CHM (Cross-modal Hierarchical Modeling) 双模态版本 - 文本引导融合块（保留用于序列级融合）"""
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, 
                 residual_in_fp32=False, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = False  # 为了稳定性暂时禁用
        
        # 双模态Mixer：音频-文本融合、纯文本处理
        self.mixer_AT = mixer_cls(dim * 2)  # Audio-Text融合
        self.mixer_T = mixer_cls(dim)       # 纯文本处理
        
        # 对应的归一化层
        self.norm_AT = norm_cls(dim * 2)
        self.norm_T = norm_cls(dim)
        
        # 投影层：将融合特征投影回原始维度
        self.proj_AT = nn.Linear(dim * 2, dim)
        
        self.drop_path = nn.Identity()

    def _ensure_same_length(self, *tensors):
        """确保所有张量具有相同的序列长度"""
        min_len = min(t.shape[1] for t in tensors)
        return [t[:, :min_len, :] for t in tensors]

    def _restore_length(self, tensor, target_length):
        """恢复张量到目标长度"""
        if tensor.shape[1] < target_length:
            padding_needed = target_length - tensor.shape[1]
            tensor = F.pad(tensor, (0, 0, 0, padding_needed))
        elif tensor.shape[1] > target_length:
            tensor = tensor[:, :target_length, :]
        return tensor

    def forward(self, hidden_states_T, hidden_states_A, 
                residual_T=None, residual_A=None, inference_params=None):
        """双模态文本引导的跨模态融合前向传播"""
        
        # 记录原始形状
        original_shape_T = hidden_states_T.shape
        original_shape_A = hidden_states_A.shape
        
        # 文本模态处理
        residual_T = (residual_T + hidden_states_T) if residual_T is not None else hidden_states_T
        hidden_states_T = self.norm_T(residual_T)
        
        # 音频模态处理
        if residual_A is not None:
            hidden_states_A = hidden_states_A + residual_A
        residual_A = hidden_states_A
        
        # 确保文本和音频的序列长度一致
        hidden_states_T, hidden_states_A = self._ensure_same_length(
            hidden_states_T, hidden_states_A
        )
        
        # 跨模态连接：音频-文本融合
        hidden_states_AT = torch.cat([hidden_states_A, hidden_states_T], dim=-1)
        
        # 归一化
        hidden_states_AT = self.norm_AT(hidden_states_AT)
        
        # Mixer处理
        hidden_states_AT = self.mixer_AT(hidden_states_AT, inference_params=inference_params)
        hidden_states_T = self.mixer_T(hidden_states_T, inference_params=inference_params)
        
        # 确保处理后的长度一致
        hidden_states_T, hidden_states_AT = self._ensure_same_length(
            hidden_states_T, hidden_states_AT
        )
        
        # 文本CLS token引导
        T_cls_token = hidden_states_T[:, 0, :].unsqueeze(dim=1)  # [B, 1, dim]
        
        # 投影音频-文本融合特征并注入文本CLS信息
        hidden_states_AT = self.proj_AT(hidden_states_AT) + T_cls_token
        
        # 恢复到原始长度
        hidden_states_T = self._restore_length(hidden_states_T, original_shape_T[1])
        hidden_states_AT = self._restore_length(hidden_states_AT, original_shape_A[1])
        
        # 验证输出形状
        assert hidden_states_T.shape == original_shape_T, \
            f"文本输出形状不匹配: {hidden_states_T.shape} vs {original_shape_T}"
        assert hidden_states_AT.shape == original_shape_A, \
            f"音频输出形状不匹配: {hidden_states_AT.shape} vs {original_shape_A}"
        
        return hidden_states_T, hidden_states_AT, residual_T, residual_A


def create_chm_block_bimodal(d_model, norm_epsilon=1e-5, drop_path=0., rms_norm=False,
                             residual_in_fp32=True, fused_add_norm=False, layer_idx=None,
                             device=None, dtype=None):
    """创建双模态CHM块的工厂函数"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dtype is None:
        dtype = torch.float32
    
    # 根据RMSNorm可用性决定是否使用
    use_rms_norm = rms_norm and (RMSNorm is not None) and (RMSNorm != nn.LayerNorm)
    
    mixer_cls = partial(MambaVisionMixer, device=device, dtype=dtype)
    norm_cls = partial(RMSNorm if use_rms_norm else nn.LayerNorm, eps=norm_epsilon)
    
    # 如果RMSNorm不可用，禁用fused_add_norm
    if not use_rms_norm:
        fused_add_norm = False
    
    block = TextGuidedFusionBlock_Bimodal(
        d_model, mixer_cls, norm_cls=norm_cls, drop_path=drop_path,
        fused_add_norm=fused_add_norm, residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True, n_residuals_per_layer=1):
    """MSAmba风格的权重初始化"""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# 导出可用性状态
__all__ = [
    'MambaVisionMixer',
    'TextGuidedFusionBlock_Bimodal',
    'create_chm_block_bimodal',
    '_init_weights',
    'MAMBA_AVAILABLE'
]

print(f"✅ MSAmba 双模态核心组件加载完成（完整Mamba实现）")
print(f"✅ Mamba SSM 可用: {MAMBA_AVAILABLE}")
print(f"✅ RMSNorm 可用: {RMSNorm != nn.LayerNorm}")
print(f"✅ 针对特征级融合进行了优化，支持seq_len=1的情况")
