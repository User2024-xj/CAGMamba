# ====================================================================================
# 文件: msamba_mmml_model_sims.py - SIMS数据集适配版本
# ====================================================================================

"""
MSAmba-MMML集成模型 - SIMS数据集版本（使用完整的Mamba实现）
适配中文RoBERTa和Hubert模型，保持完整Mamba CHM架构
"""
import torch
from torch import nn
from transformers import AutoModel
import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入完整的MSAmba组件
try:
    from msamba_blocks import (
        MambaVisionMixer,
        TextGuidedFusionBlock_Bimodal,
        create_chm_block_bimodal,
        _init_weights,
        MAMBA_AVAILABLE
    )
    print("✅ 成功导入完整MSAmba组件（SIMS版本）")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure msamba_blocks.py is in the same directory")
    raise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MambaFeatureLevelCHM_SIMS(nn.Module):
    """基于Mamba的特征级别CHM - SIMS版本（使用真正的Mamba机制）"""
    
    def __init__(self, text_dim, audio_dim, fusion_dim=None, mamba_config=None):
        super().__init__()
        
        if fusion_dim is None:
            fusion_dim = max(text_dim, audio_dim)
        
        self.fusion_dim = fusion_dim
        
        # 维度对齐
        self.text_proj = nn.Linear(text_dim, fusion_dim) if text_dim != fusion_dim else nn.Identity()
        self.audio_proj = nn.Linear(audio_dim, fusion_dim) if audio_dim != fusion_dim else nn.Identity()
        
        # Mamba配置
        if mamba_config is None:
            mamba_config = {
                'd_state': 16,
                'd_conv': 4, 
                'expand': 2,
                'dt_rank': 'auto',
                'device': device,
                'dtype': torch.float32
            }
        
        # 使用完整的Mamba混合器进行跨模态融合
        self.mamba_cross_modal = MambaVisionMixer(
            d_model=fusion_dim * 2,  # 文本+音频拼接维度
            **mamba_config
        )
        
        # 纯文本和纯音频的Mamba处理
        self.mamba_text = MambaVisionMixer(
            d_model=fusion_dim,
            **mamba_config
        )
        
        self.mamba_audio = MambaVisionMixer(
            d_model=fusion_dim,
            **mamba_config
        )
        
        # 归一化层
        self.norm_cross = nn.LayerNorm(fusion_dim * 2)
        self.norm_text = nn.LayerNorm(fusion_dim)
        self.norm_audio = nn.LayerNorm(fusion_dim)
        
        # 投影层：将融合特征投影回原始维度
        self.proj_cross_to_text = nn.Linear(fusion_dim * 2, fusion_dim)
        self.proj_cross_to_audio = nn.Linear(fusion_dim * 2, fusion_dim)
        
        # 门控机制
        self.text_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
        self.audio_gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim), 
            nn.Sigmoid()
        )
        
    def forward(self, text_features, audio_features):
        """
        text_features: [B, text_dim] 
        audio_features: [B, audio_dim]
        """
        batch_size = text_features.shape[0]
        
        # 维度对齐
        text_proj = self.text_proj(text_features)  # [B, fusion_dim]
        audio_proj = self.audio_proj(audio_features)  # [B, fusion_dim]
        
        # 添加序列维度进行Mamba处理 [B, 1, dim] -> 模拟序列长度为1的情况
        text_seq = text_proj.unsqueeze(1)  # [B, 1, fusion_dim]
        audio_seq = audio_proj.unsqueeze(1)  # [B, 1, fusion_dim]
        
        # === Mamba跨模态融合 ===
        # 拼接文本和音频特征
        cross_modal_seq = torch.cat([audio_seq, text_seq], dim=-1)  # [B, 1, fusion_dim*2]
        
        # 归一化
        cross_modal_seq = self.norm_cross(cross_modal_seq)
        
        # Mamba处理跨模态特征
        cross_modal_enhanced = self.mamba_cross_modal(cross_modal_seq)  # [B, 1, fusion_dim*2]
        
        # 投影回单模态维度
        text_from_cross = self.proj_cross_to_text(cross_modal_enhanced)  # [B, 1, fusion_dim]
        audio_from_cross = self.proj_cross_to_audio(cross_modal_enhanced)  # [B, 1, fusion_dim]
        
        # === 单模态Mamba处理 ===
        # 纯文本处理
        text_seq_norm = self.norm_text(text_seq)
        text_enhanced = self.mamba_text(text_seq_norm)  # [B, 1, fusion_dim]
        
        # 纯音频处理  
        audio_seq_norm = self.norm_audio(audio_seq)
        audio_enhanced = self.mamba_audio(audio_seq_norm)  # [B, 1, fusion_dim]
        
        # === 门控融合 ===
        # 使用门控机制融合跨模态信息和单模态信息
        cross_info_for_text = torch.cat([text_from_cross, audio_from_cross], dim=-1)
        cross_info_for_audio = torch.cat([audio_from_cross, text_from_cross], dim=-1)
        
        text_gate_weight = self.text_gate(cross_info_for_text)  # [B, 1, fusion_dim]
        audio_gate_weight = self.audio_gate(cross_info_for_audio)  # [B, 1, fusion_dim]
        
        # 门控融合：单模态增强 + 跨模态信息
        text_final = text_enhanced + text_gate_weight * text_from_cross
        audio_final = audio_enhanced + audio_gate_weight * audio_from_cross
        
        # 去掉序列维度，返回特征级别的结果
        text_output = text_final.squeeze(1)  # [B, fusion_dim]
        audio_output = audio_final.squeeze(1)  # [B, fusion_dim]
        
        return text_output, audio_output


class MSAmba_MMML_SIMS(nn.Module):
    """
    MSAmba-MMML SIMS数据集模型 - 使用完整的Mamba CHM实现
    适配中文RoBERTa和Hubert模型
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 配置参数
        self.config = config
        self.fusion_depth = getattr(config, 'chm_depth', 1)
        
        # 中文模型路径配置
        text_model_path = getattr(config, 'text_model_path', '/home/yyk/yyk09/models/chinese-roberta-wwm-ext')
        audio_model_path = getattr(config, 'audio_model_path', '/home/yyk/yyk09/models/chinese-hubert-base')
        
        # 模型加载逻辑
        def check_and_get_model_path(model_path, model_type="text"):
            import os
            
            if os.path.exists(model_path):
                print(f"✓ 找到本地{model_type}模型: {model_path}")
                return model_path
            else:
                print(f"⚠ 本地{model_type}模型不存在: {model_path}")
                
                if model_type == "text":
                    fallback_paths = [
                        '/home/yyk/yyk09/models/chinese-roberta-wwm-ext',
                        'hfl/chinese-roberta-wwm-ext'
                    ]
                else:  # audio
                    fallback_paths = [
                        '/home/yyk/yyk09/models/chinese-hubert-base',
                        'TencentGameMate/chinese-hubert-base'
                    ]
                
                for fallback in fallback_paths:
                    if os.path.exists(fallback):
                        print(f"✓ 使用备选{model_type}模型: {fallback}")
                        return fallback
                    elif not fallback.startswith('/'):
                        print(f"尝试HuggingFace {model_type}模型: {fallback}")
                        return fallback
                
                return model_path
        
        text_model_path = check_and_get_model_path(text_model_path, "text")
        audio_model_path = check_and_get_model_path(audio_model_path, "audio")
        
        # 加载中文文本模型
        try:
            print(f"正在加载中文文本模型: {text_model_path}")
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            self.roberta_model = AutoModel.from_pretrained(
                text_model_path,
                local_files_only=True
            )
            print("✓ 中文文本模型加载成功")
        except Exception as e:
            print(f"✗ 本地中文文本模型加载失败: {e}")
            print("尝试在线加载...")
            try:
                if 'TRANSFORMERS_OFFLINE' in os.environ:
                    del os.environ['TRANSFORMERS_OFFLINE']
                self.roberta_model = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
                print("✓ 在线中文文本模型加载成功")
            except Exception as e2:
                raise RuntimeError(f"无法加载任何中文文本模型: {e2}")
        
        # 加载中文音频模型
        try:
            print(f"正在加载中文音频模型: {audio_model_path}")
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            self.hubert_model = AutoModel.from_pretrained(
                audio_model_path,
                local_files_only=True
            )
            print("✓ 中文音频模型加载成功")
        except Exception as e:
            print(f"✗ 本地中文音频模型加载失败: {e}")
            print("尝试在线加载...")
            try:
                if 'TRANSFORMERS_OFFLINE' in os.environ:
                    del os.environ['TRANSFORMERS_OFFLINE']
                self.hubert_model = AutoModel.from_pretrained('TencentGameMate/chinese-hubert-base')
                print("✓ 在线中文音频模型加载成功")
            except Exception as e2:
                raise RuntimeError(f"无法加载任何中文音频模型: {e2}")
        
        # 清理环境变量
        if 'TRANSFORMERS_OFFLINE' in os.environ:
            del os.environ['TRANSFORMERS_OFFLINE']
        if 'HF_DATASETS_OFFLINE' in os.environ:
            del os.environ['HF_DATASETS_OFFLINE']
        
        # 获取实际的模型维度（中文模型通常都是768维）
        text_hidden_size = self.roberta_model.config.hidden_size
        audio_hidden_size = self.hubert_model.config.hidden_size
        
        print(f"中文文本模型隐藏维度: {text_hidden_size}")
        print(f"中文音频模型隐藏维度: {audio_hidden_size}")
        
        # === 核心修改：使用完整的Mamba CHM层 ===
        # Mamba配置
        self.mamba_config = {
            'd_state': getattr(config, 'mamba_d_state', 16),
            'd_conv': getattr(config, 'mamba_d_conv', 4),
            'expand': getattr(config, 'mamba_expand', 2),
            'dt_rank': 'auto',
            'device': device,
            'dtype': torch.float32
        }
        
        # 创建基于Mamba的CHM层
        self.chm_layers = nn.ModuleList([
            MambaFeatureLevelCHM_SIMS(
                text_hidden_size, 
                audio_hidden_size, 
                mamba_config=self.mamba_config
            ) 
            for _ in range(self.fusion_depth)
        ])
        
        print(f"✅ 使用完整Mamba CHM层（SIMS版本），共{self.fusion_depth}层")
        print(f"✅ Mamba配置: d_state={self.mamba_config['d_state']}, d_conv={self.mamba_config['d_conv']}")
        print(f"✅ Mamba可用性: {MAMBA_AVAILABLE}")
        
        # 输出层 - 适配SIMS数据集的多任务格式
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(text_hidden_size, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(audio_hidden_size, 1)
        )
        
        # 确定融合维度
        fusion_dim = max(text_hidden_size, audio_hidden_size)
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(fusion_dim * 2, 768),  # 融合特征：Mamba增强的text + audio
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Mamba风格初始化
        self._init_model()
    
    def _init_model(self):
        """使用Mamba风格的模型初始化"""
        # 对CHM层使用Mamba初始化
        for layer_idx, chm_layer in enumerate(self.chm_layers):
            _init_weights(chm_layer, n_layer=len(self.chm_layers), 
                         initializer_range=0.02, rescale_prenorm_residual=True)
    
    def extract_text_features(self, text_inputs, text_mask):
        """提取中文文本特征"""
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        pooler_output = raw_output["pooler_output"]
        return pooler_output
    
    def extract_audio_features(self, audio_inputs, audio_mask):
        """提取中文音频特征 - 适配Hubert模型"""
        audio_out = self.hubert_model(audio_inputs, audio_mask, output_attentions=True)
        hidden_states = audio_out.last_hidden_state
        
        # 获取有效长度（处理padding）- 适配Hubert
        valid_features = []
        for batch in range(hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0] != 0)
                    break
                except:
                    layer += 1
            
            if layer < 12:
                valid_feature = torch.mean(hidden_states[batch][:padding_idx], 0)
            else:
                valid_feature = torch.mean(hidden_states[batch], 0)
            
            valid_features.append(valid_feature)
        
        pooler_output = torch.stack(valid_features, 0).to(device)
        return pooler_output
    
    def forward(self, text_inputs, text_mask, audio_inputs, audio_mask):
        """SIMS数据集前向传播 - 使用完整Mamba CHM"""
        
        # === 特征提取阶段 ===
        # 文本特征提取
        text_features = self.extract_text_features(text_inputs, text_mask)
        
        # 音频特征提取
        audio_features = self.extract_audio_features(audio_inputs, audio_mask)
        
        # === 核心修改：Mamba CHM阶段 ===
        # 通过Mamba CHM层处理
        text_enhanced = text_features
        audio_enhanced = audio_features
        
        for chm_layer in self.chm_layers:
            text_enhanced, audio_enhanced = chm_layer(text_enhanced, audio_enhanced)
        
        # === 输出阶段 ===
        # 单模态输出（使用原始特征）
        T_output = self.T_output_layers(text_features)
        A_output = self.A_output_layers(audio_features)
        
        # 融合输出（使用Mamba增强的特征）
        fused_features = torch.cat((text_enhanced, audio_enhanced), dim=1)
        fused_output = self.fused_output_layers(fused_features)
        
        return {
            'T': T_output,
            'A': A_output, 
            'M': fused_output
        }


# 配置类
class MSAmbaSimsConfig:
    """MSAmba-MMML SIMS配置类"""
    def __init__(self, 
                 dropout=0.3,
                 chm_depth=1,
                 use_checkpoint=False,
                 text_model_path='/home/yyk/yyk09/models/chinese-roberta-wwm-ext',
                 audio_model_path='/home/yyk/yyk09/models/chinese-hubert-base',
                 # Mamba核心参数
                 mamba_d_state=16,
                 mamba_d_conv=4,
                 mamba_expand=2):
        self.dropout = dropout
        self.chm_depth = chm_depth
        self.use_checkpoint = use_checkpoint
        self.text_model_path = text_model_path
        self.audio_model_path = audio_model_path
        # Mamba参数
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand