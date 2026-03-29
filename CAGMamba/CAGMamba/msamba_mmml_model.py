# ====================================================================================
# 修改后的 msamba_mmml_model.py - 使用上下文构建真实序列
# ====================================================================================

"""
MSAmba-MMML集成模型 - 基于上下文的真实序列处理
使用main+context构建有意义的序列，而非伪造的单时间步
"""
import torch
from torch import nn
from transformers import RobertaModel, Data2VecAudioModel
from functools import partial
import sys
import os
from cgm_variants import get_cgm_variant

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
    print("✅ 成功导入完整MSAmba组件（Mamba实现）")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure msamba_blocks.py is in the same directory")
    raise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# CGM测试
try:
    CGM_VARIANTS_AVAILABLE = True
except ImportError:
    CGM_VARIANTS_AVAILABLE = False
    print("⚠️  cgm_variants.py未找到，使用默认CHM")

class ContextAwareMambaFeatureLevelCHM(nn.Module):
    """基于上下文的Mamba特征级别CHM - 构建真实序列而非伪序列"""
    
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
        
        print(f"✅ 初始化基于上下文的Mamba CHM，fusion_dim={fusion_dim}")
        
    def forward(self, text_features, audio_features, text_context_features=None, audio_context_features=None):
        """
        基于上下文的前向传播 - 构建真实序列
        
        Args:
            text_features: [B, text_dim] 主要文本特征
            audio_features: [B, audio_dim] 主要音频特征
            text_context_features: [B, text_dim] 文本上下文特征
            audio_context_features: [B, audio_dim] 音频上下文特征
        """
        batch_size = text_features.shape[0]
        
        # 维度对齐
        text_proj = self.text_proj(text_features)  # [B, fusion_dim]
        audio_proj = self.audio_proj(audio_features)  # [B, fusion_dim]
        
        # === 核心改进：构建真实序列而非伪序列 ===
        if text_context_features is not None and audio_context_features is not None:
            # 有上下文数据：构建真实的时序序列
            text_context_proj = self.text_proj(text_context_features)    # [B, fusion_dim]
            audio_context_proj = self.audio_proj(audio_context_features)  # [B, fusion_dim]
            
            # 构建有意义的序列：[context, main] 表示时序演化
            text_seq = torch.stack([text_context_proj, text_proj], dim=1)    # [B, 2, fusion_dim]
            audio_seq = torch.stack([audio_context_proj, audio_proj], dim=1)  # [B, 2, fusion_dim]
            
            sequence_info = "真实序列(context->main)"
        else:
            # 降级方案：生成多视角特征（仍比单一时间步好）
            print("⚠️  警告：缺少上下文数据，使用多视角特征生成")
            
            # 生成不同视角的特征表示
            text_enhanced = torch.tanh(text_proj)  # 非线性增强
            audio_enhanced = torch.tanh(audio_proj)
            
            text_seq = torch.stack([text_proj, text_enhanced], dim=1)    # [B, 2, fusion_dim]
            audio_seq = torch.stack([audio_proj, audio_enhanced], dim=1)  # [B, 2, fusion_dim]
            
            sequence_info = "多视角序列(original->enhanced)"
        
        # === Mamba跨模态融合 ===
        # 拼接文本和音频特征进行跨模态处理
        cross_modal_seq = torch.cat([audio_seq, text_seq], dim=-1)  # [B, 2, fusion_dim*2]
        
        # 归一化
        cross_modal_seq = self.norm_cross(cross_modal_seq)
        
        # Mamba处理跨模态序列 - 现在是真实的序列！
        cross_modal_enhanced = self.mamba_cross_modal(cross_modal_seq)  # [B, 2, fusion_dim*2]
        
        # 投影回单模态维度
        text_from_cross = self.proj_cross_to_text(cross_modal_enhanced)  # [B, 2, fusion_dim]
        audio_from_cross = self.proj_cross_to_audio(cross_modal_enhanced)  # [B, 2, fusion_dim]
        
        # === 单模态Mamba处理 ===
        # 纯文本序列处理
        text_seq_norm = self.norm_text(text_seq)
        text_enhanced = self.mamba_text(text_seq_norm)  # [B, 2, fusion_dim]
        
        # 纯音频序列处理  
        audio_seq_norm = self.norm_audio(audio_seq)
        audio_enhanced = self.mamba_audio(audio_seq_norm)  # [B, 2, fusion_dim]
        
        # === 门控融合 ===
        # 使用序列的最后一个时间步（main feature对应的位置）
        cross_info_for_text = torch.cat([text_from_cross[:, -1, :], audio_from_cross[:, -1, :]], dim=-1)
        cross_info_for_audio = torch.cat([audio_from_cross[:, -1, :], text_from_cross[:, -1, :]], dim=-1)
        
        text_gate_weight = self.text_gate(cross_info_for_text)   # [B, fusion_dim]
        audio_gate_weight = self.audio_gate(cross_info_for_audio)  # [B, fusion_dim]
        
        # 最终融合：单模态增强 + 门控的跨模态信息
        text_final = text_enhanced[:, -1, :] + text_gate_weight * text_from_cross[:, -1, :]
        audio_final = audio_enhanced[:, -1, :] + audio_gate_weight * audio_from_cross[:, -1, :]
        
        # 验证输出维度
        assert text_final.shape == (batch_size, self.fusion_dim), f"文本输出维度错误: {text_final.shape}"
        assert audio_final.shape == (batch_size, self.fusion_dim), f"音频输出维度错误: {audio_final.shape}"
        
        if batch_size == 1:  # 只在第一个batch打印调试信息
            print(f"🔄 Mamba处理完成 - {sequence_info}")
        
        return text_final, audio_final


class MSAmba_MMML_Context_Bimodal(nn.Module):
    """
    MSAmba-MMML双模态集成模型 - 基于上下文的真实序列处理版本
    """
    
    def __init__(self, config):
        super().__init__()
        
        # 配置参数（保持不变）
        self.config = config
        self.fusion_depth = getattr(config, 'chm_depth', 1)
        
        # 模型加载逻辑（保持不变）
        text_model_path = getattr(config, 'text_model_path', '/home/yyk/yyk09/models/roberta-large')
        audio_model_path = getattr(config, 'audio_model_path', '/home/yyk/yyk09/models/data2vec-audio-large-960h')
        
        def check_and_get_model_path(model_path, model_type="text"):
            import os
            
            if os.path.exists(model_path):
                print(f"✓ 找到本地{model_type}模型: {model_path}")
                return model_path
            else:
                print(f"⚠ 本地{model_type}模型不存在: {model_path}")
                
                if model_type == "text":
                    fallback_paths = [
                        '/home/yyk/yyk09/models/roberta-base',
                        'roberta-base',
                        'roberta-large'
                    ]
                else:  # audio
                    fallback_paths = [
                        '/home/yyk/yyk09/models/data2vec-audio-large',
                        '/home/yyk/yyk09/models/data2vec-audio-base',
                        'facebook/data2vec-audio-large',
                        'facebook/data2vec-audio-base'
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
        
        # 加载文本模型（保持不变）
        try:
            print(f"正在加载文本模型: {text_model_path}")
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            self.roberta_model = RobertaModel.from_pretrained(
                text_model_path,
                local_files_only=True
            )
            print("✓ 文本模型加载成功")
        except Exception as e:
            print(f"✗ 本地文本模型加载失败: {e}")
            print("尝试在线加载...")
            try:
                if 'TRANSFORMERS_OFFLINE' in os.environ:
                    del os.environ['TRANSFORMERS_OFFLINE']
                self.roberta_model = RobertaModel.from_pretrained('roberta-base')
                print("✓ 在线文本模型加载成功")
            except Exception as e2:
                raise RuntimeError(f"无法加载任何文本模型: {e2}")
        
        # 加载音频模型（保持不变）
        try:
            print(f"正在加载音频模型: {audio_model_path}")
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            self.data2vec_model = Data2VecAudioModel.from_pretrained(
                audio_model_path,
                local_files_only=True
            )
            print("✓ 音频模型加载成功")
        except Exception as e:
            print(f"✗ 本地音频模型加载失败: {e}")
            print("尝试在线加载...")
            try:
                if 'TRANSFORMERS_OFFLINE' in os.environ:
                    del os.environ['TRANSFORMERS_OFFLINE']
                try:
                    self.data2vec_model = Data2VecAudioModel.from_pretrained('facebook/data2vec-audio-large')
                    print("✓ 在线音频large模型加载成功")
                except:
                    self.data2vec_model = Data2VecAudioModel.from_pretrained('facebook/data2vec-audio-base')
                    print("✓ 在线音频base模型加载成功（large模型不可用）")
            except Exception as e2:
                raise RuntimeError(f"无法加载任何音频模型: {e2}")
        
        # 清理环境变量
        if 'TRANSFORMERS_OFFLINE' in os.environ:
            del os.environ['TRANSFORMERS_OFFLINE']
        if 'HF_DATASETS_OFFLINE' in os.environ:
            del os.environ['HF_DATASETS_OFFLINE']
        
        # 获取实际的模型维度
        text_hidden_size = self.roberta_model.config.hidden_size
        audio_hidden_size = self.data2vec_model.config.hidden_size
        
        print(f"文本模型隐藏维度: {text_hidden_size}")
        print(f"音频模型隐藏维度: {audio_hidden_size}")
        print(f"音频模型类型: {'Large' if audio_hidden_size > 768 else 'Base'}")
        
        # === 核心修改：使用基于上下文的Mamba CHM层 ===
        # Mamba配置
        self.mamba_config = {
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dt_rank': 'auto',
            'device': device,
            'dtype': torch.float32
        }
        
        
        #CGM测试
        # 检查是否需要使用实验性的CGM变体
        cgm_variant = getattr(config, 'cgm_variant', 'full')
        print(f"🔍 [DEBUG] cgm_variant='{cgm_variant}', CGM_VARIANTS_AVAILABLE={CGM_VARIANTS_AVAILABLE}")

        if cgm_variant != 'full' and CGM_VARIANTS_AVAILABLE:
            # 使用实验性变体（用于消融实验）
            print(f"🧪 [实验模式] 使用CGM变体: {cgm_variant.upper()}")
            
            try:
                # 根据不同变体传递不同的参数
                if cgm_variant == 'transformer':
                    # Transformer不需要Mamba参数
                    self.chm_layers = nn.ModuleList([
                        get_cgm_variant(
                            variant_name=cgm_variant,
                            d_model=max(text_hidden_size, audio_hidden_size),
                            nhead=16,
                            num_layers=2,
                            dropout=0.1
                        )
                        for _ in range(self.fusion_depth)
                    ])
                else:
                    # full, unidirectional_ssm, vision_mamba需要Mamba参数
                    self.chm_layers = nn.ModuleList([
                        get_cgm_variant(
                            variant_name=cgm_variant,
                            d_model=max(text_hidden_size, audio_hidden_size),
                            d_state=self.mamba_config['d_state'],
                            d_conv=self.mamba_config['d_conv'],
                            expand=self.mamba_config['expand'],
                            dropout=0.1
                        )
                        for _ in range(self.fusion_depth)
                    ])
                print(f"✅ 成功初始化 {cgm_variant} 变体CHM层，共{self.fusion_depth}层")
                
            except Exception as e:
                print(f"❌ 初始化 {cgm_variant} 变体失败: {e}")
                print(f"⚠️  回退到默认CHM")
                cgm_variant = 'full'
        
        # 使用默认CHM（cgm_variant=='full' 或变体初始化失败）
        if cgm_variant == 'full':
            print(f"✅ 使用默认ContextAwareMambaFeatureLevelCHM")
            
            # 创建基于上下文的Mamba CHM层
            self.chm_layers = nn.ModuleList([
                ContextAwareMambaFeatureLevelCHM(
                    text_hidden_size, 
                    audio_hidden_size, 
                    mamba_config=self.mamba_config
                ) 
                for _ in range(self.fusion_depth)
            ])
        
        print(f"✅ 使用基于上下文的Mamba CHM层，共{self.fusion_depth}层")
        print(f"✅ 上下文序列构建：context->main (序列长度=2)")
        print(f"✅ Mamba配置: d_state={self.mamba_config['d_state']}, d_conv={self.mamba_config['d_conv']}")
        print(f"✅ Mamba可用性: {MAMBA_AVAILABLE}")
        
        # 输出层 - 保持与MMML兼容的接口（不变）
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(text_hidden_size * 2, 1)  # text + text_context
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(audio_hidden_size * 2, 1)  # audio + audio_context
        )
        
        # 确定融合维度
        fusion_dim = max(text_hidden_size, audio_hidden_size)
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(fusion_dim * 4, 768),  # 融合特征：text + audio + text_context + audio_context（Mamba增强）
            nn.ReLU(),
            nn.Linear(768, 1)
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
        """提取文本特征 - 与原MMML保持一致（不变）"""
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        hidden_states = raw_output.last_hidden_state
        pooler_output = raw_output["pooler_output"]
        return hidden_states, pooler_output
    
    def extract_audio_features(self, audio_inputs, audio_mask):
        """提取音频特征 - 与原MMML保持一致（不变）"""
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        hidden_states = audio_out.last_hidden_state
        
        # 获取有效长度（处理padding）- 与原MMML保持一致
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
        return hidden_states, pooler_output
    
    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask,
                audio_inputs, audio_mask, audio_context_inputs, audio_context_mask):
        """双模态前向传播 - 基于上下文的Mamba CHM处理"""
        
        # === 特征提取阶段（保留MMML逻辑，不变）===
        # 文本特征提取
        _, text_pooler = self.extract_text_features(text_inputs, text_mask)
        _, text_context_pooler = self.extract_text_features(
            text_context_inputs, text_context_mask
        )
        
        # 音频特征提取
        _, audio_pooler = self.extract_audio_features(audio_inputs, audio_mask)
        _, audio_context_pooler = self.extract_audio_features(
            audio_context_inputs, audio_context_mask
        )
        
        # === 核心修改：基于上下文的Mamba CHM阶段 ===
        # 对main features进行CHM处理（传入context）
        text_enhanced = text_pooler
        audio_enhanced = audio_pooler
        
        for chm_layer in self.chm_layers:
            text_enhanced, audio_enhanced = chm_layer(
                text_enhanced, audio_enhanced,
                text_context_features=text_context_pooler,  # 关键：传入上下文
                audio_context_features=audio_context_pooler
            )
        
        # 对context features也进行CHM处理（以main为context）
        text_context_enhanced = text_context_pooler  
        audio_context_enhanced = audio_context_pooler
        
        for chm_layer in self.chm_layers:
            text_context_enhanced, audio_context_enhanced = chm_layer(
                text_context_enhanced, audio_context_enhanced,
                text_context_features=text_pooler,      # 互为上下文
                audio_context_features=audio_pooler
            )
        
        # === 输出阶段（保持不变）===
        # 单模态输出（使用原始特征 + context）
        T_features = torch.cat((text_pooler, text_context_pooler), dim=1)
        A_features = torch.cat((audio_pooler, audio_context_pooler), dim=1)
        T_output = self.T_output_layers(T_features)
        A_output = self.A_output_layers(A_features)
        
        # 融合输出（使用Mamba增强的特征）
        fused_features = torch.cat((
            text_enhanced,           # Mamba增强的文本特征
            audio_enhanced,          # Mamba增强的音频特征  
            text_context_enhanced,   # Mamba增强的文本context特征
            audio_context_enhanced   # Mamba增强的音频context特征
        ), dim=1)
        fused_output = self.fused_output_layers(fused_features)
        
        return {
            'T': T_output,
            'A': A_output, 
            'M': fused_output
        }


# 配置类保持不变
class MSAmbaConfig:
    """MSAmba-MMML配置类"""
    def __init__(self, 
                 dropout=0.3,
                 chm_depth=1,
                 use_checkpoint=False,
                 text_model_path='/home/yyk/yyk09/models/roberta-large',
                 audio_model_path='/home/yyk/yyk09/models/data2vec-audio-large-960h'):
        self.dropout = dropout
        self.chm_depth = chm_depth
        self.use_checkpoint = use_checkpoint
        self.text_model_path = text_model_path
        self.audio_model_path = audio_model_path