# ====================================================================================
# 文件3: main_msamba.py - 更新参数描述以反映完整Mamba实现
# ====================================================================================

"""
MSAmba-MMML主函数 - 完整Mamba实现版本
使用真正的Mamba机制进行跨模态融合，而非简化的MultiheadAttention
修改：更新参数描述，反映完整Mamba的使用
"""
import argparse
import sys
import os
from distutils.util import strtobool

# 设置环境变量，减少不必要的警告
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入组件
try:
    from utils.en_train import EnConfig, EnRun
    from utils.ch_train import ChConfig, ChRun
    from msamba_train import MSAmbaEnConfig, MSAmbaEnRun  # 使用完整Mamba版本
    print("✅ 成功导入完整Mamba MSAmba训练组件")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required files are in the correct directories")
    # 如果无法导入utils，尝试从当前目录导入
    try:
        from en_train import EnConfig, EnRun
        from ch_train import ChConfig, ChRun
        from msamba_train import MSAmbaEnConfig, MSAmbaEnRun
        print("✅ 从当前目录导入完整Mamba MSAmba训练组件")
    except ImportError:
        print("Warning: Some modules not found. Only MSAmba functionality will be available.")
        from msamba_train import MSAmbaEnConfig, MSAmbaEnRun
        EnConfig = EnRun = ChConfig = ChRun = None

def main(args):
    if args.dataset != 'sims':
        if args.model == 'msamba':
            # 使用完整Mamba MSAmba-MMML模型
            print("=" * 60)
            print("🚀 启动完整Mamba MSAmba-MMML训练")
            print("=" * 60)
            print("📊 架构特点:")
            print("  ✓ 真正的Mamba机制: 线性复杂度O(L)状态空间模型")
            print("  ✓ Selective Scan: 动态选择性扫描算法")
            print("  ✓ CHM: 基于Mamba的跨模态融合")
            print("  ✓ 支持长序列: 相比Transformer的O(L²)复杂度优势")
            print("  ✓ 状态空间建模: 时序信息的有效捕获")
            print("  ✓ 特征级优化: 针对pooled features的Mamba处理")
            print("=" * 60)
            
            MSAmbaEnRun(MSAmbaEnConfig(
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed, 
                model=args.model, 
                tasks=args.tasks,
                dataset_name=args.dataset,
                context=args.context, 
                text_context_len=args.text_context_len, 
                audio_context_len=args.audio_context_len,
                # MSAmba特有参数（完整Mamba实现）
                chm_depth=args.chm_depth,
                use_checkpoint=args.use_checkpoint,
                # Mamba特有配置
                mamba_d_state=args.mamba_d_state,
                mamba_d_conv=args.mamba_d_conv,
                mamba_expand=args.mamba_expand,
                # 模型路径
                text_model_path=args.text_model_path,
                audio_model_path=args.audio_model_path,
                cgm_variant=args.cgm_variant,
                # 训练控制参数
                epochs=args.epochs,
                early_stop=args.early_stop,
            ))
        else:
            # 使用原MMML模型
            print("=" * 60)
            print("🔄 启动原MMML训练")
            print("=" * 60)
            if EnRun is None:
                raise ImportError("Original MMML modules not found. Please ensure utils/ directory exists.")
            EnRun(EnConfig(
                batch_size=args.batch_size,
                learning_rate=args.lr,
                seed=args.seed, 
                model=args.model, 
                tasks=args.tasks,
                cme_version=args.cme_version, 
                dataset_name=args.dataset,
                num_hidden_layers=args.num_hidden_layers,
                context=args.context, 
                text_context_len=args.text_context_len, 
                audio_context_len=args.audio_context_len
            ))
    else:
        # SIMS数据集暂时使用原有逻辑
        print("=" * 60)
        print("🇨🇳 启动SIMS数据集训练")
        print("=" * 60)
        if ChRun is None:
            raise ImportError("Chinese MMML modules not found. Please ensure utils/ directory exists.")
        ChRun(ChConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed, 
            model=args.model, 
            tasks=args.tasks,
            cme_version=args.cme_version, 
            num_hidden_layers=args.num_hidden_layers
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSAmba-MMML: 多模态情感分析训练脚本（完整Mamba实现版本）')
    
    # 基础参数
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-6, help='学习率，推荐: mosi/mosei 5e-6, sims 1e-5')
    parser.add_argument('--model', type=str, default='msamba', 
                       choices=['cc', 'cme', 'msamba'],
                       help='模型类型: cc(concatenate), cme(cross-modality encoder), msamba(完整Mamba实现)')
    parser.add_argument('--dataset', type=str, default='mosi', 
                       choices=['mosi', 'mosei', 'sims'],
                       help='数据集名称')
    parser.add_argument('--tasks', type=str, default='MTA', 
                       help='训练任务: M(多模态), T(文本), A(音频)')
    
    # 原MMML参数（兼容性）
    parser.add_argument('--cme_version', type=str, default='v1', help='CME模型版本')
    parser.add_argument('--num_hidden_layers', type=int, default=5, help='跨模态编码器层数')
    
    # Context参数
    parser.add_argument('--context', default=True, help='是否使用上下文', 
                       dest='context', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--text_context_len', type=int, default=2, help='文本上下文长度')
    parser.add_argument('--audio_context_len', type=int, default=1, help='音频上下文长度')
    
    # MSAmba特有参数（完整Mamba实现）
    parser.add_argument('--chm_depth', type=int, default=1, help='CHM层深度（基于真正的Mamba机制）')
    parser.add_argument('--use_checkpoint', default=False, help='使用梯度检查点节省内存', 
                       dest='use_checkpoint', type=lambda x: bool(strtobool(x)))
    
    # === 新增：Mamba核心参数 ===
    parser.add_argument('--mamba_d_state', type=int, default=16, 
                       help='Mamba状态维度，控制状态空间模型的表达能力')
    parser.add_argument('--mamba_d_conv', type=int, default=4, 
                       help='Mamba卷积核大小，用于局部特征提取')
    parser.add_argument('--mamba_expand', type=int, default=2, 
                       help='Mamba内部维度扩展倍数，影响模型容量')
    
    # 模型路径参数
    parser.add_argument('--text_model_path', type=str, 
                       default='/home/yyk/yyk09/models/roberta-large', 
                       help='文本模型路径')
    parser.add_argument('--audio_model_path', type=str, 
                       default='/home/yyk/yyk09/models/data2vec-audio-large-960h', 
                       help='音频模型路径（推荐large版本以匹配Mamba能力）')
    
    parser.add_argument('--cgm_variant', type=str, default='full',
                   choices=['full', 'transformer', 'unidirectional_ssm', 'vision_mamba'],
                   help='CGM变体：full(完整), transformer, unidirectional_ssm, vision_mamba')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--early_stop', type=int, default=8, help='早停轮数')
    
    args = parser.parse_args()
    
    # 输出配置信息
    print("=" * 60)
    print("🔧 MSAmba-MMML 完整Mamba配置")
    print("=" * 60)
    print(f"📋 基础配置:")
    print(f"   模型类型: {args.model}")
    print(f"   数据集: {args.dataset}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   学习率: {args.lr}")
    print(f"   随机种子: {args.seed}")
    print(f"   训练任务: {args.tasks}")
    print("")
    print(f"📁 模型路径:")
    print(f"   文本模型: {args.text_model_path}")
    print(f"   音频模型: {args.audio_model_path}")
    
    if args.model == 'msamba':
        print("")
        print(f"🧠 完整Mamba参数:")
        print(f"   CHM深度: {args.chm_depth} (完整Mamba CHM层)")
        print(f"   状态维度: {args.mamba_d_state} (状态空间容量)")
        print(f"   卷积核大小: {args.mamba_d_conv} (局部特征提取)")
        print(f"   内部扩展: {args.mamba_expand}x (模型容量)")
        print(f"   梯度检查点: {args.use_checkpoint}")
        print("")
        print(f"✨ 完整Mamba架构优势:")
        print(f"   ✓ 线性复杂度: O(L) vs Transformer的O(L²)")
        print(f"   ✓ Selective Scan: 动态选择机制，有效处理长序列")
        print(f"   ✓ 状态空间建模: 捕获时序依赖和长程关系")
        print(f"   ✓ 内存高效: 相比标准注意力机制显著节省内存")
        print(f"   ✓ 真正的CHM: 基于Mamba的跨模态层次建模")
        print(f"   ✓ 特征级优化: 针对特征融合场景优化的实现")
    
    print("=" * 60)
    
    # 开始训练
    try:
        main(args)
        print("\n🎉 训练完成！")
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)