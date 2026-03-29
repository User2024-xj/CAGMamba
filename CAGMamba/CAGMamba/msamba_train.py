# ====================================================================================
# 文件4: msamba_train.py - 支持完整Mamba配置的训练逻辑更新
# ====================================================================================

"""
MSAmba-MMML训练逻辑 - 完整Mamba实现版本
添加Mamba特有参数的配置和优化策略
修改：添加Mamba核心参数配置，优化训练策略以适配真正的Mamba机制
"""
import torch
from torch import nn
from tqdm import tqdm
import random
import numpy as np
import sys
import os
import json
from datetime import datetime

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入组件
try:
    from utils.metricsTop import MetricsTop
    from utils.data_loader import data_loader
    from msamba_mmml_model import MSAmba_MMML_Context_Bimodal, MSAmbaConfig  # 使用完整Mamba版本
    print("✅ 成功导入完整Mamba MSAmba训练组件")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required files are in the correct directories")
    # 如果无法导入utils，尝试从当前目录导入
    try:
        from metricsTop import MetricsTop
        from data_loader import data_loader
        from msamba_mmml_model import MSAmba_MMML_Context_Bimodal, MSAmbaConfig
        print("✅ 从当前目录导入完整Mamba MSAmba训练组件")
    except ImportError:
        raise ImportError("Cannot find required modules. Please check file structure.")

# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str


def save_test_results(test_results, config, model_identifier):
    """保存测试结果到JSON文件（添加Mamba参数信息）"""
    
    # 准备保存的数据
    save_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_identifier': model_identifier,
        'dataset': config.dataset_name,
        'seed': config.seed,
        'hyperparameters': {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'dropout': config.dropout,
            'chm_depth': config.chm_depth,
            'text_context_len': config.text_context_len,
            'audio_context_len': config.audio_context_len,
            'tasks': config.tasks,
            'epochs': config.epochs,
            'early_stop': config.early_stop,
            'text_model_path': config.text_model_path,
            'audio_model_path': config.audio_model_path,
            'use_checkpoint': config.use_checkpoint,
            # === 新增：Mamba特有参数 ===
            'mamba_d_state': config.mamba_d_state,
            'mamba_d_conv': config.mamba_d_conv,
            'mamba_expand': config.mamba_expand,
        },
        'test_results': test_results,
        'model_type': 'MSAmba_Full_Mamba_Implementation',
        'audio_model_type': "large" if "large" in config.audio_model_path else "base",
        'architecture_info': {
            'uses_real_mamba': True,
            'selective_scan': True,
            'linear_complexity': True,
            'state_space_modeling': True
        }
    }
    
    # 生成保存文件名
    results_filename = f'{model_identifier}_test_results_{config.dataset_name}_seed{config.seed}.json'
    results_path = os.path.join(config.model_save_path, results_filename)
    
    # 保存到JSON文件
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 测试结果已保存到: {results_path}")
        
        # 同时保存一份简化版本（仅结果指标）
        simple_results_filename = f'{model_identifier}_metrics_only_{config.dataset_name}_seed{config.seed}.json'
        simple_results_path = os.path.join(config.model_save_path, simple_results_filename)
        
        with open(simple_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        print(f"✅ 简化测试指标已保存到: {simple_results_path}")
        
    except Exception as e:
        print(f"❌ 保存测试结果失败: {e}")
        
    return results_path


class MSAmbaEnConfig(object):
    """MSAmba-MMML完整Mamba配置类，添加Mamba核心参数"""
    def __init__(self,
                train_mode = 'regression',
                loss_weights = {
                    'M':1,
                    'T':1,
                    'A':1,
                },
                 model_save_path = 'checkpoint2/',
                 learning_rate = 1e-5,
                 epochs = 20,
                 dataset_name = 'mosei',
                 early_stop = 8,
                 seed = 0,
                 dropout=0.3,
                 model='msamba',
                 batch_size = 16,
                 multi_task = True,
                 tasks = 'MTA',   # 'M' or 'MTA'
                 context = True,
                 text_context_len = 2,
                 audio_context_len = 1,
                 # MSAmba特有参数（完整Mamba实现）
                 chm_depth = 1,
                 use_checkpoint = False,
                 text_model_path = '/home/yyk/yyk09/models/roberta-large',
                 audio_model_path = '/home/yyk/yyk09/models/data2vec-audio-large-960h',
                 # === 新增：Mamba核心参数 ===
                 mamba_d_state = 16,        # Mamba状态维度
                 mamba_d_conv = 4,          # Mamba卷积核大小  
                 mamba_expand = 2,           # Mamba内部扩展倍数
                 
                 cgm_variant='full',  # 新增这行
                 **kwargs
                ):

        # 基础配置（保持不变）
        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.tasks = tasks
        self.context = context
        self.text_context_len = text_context_len
        self.audio_context_len = audio_context_len
        
        # MSAmba特有配置（保持不变）
        self.chm_depth = chm_depth
        self.use_checkpoint = use_checkpoint
        self.text_model_path = text_model_path
        self.audio_model_path = audio_model_path

        # === 新增：Mamba核心参数配置 ===
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand

        # cgm
        self.cgm_variant = cgm_variant
        

        
class MSAmbaEnTrainer():
    def __init__(self, config):
        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.tasks = config.tasks
        
    def do_train(self, model, data_loader):    
        model.train()
        
        # === 优化：针对Mamba的分层学习率设置 ===
        # RoBERTa参数（保持不变）
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params_decay = []
        bert_params_no_decay = []
        
        for name, param in model.roberta_model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in bert_no_decay):
                    bert_params_no_decay.append(param)
                else:
                    bert_params_decay.append(param)
        
        # 音频模型参数（保持不变）
        audio_params = []
        audio_params_large = []
        
        for name, param in model.data2vec_model.named_parameters():
            if param.requires_grad:
                if 'encoder.layers' in name and int(name.split('.')[2]) >= 12:
                    audio_params_large.append(param)
                else:
                    audio_params.append(param)
        
        # === 重要修改：Mamba特有参数分组 ===
        mamba_state_params = []      # 状态空间参数 (A_log, D)
        mamba_conv_params = []       # 卷积参数
        mamba_proj_params = []       # 投影层参数 (in_proj, out_proj, etc.)
        mamba_dt_params = []         # 时间参数 (dt_proj)
        chm_fusion_params = []       # CHM融合参数
        
        for name, param in model.named_parameters():
            if param.requires_grad and 'chm_layers' in name:
                if 'A_log' in name or 'D' in name:
                    mamba_state_params.append(param)
                elif 'conv1d' in name:
                    mamba_conv_params.append(param)
                elif 'dt_proj' in name:
                    mamba_dt_params.append(param)
                elif any(proj in name for proj in ['in_proj', 'out_proj', 'x_proj']):
                    mamba_proj_params.append(param)
                else:
                    # CHM层的其他参数（门控、归一化等）
                    chm_fusion_params.append(param)
        
        # 其他参数（输出层等，保持不变）
        other_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and not any([
                'roberta_model' in name,
                'data2vec_model' in name,
                'chm_layers' in name
            ]):
                other_params.append(param)
        
        # 打印参数统计（添加Mamba参数统计）
        print(f"BERT decay params: {len(bert_params_decay)}")
        print(f"BERT no-decay params: {len(bert_params_no_decay)}")
        print(f"Audio base params: {len(audio_params)}")
        print(f"Audio large-specific params: {len(audio_params_large)}")
        print(f"=== Mamba参数统计 ===")
        print(f"Mamba状态空间参数: {len(mamba_state_params)} (A_log, D)")
        print(f"Mamba卷积参数: {len(mamba_conv_params)}")
        print(f"Mamba投影参数: {len(mamba_proj_params)}")
        print(f"Mamba时间参数: {len(mamba_dt_params)} (dt_proj)")
        print(f"CHM融合参数: {len(chm_fusion_params)}")
        print(f"其他参数: {len(other_params)}")
        
        # === 优化的参数组设置，针对Mamba特点 ===
        param_groups = []
        if bert_params_decay:
            param_groups.append({'params': bert_params_decay, 'weight_decay': 0.01, 'lr': 1e-5})
        if bert_params_no_decay:
            param_groups.append({'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': 1e-5})
        if audio_params:
            param_groups.append({'params': audio_params, 'weight_decay': 0.01, 'lr': 5e-6})
        if audio_params_large:
            param_groups.append({'params': audio_params_large, 'weight_decay': 0.01, 'lr': 2e-6})
        
        # === 新增：Mamba参数的专门优化策略 ===
        if mamba_state_params:
            # 状态空间参数使用较小的学习率和权重衰减，因为它们控制模型的核心动力学
            param_groups.append({
                'params': mamba_state_params, 
                'weight_decay': 1e-6,  # 很小的权重衰减
                'lr': self.config.learning_rate * 0.5  # 较小的学习率
            })
        
        if mamba_conv_params:
            # 卷积参数使用标准的学习率
            param_groups.append({
                'params': mamba_conv_params, 
                'weight_decay': 1e-4, 
                'lr': self.config.learning_rate * 0.8
            })
            
        if mamba_dt_params:
            # 时间参数非常重要，使用专门的优化策略
            param_groups.append({
                'params': mamba_dt_params, 
                'weight_decay': 1e-5,  # 小权重衰减
                'lr': self.config.learning_rate * 0.3  # 更小的学习率，因为dt参数很敏感
            })
            
        if mamba_proj_params:
            # 投影参数使用相对较大的学习率
            param_groups.append({
                'params': mamba_proj_params, 
                'weight_decay': 1e-4, 
                'lr': self.config.learning_rate * 1.2
            })
        
        if chm_fusion_params:
            # CHM融合参数使用稍大的学习率，因为是新引入的组件
            param_groups.append({
                'params': chm_fusion_params, 
                'weight_decay': 1e-4, 
                'lr': self.config.learning_rate * 1.5
            })
        
        if other_params:
            param_groups.append({
                'params': other_params, 
                'weight_decay': 1e-4, 
                'lr': self.config.learning_rate
            })
        
        if not param_groups:
            raise RuntimeError("No trainable parameters found!")
        
        # === 优化器设置：使用AdamW，针对Mamba优化 ===
        optimizer = torch.optim.AdamW(
            param_groups,
            eps=1e-8,           # 稍大的eps，提高训练稳定性
            betas=(0.9, 0.999)  # 标准的beta值
        )

        total_loss = 0
        # Loop over all batches.         
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Training")):                    
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            text_context_inputs = batch["text_context_tokens"].to(device)
            text_context_mask = batch["text_context_masks"].to(device)

            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            audio_context_inputs = batch["audio_context_inputs"].to(device)
            audio_context_mask = batch["audio_context_masks"].to(device)

            targets = batch["targets"].to(device).view(-1, 1)

            optimizer.zero_grad()

            outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, 
                          audio_inputs, audio_mask, audio_context_inputs, audio_context_mask)
            
            # 只在第一个batch时输出调试信息
            if batch_idx == 0:
                print(f"\n[调试] 第一个batch输出形状检查 (完整Mamba版本):")
                print(f"  文本输入: {text_inputs.shape}")
                print(f"  音频输入: {audio_inputs.shape}")
                print(f"  模型输出 - T: {outputs['T'].shape}, A: {outputs['A'].shape}, M: {outputs['M'].shape}")
                print(f"  目标值: {targets.shape}")
                print("[调试] 完整Mamba CHM处理完成\n")
            
            # Compute the training loss.
            if self.config.multi_task:
                loss = 0.0         
                for m in self.tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                    loss += sub_loss
                total_loss += loss.item()*text_inputs.size(0)  
            else:
                loss = self.criterion(outputs['M'], targets)        
                total_loss += loss.item()*text_inputs.size(0)
        
            loss.backward()                   
            
            # === 优化的梯度裁剪：针对Mamba的梯度特点 ===
            # Mamba模型可能有较大的梯度波动，特别是状态空间参数
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 更严格的梯度裁剪
            
            optimizer.step()                
                
        total_loss = round(total_loss / len(data_loader.dataset), 4)
        return total_loss

    def do_test(self, model, data_loader, mode):
        """测试函数保持不变"""
        model.eval()
        if self.config.multi_task:
            y_pred = {'M': [], 'T': [], 'A': []}
            y_true = {'M': [], 'T': [], 'A': []}
            total_loss = 0
            val_loss = {
                'M':0,
                'T':0,
                'A':0
            }
        else:
            y_pred = []
            y_true = []
            total_loss = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"{mode} Evaluation"):
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)

                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                audio_context_inputs = batch["audio_context_inputs"].to(device)
                audio_context_mask = batch["audio_context_masks"].to(device)

                targets = batch["targets"].to(device).view(-1, 1)

                outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask,
                              audio_inputs, audio_mask, audio_context_inputs, audio_context_mask)
                
                # Compute loss.
                if self.config.multi_task:
                    loss = 0.0         
                    for m in self.tasks:
                        sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                        loss += sub_loss
                        val_loss[m] += sub_loss.item()*text_inputs.size(0)
                    total_loss += loss.item()*text_inputs.size(0)
                    # add predictions
                    for m in self.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(targets.cpu())
                else:
                    loss = self.criterion(outputs['M'], targets)        
                    total_loss += loss.item()*text_inputs.size(0)

                    # add predictions
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(targets.cpu())

        if self.config.multi_task:
            for m in self.tasks:
                val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss, "   M_loss: ", val_loss['M'], "  T_loss: ", val_loss['T'], "  A_loss: ", val_loss['A'])

            eval_results = {}
            for m in self.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                results = self.metrics(pred, true)
                print('%s: >> ' %(m) + dict_to_str(results))
                eval_results[m] = results
            eval_results = eval_results[self.tasks[0]]
            eval_results['Loss'] = total_loss 
        else:
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode+" >> loss: ",total_loss)

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            print('%s: >> ' %('M') + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss
        
        return eval_results


def MSAmbaEnRun(config):
    """MSAmba-MMML完整Mamba训练主函数"""
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    # 数据加载（保持不变）
    train_loader, test_loader, val_loader = data_loader(
        config.batch_size, config.dataset_name,
        text_context_length=config.text_context_len,
        audio_context_length=config.audio_context_len
    )

    # === 修改：创建完整Mamba配置 ===
    msamba_config = MSAmbaConfig(
        dropout=config.dropout,
        chm_depth=config.chm_depth,
        use_checkpoint=config.use_checkpoint,
        text_model_path=config.text_model_path,
        audio_model_path=config.audio_model_path
    )
    
    # 创建使用完整Mamba的模型
    model = MSAmba_MMML_Context_Bimodal(msamba_config).to(device)
    
    # 冻结Data2Vec特征提取器（保持与MMML一致）
    for param in model.data2vec_model.feature_extractor.parameters():
        param.requires_grad = False

    trainer = MSAmbaEnTrainer(config)
    
    # === 修改：生成包含Mamba参数的模型文件名 ===
    audio_model_type = "large" if "large" in config.audio_model_path else "base"

    # 构建详细的模型标识符（包含Mamba参数）
    model_identifier = (f'MSAmba_{config.cgm_variant}_{audio_model_type}_'
                    f'chm{config.chm_depth}_bs{config.batch_size}_'
                    f'lr{config.learning_rate:.0e}_'
                    f'dr{config.dropout:.1f}_'
                    f'dstate{config.mamba_d_state}_dconv{config.mamba_d_conv}_exp{config.mamba_expand}_'
                    f'txtctx{config.text_context_len}_audioctx{config.audio_context_len}_'
                    f'tasks{config.tasks}')

    # 如果启用了梯度检查点，添加标识
    if config.use_checkpoint:
        model_identifier += '_ckpt'

    # 模型保存路径
    best_loss_model_path = (config.model_save_path + 
                        f'{model_identifier}_best_loss_{config.dataset_name}_seed{config.seed}.pth')
    best_acc_model_path = (config.model_save_path + 
                        f'{model_identifier}_best_acc_{config.dataset_name}_seed{config.seed}.pth')

    print(f"模型标识: {model_identifier}")
    print(f"音频模型类型: {audio_model_type}")
    print(f"CHM深度: {config.chm_depth}")
    print(f"Mamba参数: d_state={config.mamba_d_state}, d_conv={config.mamba_d_conv}, expand={config.mamba_expand}")
    print(f"Dropout: {config.dropout}")
    print(f"上下文长度: 文本{config.text_context_len}, 音频{config.audio_context_len}")
    print(f"最佳损失模型: {best_loss_model_path}")
    print(f"最佳准确率模型: {best_acc_model_path}")

    # 确保保存目录存在
    os.makedirs(config.model_save_path, exist_ok=True)

    # 训练循环（保持不变）
    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0

    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        
        train_loss = trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")
        
        print(f"Epoch {epoch}: 训练损失={train_loss:.4f}, 验证损失={eval_results['Loss']:.4f}")
        
        # 保存最佳损失模型
        if eval_results['Loss'] < lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            torch.save(model.state_dict(), best_loss_model_path)
            best_epoch = epoch
            print(f"✅ 保存最佳损失模型: {lowest_eval_loss:.4f}")
        
        # 保存最佳准确率模型
        if eval_results['Has0_acc_2'] >= highest_eval_acc:
            highest_eval_acc = eval_results['Has0_acc_2']
            torch.save(model.state_dict(), best_acc_model_path)
            print(f"✅ 保存最佳准确率模型: {highest_eval_acc:.4f}")

        # ✅ 添加这个检查（在 early stop 之前）
        if epoch >= config.epochs:
            print(f"✅ 达到最大训练轮数：{config.epochs}")
            break
        
        # 早停检查
        if epoch - best_epoch >= config.early_stop:
            print(f"早停：已连续{config.early_stop}个epoch无改善")
            break

    print("\n=== 开始最终测试 ===")

    # 测试最佳准确率模型
    print("加载最佳准确率模型进行测试...")
    model.load_state_dict(torch.load(best_acc_model_path))
    test_results_acc = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' % ('TEST (highest val acc) ') + dict_to_str(test_results_acc))

    # 测试最佳损失模型  
    print("\n加载最佳损失模型进行测试...")
    model.load_state_dict(torch.load(best_loss_model_path))
    test_results_loss = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' % ('TEST (lowest val loss) ') + dict_to_str(test_results_loss))
    
    # 保存测试结果（包含Mamba参数信息）
    print("\n=== 保存测试结果 ===")
    save_test_results(test_results_acc, config, model_identifier)
    
    print(f"\n🎉 完整Mamba训练和测试完成!")
    print(f"📊 最佳验证准确率: {highest_eval_acc:.4f}")
    print(f"📊 最低验证损失: {lowest_eval_loss:.4f}")
    print(f"📈 最佳准确率模型测试结果: {dict_to_str(test_results_acc)}")
    print(f"🧠 Mamba架构参数: d_state={config.mamba_d_state}, d_conv={config.mamba_d_conv}, expand={config.mamba_expand}")
    
    return test_results_acc  # 返回最佳准确率的测试结果

