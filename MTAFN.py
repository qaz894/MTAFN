""" 多任务注意力融合网络 (Multi-Task Attention Fusion Network, MTAFN) """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

class WaveletTransform(nn.Module):
    """小波变换层 - 时频域特征提取"""
    
    def __init__(self, in_channels: int, wavelet_type: str = 'db4'):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet_type = wavelet_type
        self.h = nn.Parameter(torch.tensor([
            -0.010597401784997, -0.032883011666983, 0.030841381835987,
            0.187034811718881, 0.027983769416984, -0.630880767929590,
            0.714846570552542, -0.230377813308896
        ], dtype=torch.float32), requires_grad=False)
        
        self.g = nn.Parameter(torch.tensor([
            -0.230377813308896, -0.714846570552542, -0.630880767929590,
            -0.027983769416984, 0.187034811718881, -0.030841381835987,
            -0.032883011666983, 0.010597401784997
        ], dtype=torch.float32), requires_grad=False)
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, length)
        Returns:
            approximation: (batch, channels, length//2) - 低频成分
            detail: (batch, channels, length//2) - 高频成分
        """
        batch_size, channels, length = x.shape
        device = x.device
        
        h_filter = self.h.to(device).flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 8)
        g_filter = self.g.to(device).flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 8)
        
        approx_list = []
        detail_list = []
        
        for c in range(channels):
            x_c = x[:, c, :]  # (batch, length)
            
            x_c = x_c.unsqueeze(1)  # (batch, 1, length)
            
            approx = F.conv1d(x_c, h_filter, stride=2, padding=3)

            detail = F.conv1d(x_c, g_filter, stride=2, padding=3)
            
            approx_list.append(approx.squeeze(1))
            detail_list.append(detail.squeeze(1))
        
        approximation = torch.stack(approx_list, dim=1)  # (batch, channels, length//2)
        detail = torch.stack(detail_list, dim=1)  # (batch, channels, length//2)
        
        return approximation, detail

class MultiScaleAttention(nn.Module):
    """多尺度自适应注意力机制"""
    
    def __init__(self, in_channels: int, scales: List[int] = [1, 3, 5, 7]):
        super().__init__()
        self.scales = scales
        self.in_channels = in_channels
        
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels // len(scales), 
                     kernel_size=scale, padding=scale//2)
            for scale in scales
        ])
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.bn = nn.BatchNorm1d(in_channels)
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, length)
        Returns:
            attended_x: (batch, channels, length)
        """

        scale_features = []
        for conv in self.scale_convs:
            scale_features.append(conv(x))
        
        multi_scale = torch.cat(scale_features, dim=1)  # (batch, channels, length)
        
        channel_att = self.channel_attention(multi_scale)
        channel_refined = multi_scale * channel_att
        
        avg_pool = torch.mean(channel_refined, dim=1, keepdim=True)  # (batch, 1, length)
        max_pool = torch.max(channel_refined, dim=1, keepdim=True)[0]  # (batch, 1, length)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # (batch, 2, length)
        spatial_att = self.spatial_attention(spatial_input)
        
        attended = channel_refined * spatial_att
        
        output = self.bn(attended + x)
        
        return output

class CrossAttention(nn.Module):
    """跨域注意力机制 - 融合时域和频域特征"""
    
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, time_features, freq_features):
        """
        Args:
            time_features: (batch, length, d_model) - 时域特征
            freq_features: (batch, length, d_model) - 频域特征
        Returns:
            fused_features: (batch, length, d_model) - 融合特征
        """
        batch_size, seq_len, _ = time_features.shape

        Q = self.w_q(time_features).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(freq_features).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(freq_features).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        fused_features = self.layer_norm(output + time_features)
        
        return fused_features

class FaultLocalizationHead(nn.Module):
    """故障定位头 - 输出故障发生的时间段"""
    
    def __init__(self, in_channels: int, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        
        self.temporal_classifier = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, 1, kernel_size=1), 
            nn.Sigmoid()
        )
        
        self.severity_regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 1),
            nn.Sigmoid()  # 输出0-1之间的严重程度
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, length) - 特征图
        Returns:
            localization: (batch, 1, length) - 每个时间步的故障概率
            severity: (batch, 1) - 故障严重程度
        """
        localization = self.temporal_classifier(x)  # (batch, 1, length)
        severity = self.severity_regressor(x)  # (batch, 1)
        
        return localization, severity

class MTAFN(nn.Module):
    """多任务注意力融合网络"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 5,
                 sequence_length: int = 256,
                 hidden_dim: int = 128,
                 n_heads: int = 4):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # 1. 小波变换层 - 时频域分解
        self.wavelet_transform = WaveletTransform(input_channels)
        
        # 2. 时域特征提取路径
        self.time_path_conv = nn.Sequential(
            # 深度可分离卷积减少参数
            nn.Conv1d(input_channels, input_channels, kernel_size=7, padding=3, groups=input_channels),
            nn.Conv1d(input_channels, hidden_dim // 2, kernel_size=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # 多尺度注意力
            MultiScaleAttention(hidden_dim // 2)
        )
        
        # 轻量级LSTM
        self.time_lstm = nn.LSTM(hidden_dim // 2, hidden_dim // 4, batch_first=False, bidirectional=True)
        
        # 3. 频域特征提取路径
        self.freq_path_approx = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            MultiScaleAttention(hidden_dim // 4)
        )
        
        self.freq_path_detail = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            MultiScaleAttention(hidden_dim // 4)
        )
        
        # 4. 跨域注意力融合
        self.cross_attention = CrossAttention(hidden_dim, n_heads)
        
        # 5. 多任务输出头
        # 故障分类头
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 故障定位头
        self.localization_head = FaultLocalizationHead(hidden_dim, sequence_length)
        
        # 6. 动态权重调整
        self.task_weights = nn.Parameter(torch.ones(3))  # [分类, 定位, 严重程度]
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, channels, length) - 输入信号
            return_attention: 是否返回注意力权重
        Returns:
            outputs: 包含分类、定位、严重程度的字典
        """
        batch_size = x.shape[0]
        
        # 1. 小波变换 - 时频域分解
        approx, detail = self.wavelet_transform(x)  # 低频和高频成分
        
        # 2. 时域特征提取
        # 卷积特征提取
        time_conv = self.time_path_conv(x)  # (batch, hidden_dim//2, length)
        
        # LSTM特征提取
        time_conv_seq = time_conv.transpose(1, 2)  # (batch, length, hidden_dim//2)
        lstm_input = time_conv_seq.transpose(0, 1)  # (length, batch, hidden_dim//2)
        time_lstm_out, _ = self.time_lstm(lstm_input)  # (length, batch, hidden_dim//2)
        time_lstm_out = time_lstm_out.transpose(0, 1)  # (batch, length, hidden_dim//2)
        
        # 结合卷积和LSTM特征
        time_features_combined = torch.cat([time_conv_seq, time_lstm_out], dim=-1)  # (batch, length, hidden_dim)
        
        # 3. 频域特征提取
        freq_approx = self.freq_path_approx(approx).transpose(1, 2)  # (batch, length//2, hidden_dim//4)
        freq_detail = self.freq_path_detail(detail).transpose(1, 2)   # (batch, length//2, hidden_dim//4)
        
        # 上采样到原始长度
        freq_approx = F.interpolate(freq_approx.transpose(1, 2), size=x.shape[-1], mode='linear', align_corners=False).transpose(1, 2)
        freq_detail = F.interpolate(freq_detail.transpose(1, 2), size=x.shape[-1], mode='linear', align_corners=False).transpose(1, 2)
        
        freq_features = torch.cat([freq_approx, freq_detail], dim=-1)  # (batch, length, hidden_dim//2)
        
        # 补齐到hidden_dim维度
        freq_features = F.pad(freq_features, (0, self.hidden_dim - freq_features.shape[-1]))
        
        # 4. 跨域注意力融合
        fused_features = self.cross_attention(time_features_combined, freq_features)  # (batch, length, hidden_dim)
        
        # 转换回卷积格式
        fused_conv = fused_features.transpose(1, 2)  # (batch, hidden_dim, length)
        
        # 5. 多任务输出
        # 故障分类
        classification_logits = self.classification_head(fused_conv)  # (batch, num_classes)
        
        # 故障定位和严重程度
        localization_prob, severity_score = self.localization_head(fused_conv)
        
        # 6. 动态权重归一化
        normalized_weights = F.softmax(self.task_weights, dim=0)
        
        outputs = {
            'classification': classification_logits,
            'localization': localization_prob.squeeze(1),  # (batch, length)
            'severity': severity_score.squeeze(1),  # (batch,)
            'task_weights': normalized_weights,
            'fused_features': fused_features if return_attention else None
        }
        
        return outputs

class MTAFNLoss(nn.Module):
    """多任务损失函数"""
    
    def __init__(self, alpha_cls=1.0, alpha_loc=0.5, alpha_sev=0.3):
        super().__init__()
        self.alpha_cls = alpha_cls
        self.alpha_loc = alpha_loc
        self.alpha_sev = alpha_sev
        
        self.cls_loss = nn.CrossEntropyLoss()
        self.loc_loss = nn.BCELoss()
        self.sev_loss = nn.MSELoss()
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: 模型输出字典
            targets: 目标字典 {'labels', 'localization', 'severity'}
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 分类损失
        cls_loss = self.cls_loss(outputs['classification'], targets['labels'])
        
        # 定位损失 (如果有定位标签)
        loc_loss = torch.tensor(0.0, device=outputs['classification'].device)
        if 'localization' in targets and targets['localization'] is not None:
            loc_loss = self.loc_loss(outputs['localization'], targets['localization'])
        
        # 严重程度损失 (如果有严重程度标签)
        sev_loss = torch.tensor(0.0, device=outputs['classification'].device)
        if 'severity' in targets and targets['severity'] is not None:
            sev_loss = self.sev_loss(outputs['severity'], targets['severity'])
        
        # 动态权重调整
        task_weights = outputs['task_weights']
        total_loss = (task_weights[0] * self.alpha_cls * cls_loss + 
                     task_weights[1] * self.alpha_loc * loc_loss + 
                     task_weights[2] * self.alpha_sev * sev_loss)
        
        loss_dict = {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'localization_loss': loc_loss,
            'severity_loss': sev_loss,
            'task_weights': task_weights
        }
        
        return total_loss, loss_dict

def create_lightweight_mtafn(input_channels=3, num_classes=5, sequence_length=256):
    """创建轻量级版本的MTAFN"""
    return MTAFN(
        input_channels=input_channels,
        num_classes=num_classes,
        sequence_length=sequence_length,
        hidden_dim=64,  # 减小隐藏维度
        n_heads=2       # 减少注意力头数
    )

def get_model_complexity(model, input_shape):
    """计算模型复杂度"""
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型大小 (MB)
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    # 测试推理时间
    model.eval()
    dummy_input = torch.randn(1, *input_shape)
    
    # 确保输入在正确的设备上
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    import time
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(dummy_input)
        
        # 测量推理时间
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        avg_inference_time = (time.time() - start_time) / 100 * 1000  # ms
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'avg_inference_time_ms': avg_inference_time
    }
