# -*- coding: utf-8 -*-
import math
import numpy as np
import os
import re
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import TensorDataset, DataLoader
import random
from scipy.signal import hilbert

class Config:
##根据实际任务需求还需自行改进
##Further improvements should be made according to the requirements of the actual task
    DATA_ROOT = 'D:/DATA/IGBT_data/data1'
    TRAIN_FILE_START = 10
    TRAIN_FILE_END = 50
    TEST_FILE_START = 96
    TEST_FILE_END = 106

    NUM_CLASSES = 6
    TIME_WINDOW = 4000
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50
    DELTA = 1.0
    LABEL_SMOOTHING = 0.05
    COSINE_T_MAX = 50
    GRAD_CLIP_NORM = 1.0
    DEVICE = 'cuda:0'
    RANDOM_SEED = 204
CFG = Config()

def validate_config():
    if CFG.TIME_WINDOW <= 0:
        raise ValueError('TIME_WINDOW 必须大于 0。')
    if CFG.BATCH_SIZE <= 0:
        raise ValueError('BATCH_SIZE 必须大于 0。')

def extract_features(signal):
    complex_features = []
    for ch in range(signal.shape[1]):
        analytic_signal = hilbert(signal[:, ch])
        real_part = np.real(analytic_signal)
        imag_part = np.imag(analytic_signal)
        complex_features.append(np.stack([real_part, imag_part], axis=1))
    return np.concatenate(complex_features, axis=1)

def load_data_and_normalize(root_dir):
    # For easier code reuse, sliding-window sampling is removed and each file directly provides 4000 points.
    # 为便于代码接入与复用，已去除滑窗采样，每个文件直接输入4000个数据点。
    folder_list = sorted(
        folder for folder in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, folder))
    )
    selected_folders = folder_list[:CFG.NUM_CLASSES]
    folder_to_label = {
        folder: idx for idx, folder in enumerate(selected_folders)
    }

    train_raw = []
    train_labels = []
    test_raw = []
    test_labels = []

    for folder_name in selected_folders:
        folder_path = os.path.join(root_dir, folder_name)
        label = folder_to_label[folder_name]

        files = sorted(
            f for f in os.listdir(folder_path)
            if f.endswith('.mat')
        )

        file_info_list = []
        for mat_file in files:
            match = re.search(r'-(\d+)V-(\d+)\.mat$', mat_file)
            if not match:
                match = re.search(r'(\d+)V-(\d+)\.mat$', mat_file)
            if not match:
                continue

            try:
                voltage = int(match.group(1))
                resistance = int(match.group(2))
                if resistance == 0:
                    continue

                power = voltage ** 2 / float(resistance)
                file_info_list.append((power, mat_file))
            except Exception:
                continue

        sorted_files = sorted(file_info_list, key=lambda x: x[0])

        train_files = [
            item[1]
            for item in sorted_files[
                CFG.TRAIN_FILE_START:CFG.TRAIN_FILE_END
            ]
        ]
        test_files = [
            item[1]
            for item in sorted_files[
                CFG.TEST_FILE_START:CFG.TEST_FILE_END
            ]
        ]

        for mat_file in train_files:
            try:
                mat_data = sio.loadmat(
                    os.path.join(folder_path, mat_file)
                )
                signals = mat_data['newData'][0, 0]['signals'][0, 0]['values']

                if signals.ndim == 3:
                    signals = signals.squeeze()

                if signals.shape[0] == 3 and signals.shape[1] > 3:
                    signals = signals.T

                if signals.ndim != 2 or signals.shape[1] != 3:
                    continue

                if len(signals) < CFG.TIME_WINDOW:
                    continue

                segment = np.asarray(
                    signals[:CFG.TIME_WINDOW],
                    dtype=np.float64,
                )

                if np.isnan(segment).any() or np.isinf(segment).any():
                    continue

                train_raw.append(segment)
                train_labels.append(label)

            except Exception:
                continue

        for mat_file in test_files:
            try:
                mat_data = sio.loadmat(
                    os.path.join(folder_path, mat_file)
                )
                signals = mat_data['newData'][0, 0]['signals'][0, 0]['values']

                if signals.ndim == 3:
                    signals = signals.squeeze()

                if signals.shape[0] == 3 and signals.shape[1] > 3:
                    signals = signals.T

                if signals.ndim != 2 or signals.shape[1] != 3:
                    continue

                if len(signals) < CFG.TIME_WINDOW:
                    continue

                segment = np.asarray(
                    signals[:CFG.TIME_WINDOW],
                    dtype=np.float64,
                )

                if np.isnan(segment).any() or np.isinf(segment).any():
                    continue

                test_raw.append(segment)
                test_labels.append(label)

            except Exception:
                continue

    if not train_raw or not test_raw:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            {},
        )

    scaler = MaxAbsScaler()
    scaler.fit(np.vstack(train_raw))

    X_train = np.stack([
        extract_features(scaler.transform(signal))
        for signal in train_raw
    ]).astype(np.float32)

    X_test = np.stack([
        extract_features(scaler.transform(signal))
        for signal in test_raw
    ]).astype(np.float32)

    y_train = np.asarray(train_labels, dtype=np.int64)
    y_test = np.asarray(test_labels, dtype=np.int64)

    valid_train = np.isfinite(X_train).all(axis=(1, 2))
    valid_test = np.isfinite(X_test).all(axis=(1, 2))

    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    X_test = X_test[valid_test]
    y_test = y_test[valid_test]

    return X_train, y_train, X_test, y_test, folder_to_label

def interleave_complex_channels(real, imag):
    return torch.stack([real, imag], dim=2).flatten(1, 2)

class TemporalCausalityLayer(nn.Module):

    def __init__(self, in_channels, kernel_size, stride=1, dilation=1, bias=False):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(f'in_channels 必须为偶数，但得到了 {in_channels}')
        self.complex_channels = in_channels // 2
        self.left_padding = (kernel_size - 1) * dilation
        self.temporal_conv = nn.Conv1d(self.complex_channels, self.complex_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=self.complex_channels, bias=bias)

    def forward(self, x):
        x_real = x[:, 0::2, :]
        x_imag = x[:, 1::2, :]
        if self.left_padding > 0:
            x_real = F.pad(x_real, (self.left_padding, 0))
            x_imag = F.pad(x_imag, (self.left_padding, 0))
        out_real = self.temporal_conv(x_real)
        out_imag = self.temporal_conv(x_imag)
        return interleave_complex_channels(out_real, out_imag)

class ComplexConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
        super().__init__()
        if in_channels % 2 != 0:
            raise ValueError(f'in_channels 必须为偶数，但得到了 {in_channels}')
        if out_channels % 2 != 0:
            raise ValueError(f'out_channels 必须为偶数，但得到了 {out_channels}')
        if kernel_size <= 0:
            raise ValueError(f'kernel_size 必须大于 0，但得到了 {kernel_size}')
        self.in_complex_channels = in_channels // 2
        self.out_complex_channels = out_channels // 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.left_padding = (kernel_size - 1) * dilation
        self.conv_wr = nn.Conv1d(self.in_complex_channels, self.out_complex_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=False)
        self.conv_wi = nn.Conv1d(self.in_complex_channels, self.out_complex_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, bias=False)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(self.out_complex_channels))
            self.bias_imag = nn.Parameter(torch.zeros(self.out_complex_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x):
        x_real = x[:, 0::2, :]
        x_imag = x[:, 1::2, :]
        if self.left_padding > 0:
            x_real = F.pad(x_real, (self.left_padding, 0))
            x_imag = F.pad(x_imag, (self.left_padding, 0))
        wr_xr = self.conv_wr(x_real)
        wr_xi = self.conv_wr(x_imag)
        wi_xr = self.conv_wi(x_real)
        wi_xi = self.conv_wi(x_imag)
        out_real = wr_xr - wi_xi
        out_imag = wi_xr + wr_xi
        if self.bias_real is not None:
            out_real = out_real + self.bias_real.view(1, -1, 1)
            out_imag = out_imag + self.bias_imag.view(1, -1, 1)
        return interleave_complex_channels(out_real, out_imag)

class ComplexGELU(nn.Module):

    def forward(self, x):
        x_real = F.gelu(x[:, 0::2, :])
        x_imag = F.gelu(x[:, 1::2, :])
        return interleave_complex_channels(x_real, x_imag)

class ComplexMagnitudeNorm1d(nn.Module):

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        if num_channels % 2 != 0:
            raise ValueError(f'num_channels 必须为偶数，但得到了 {num_channels}')
        self.complex_channels = num_channels // 2
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.complex_channels))

    def forward(self, x):
        real = x[:, 0::2, :]
        imag = x[:, 1::2, :]

        rms = torch.sqrt(
            (real.pow(2) + imag.pow(2)).mean(dim=1, keepdim=True)
            + self.eps
        )

        scale = self.weight.view(1, -1, 1)

        real = real / rms * scale
        imag = imag / rms * scale

        return interleave_complex_channels(real, imag)

class EnhancedComplexCausalCNN(nn.Module):

    def __init__(self, input_dim, d_model):
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError(f'input_dim 必须为偶数，但得到了 {input_dim}')
        if d_model % 2 != 0:
            raise ValueError(f'd_model 必须为偶数，但得到了 {d_model}')
        self.temporal_downsample_factor = 8
        self.temporal1 = TemporalCausalityLayer(input_dim, kernel_size=64, stride=2, dilation=1)
        self.complex1 = ComplexConv1d(input_dim, 16, kernel_size=3, stride=1, dilation=1)
        self.act1 = ComplexGELU()
        self.norm1 = ComplexMagnitudeNorm1d(16)
        self.temporal2 = TemporalCausalityLayer(16, kernel_size=3, stride=2, dilation=1)
        self.complex2 = ComplexConv1d(16, 32, kernel_size=3, stride=1, dilation=1)
        self.act2 = ComplexGELU()
        self.norm2 = ComplexMagnitudeNorm1d(32)
        self.temporal3 = TemporalCausalityLayer(32, kernel_size=3, stride=2, dilation=1)
        self.complex3 = ComplexConv1d(32, 64, kernel_size=3, stride=1, dilation=1)
        self.act3 = ComplexGELU()
        self.norm3 = ComplexMagnitudeNorm1d(64)
        self.temporal4 = TemporalCausalityLayer(64, kernel_size=3, stride=1, dilation=1)
        self.complex4 = ComplexConv1d(64, 128, kernel_size=3, stride=1, dilation=1)
        self.act4 = ComplexGELU()
        self.norm4 = ComplexMagnitudeNorm1d(128)
        self.temporal5 = TemporalCausalityLayer(128, kernel_size=3, stride=1, dilation=1)
        self.complex5 = ComplexConv1d(128, 256, kernel_size=3, stride=1, dilation=1)
        self.act5 = ComplexGELU()
        self.norm5 = ComplexMagnitudeNorm1d(256)
        self.output_proj = nn.Sequential(ComplexConv1d(256, d_model, kernel_size=1, stride=1), ComplexGELU())

    def forward(self, x):
        x = self.temporal1(x)
        x = self.norm1(self.act1(self.complex1(x)))
        x = self.temporal2(x)
        x = self.norm2(self.act2(self.complex2(x)))
        x = self.temporal3(x)
        x = self.norm3(self.act3(self.complex3(x)))
        x = self.temporal4(x)
        x = self.norm4(self.act4(self.complex4(x)))
        x = self.temporal5(x)
        x = self.norm5(self.act5(self.complex5(x)))
        x = self.output_proj(x)
        return x

class CausalMaskAttention(nn.Module):

    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, 'd_model 必须被 nhead 整除。'
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, C = x.shape
        q = self.q_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0 if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, C)
        return self.out_proj(out)

class SpatialFeatureAttention(nn.Module):

    def __init__(self, d_model, spatial_dim, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.spatial_dim = spatial_dim
        self.q_proj = nn.Linear(spatial_dim, spatial_dim)
        self.k_proj = nn.Linear(spatial_dim, spatial_dim)
        self.v_proj = nn.Linear(spatial_dim, spatial_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, C = x.shape
        if L != self.spatial_dim:
            raise ValueError(f'SpatialFeatureAttention 期望序列长度 {self.spatial_dim}，但实际得到 {L}。')
        f_spatial = x.transpose(1, 2)
        q_spatial = self.q_proj(f_spatial)
        k_spatial = self.k_proj(f_spatial)
        v_spatial = self.v_proj(f_spatial)
        d_spa = q_spatial.size(-1)
        attn_logits = torch.matmul(q_spatial, k_spatial.transpose(-2, -1)) / math.sqrt(d_spa)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        spatial_out = torch.matmul(attn, v_spatial)
        out = spatial_out.transpose(1, 2).contiguous()
        return self.out_proj(out)

class HybridCausalTransformerLayer(nn.Module):

    def __init__(self, d_model, nhead, spatial_dim, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.temporal_attn = CausalMaskAttention(d_model, nhead)
        self.spatial_attn = SpatialFeatureAttention(d_model=d_model, spatial_dim=spatial_dim, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout_ffn = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1_temporal = nn.LayerNorm(d_model)
        self.norm1_spatial = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1_temporal = nn.Dropout(dropout)
        self.dropout1_spatial = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        temporal_att = self.temporal_attn(x)
        x_temporal = self.norm1_temporal(x + self.dropout1_temporal(temporal_att))
        spatial_att = self.spatial_attn(x)
        x_spatial = self.norm1_spatial(x + self.dropout1_spatial(spatial_att))
        g = self.gate(x_temporal)
        x_gated = g * x_temporal + (1.0 - g) * x_spatial
        x2 = self.linear2(self.dropout_ffn(self.activation(self.linear1(x_gated))))
        x_out = self.norm2(x_gated + self.dropout2(x2))
        return x_out

class ComplexCausalTransformer(nn.Module):

    def __init__(self, input_dim, d_model, max_len, nhead, num_layers, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        if input_dim % 2 != 0:
            raise ValueError(f'input_dim 必须为偶数，以按实部/虚部成对重构，但得到了 {input_dim}')
        if d_model % 2 != 0:
            raise ValueError(f'd_model 必须是偶数以适配复数特征，但得到了 {d_model}')
        self.complex_channels = d_model // 2
        self.cvnn_stem = EnhancedComplexCausalCNN(input_dim=input_dim, d_model=d_model)
        self.temporal_downsample_factor = self.cvnn_stem.temporal_downsample_factor
        self.reconstructor = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model, d_model), nn.GELU())
        self.real_recon_decoder = nn.ConvTranspose1d(in_channels=d_model, out_channels=input_dim // 2, kernel_size=self.temporal_downsample_factor, stride=self.temporal_downsample_factor, padding=0, output_padding=0)
        self.imag_recon_decoder = nn.ConvTranspose1d(in_channels=d_model, out_channels=input_dim // 2, kernel_size=self.temporal_downsample_factor, stride=self.temporal_downsample_factor, padding=0, output_padding=0)
        self.amp_phase_dim = d_model
        self.pos_encoder = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([HybridCausalTransformerLayer(d_model=d_model, nhead=nhead, spatial_dim=max_len) for _ in range(num_layers)])
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(0.3), nn.Linear(d_model // 2, num_classes))

    @staticmethod
    def _to_amplitude_phase(x_features):
        real = x_features[..., 0::2]
        imag = x_features[..., 1::2]
        eps = 0.0001
        amplitude = torch.sqrt(real.pow(2) + imag.pow(2) + eps)

        real_unit = real / amplitude
        imag_unit = imag / amplitude

        phase = torch.atan2(imag_unit + eps, real_unit + eps)
        phase = torch.nan_to_num(
            phase,
            nan=0.0,
            posinf=math.pi,
            neginf=-math.pi,
        )

        amp_phase_features = torch.cat([phase, amplitude], dim=-1)
        return (amp_phase_features, phase, amplitude)

    @staticmethod
    def _interleave_last_dim(real, imag):
        return torch.stack([real, imag], dim=-1).flatten(-2)

    def forward(self, x):
        B, input_length, _ = x.shape
        x_stem_input = x.transpose(1, 2)
        x_features = self.cvnn_stem(x_stem_input)
        x_features = x_features.transpose(1, 2)
        feature_length = x_features.size(1)
        amp_phase_features, phase_features, amplitude_features = self._to_amplitude_phase(x_features)
        recon_features = self.reconstructor(amp_phase_features)
        recon_features = recon_features.transpose(1, 2).contiguous()
        recon_real = self.real_recon_decoder(recon_features).transpose(1, 2).contiguous()
        recon_imag = self.imag_recon_decoder(recon_features).transpose(1, 2).contiguous()
        reconstructed_signal = self._interleave_last_dim(recon_real, recon_imag)
        if reconstructed_signal.size(1) != input_length:
            raise RuntimeError(f'重构长度异常: 输入长度为 {input_length}，但重构得到 {reconstructed_signal.size(1)}。')
        if feature_length > self.pos_encoder.num_embeddings:
            raise ValueError(f'Transformer 特征长度 {feature_length} 超过位置编码上限 {self.pos_encoder.num_embeddings}。')
        pos = torch.arange(feature_length, device=x.device).unsqueeze(0).repeat(B, 1)
        pos_emb = self.pos_encoder(pos)
        x_transformer_input = amp_phase_features + pos_emb
        for layer in self.layers:
            x_transformer_input = layer(x_transformer_input)
        x_pooled = x_transformer_input.mean(dim=1)
        class_logits = self.classifier(x_pooled)
        return (class_logits, reconstructed_signal, x_pooled)

class ComplexSignalReconstructionLoss(nn.Module):

    def forward(self, reconstructed_signal, original_signal):
        if reconstructed_signal.shape != original_signal.shape:
            raise ValueError(f'重构信号与原始信号形状不一致: {reconstructed_signal.shape} vs {original_signal.shape}')
        if original_signal.shape[-1] % 2 != 0:
            raise ValueError('输入特征维度必须为偶数，才能按实部/虚部成对计算重构损失。')
        real_diff = reconstructed_signal[..., 0::2] - original_signal[..., 0::2]
        imag_diff = reconstructed_signal[..., 1::2] - original_signal[..., 1::2]
        real_loss = torch.linalg.vector_norm(real_diff, ord=2, dim=-1)
        imag_loss = torch.linalg.vector_norm(imag_diff, ord=2, dim=-1)
        loss_recon = (real_loss + imag_loss).mean()
        return torch.nan_to_num(loss_recon, nan=0.0, posinf=10000.0, neginf=0.0)

def calculate_dynamic_lambda(features, labels, delta=1.0):
    with torch.no_grad():
        features_detached = features.detach()
        labels_detached = labels.detach()
        batch_size = features_detached.size(0)
        if batch_size < 2:
            return (1.0, float('nan'), float('nan'))
        pairwise_dist = torch.cdist(features_detached, features_detached, p=2)
        upper_triangle = torch.triu(torch.ones(batch_size, batch_size, dtype=torch.bool, device=features_detached.device), diagonal=1)
        same_class_mask = (labels_detached.unsqueeze(0) == labels_detached.unsqueeze(1)) & upper_triangle
        different_class_mask = (labels_detached.unsqueeze(0) != labels_detached.unsqueeze(1)) & upper_triangle
        same_pair_count = int(same_class_mask.sum().item())
        different_pair_count = int(different_class_mask.sum().item())
        if same_pair_count == 0 or different_pair_count == 0:
            return (1.0, float('nan'), float('nan'))
        avg_intra_dist = pairwise_dist[same_class_mask].mean()
        avg_inter_dist = pairwise_dist[different_class_mask].mean()
        if not torch.isfinite(avg_intra_dist) or not torch.isfinite(avg_inter_dist):
            return (1.0, float('nan'), float('nan'))
        if avg_inter_dist.item() <= 1e-12:
            return (1.0, float(avg_intra_dist.item()), float(avg_inter_dist.item()))
        distance_ratio = avg_intra_dist / avg_inter_dist
        separation_term = torch.clamp(1.0 - distance_ratio, min=0.0)
        lambda_recon = 1.0 - float(delta) * separation_term
        return (float(lambda_recon.item()), float(avg_intra_dist.item()), float(avg_inter_dist.item()))

def train_epoch(model, dataloader, criterion_class, criterion_recon, optimizer, device, delta, epoch, total_epochs):
    model.train()
    correct = 0
    total = 0
    total_batches = len(dataloader)
    bar_width = 40
    for batch_idx, (x_complex, y_label) in enumerate(dataloader, start=1):
        x_complex = x_complex.to(device)
        y_label = y_label.to(device)
        optimizer.zero_grad()
        class_logits, reconstructed_signal, fused_features = model(x_complex)
        if torch.isfinite(class_logits).all() and torch.isfinite(reconstructed_signal).all() and torch.isfinite(fused_features).all():
            loss_class = criterion_class(class_logits, y_label)
            loss_recon = criterion_recon(reconstructed_signal, x_complex)
            lambda_recon, _, _ = calculate_dynamic_lambda(fused_features, y_label, delta=delta)
            loss = loss_class + lambda_recon * loss_recon
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.GRAD_CLIP_NORM)
                optimizer.step()
                pred = torch.argmax(class_logits, dim=1)
                correct += (pred == y_label).sum().item()
                total += y_label.size(0)
        progress = batch_idx / total_batches
        filled = int(bar_width * progress)
        bar = '=' * filled + '-' * (bar_width - filled)
        print(f'\rEpoch {epoch:02d}/{total_epochs} [{bar}] {progress * 100:6.2f}%', end='', flush=True)
    print()
    if total == 0:
        return 0.0
    return correct / total

def test_epoch(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_complex, y_label in dataloader:
            x_complex = x_complex.to(device)
            y_label = y_label.to(device)
            class_logits, _, _ = model(x_complex)
            if not torch.isfinite(class_logits).all():
                continue
            pred = torch.argmax(class_logits, dim=1)
            correct += (pred == y_label).sum().item()
            total += y_label.size(0)
    if total == 0:
        return 0.0
    return correct / total

def main():
    if CFG.DEVICE.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(CFG.DEVICE)
    else:
        device = torch.device('cpu')
    transformer_length = CFG.TIME_WINDOW // 8
    Xtr_complex, ytr_label, Xte_complex, yte_label, folder_to_label = load_data_and_normalize(CFG.DATA_ROOT)
    if len(Xtr_complex) == 0 or len(Xte_complex) == 0:
        return
    Xtr = torch.from_numpy(Xtr_complex)
    Xte = torch.from_numpy(Xte_complex)
    ytr_label = torch.from_numpy(ytr_label)
    yte_label = torch.from_numpy(yte_label)
    train_dl = DataLoader(TensorDataset(Xtr, ytr_label), batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    test_dl = DataLoader(TensorDataset(Xte, yte_label), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    input_dim = Xtr.shape[-1]
    num_classes = len(folder_to_label)
    model = ComplexCausalTransformer(input_dim=input_dim, d_model=64, max_len=transformer_length, nhead=4, num_layers=2, num_classes=num_classes).to(device)
    criterion_class = nn.CrossEntropyLoss(label_smoothing=CFG.LABEL_SMOOTHING)
    criterion_recon = ComplexSignalReconstructionLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.COSINE_T_MAX)
    for epoch in range(1, CFG.EPOCHS + 1):
        train_acc = train_epoch(model, train_dl, criterion_class, criterion_recon, optimizer, device, CFG.DELTA, epoch, CFG.EPOCHS)
        test_acc = test_epoch(model, test_dl, device)
        scheduler.step()
        print(f'Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
        
if __name__ == '__main__':
    validate_config()
    random.seed(CFG.RANDOM_SEED)
    np.random.seed(CFG.RANDOM_SEED)
    torch.manual_seed(CFG.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    main()
