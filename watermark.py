import torch
import torch.nn.functional as F
from scipy.stats import norm, truncnorm
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import logging

class Gaussian_Shading:
    def __init__(self, ch_factor=1, hw_factor=8, fpr=0.000001, user_number=1000000):
        self.patch_size = 2
        self.logical_size = 64 // self.patch_size  # 32x32 宏像素
        self.channels = 4
        self.logical_length = self.channels * self.logical_size * self.logical_size
        
        # 全域均匀拼图 (Uniform Tiling) 掩码
        self.mask_anchor = torch.zeros((self.logical_size, self.logical_size), dtype=torch.bool).cuda()
        self.mask_local1 = torch.zeros((self.logical_size, self.logical_size), dtype=torch.bool).cuda()
        self.mask_local2 = torch.zeros((self.logical_size, self.logical_size), dtype=torch.bool).cuda()

        for i in range(4):
            for j in range(4):
                r0, r1 = i * 8, (i + 1) * 8
                c0, c1 = j * 8, (j + 1) * 8
                self.mask_anchor[r0:r0+4, c0:c0+4] = True
                self.mask_anchor[r0+4:r1, c0+4:c1] = True
                self.mask_local1[r0:r0+4, c0+4:c1] = True
                self.mask_local2[r0+4:r1, c0:c0+4] = True

        # ==========================================
        # 创新点：容量飙升至 320 Bits！
        # ==========================================
        self.global_bits_len = 64   # 维持 64 bit 锚点，确保同步极度稳定
        self.local1_bits_len = 256  # 局部 1 容量翻倍
        self.local2_bits_len = 256  # 局部 2 容量翻倍
        
        self.key_32 = None
        self.watermark_anchor = None
        self.watermark_local1 = None
        self.watermark_local2 = None
        
        self.roar_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 0.5))

    def create_watermark_and_return_w(self):
        w_64_size = self.logical_size * self.patch_size
        
        # 核心创新 1：将密钥提升到最高频率（64x64），彻底打断空间相关性
        self.key_64 = torch.randint(0, 2, [1, self.channels, w_64_size, w_64_size], dtype=torch.long).cuda()
        
        self.watermark_anchor = torch.randint(0, 2, [self.global_bits_len]).cuda()
        self.watermark_local1 = torch.randint(0, 2, [self.local1_bits_len]).cuda()
        self.watermark_local2 = torch.randint(0, 2, [self.local2_bits_len]).cuda()

        sd_tensor_32 = torch.zeros((1, self.channels, self.logical_size, self.logical_size), dtype=torch.long).cuda()
        
        def fill_mask_circular(mask, watermark_bits):
            """动态环形扩频：将比特序列填充满指定的 Mask 区域"""
            spatial_indices = mask.nonzero(as_tuple=True)
            pixels_in_mask = spatial_indices[0].shape[0]
            target_length = pixels_in_mask * self.channels
            bits_len = watermark_bits.shape[0]
            
            repeats = (target_length // bits_len) + 1
            repeated = watermark_bits.repeat(repeats)[:target_length]
            
            idx = 0
            for c in range(self.channels):
                sd_tensor_32[0, c, spatial_indices[0], spatial_indices[1]] = repeated[idx : idx + pixels_in_mask]
                idx += pixels_in_mask

        fill_mask_circular(self.mask_anchor, self.watermark_anchor)
        fill_mask_circular(self.mask_local1, self.watermark_local1)
        fill_mask_circular(self.mask_local2, self.watermark_local2)

        # 核心创新 2：只对水印载荷进行宏像素插值，维持 2x2 的空间鲁棒性
        m_64 = F.interpolate(sd_tensor_32.float(), scale_factor=self.patch_size, mode='nearest').long()
        
        # 核心创新 3：低频载荷 + 高频密钥 = 完美的伪随机白噪声
        target_bits_64 = (m_64 + self.key_64) % 2
        
        signs_64 = (target_bits_64 * 2 - 1).half()
        z_64 = torch.randn((1, self.channels, w_64_size, w_64_size), dtype=torch.float16, device='cuda')
        w_64 = torch.abs(z_64) * signs_64

        return w_64

    def eval_global_anchor(self, soft_m_32):
        spatial_indices = self.mask_anchor.nonzero(as_tuple=True)
        gathered_soft = []
        for c in range(self.channels):
            gathered_soft.append(soft_m_32[0, c, spatial_indices[0], spatial_indices[1]])
        gathered_soft = torch.cat(gathered_soft)
        
        target_length = gathered_soft.shape[0]
        
        # 环形 LLR 极速累加
        extracted_anchor_llr = torch.zeros(self.global_bits_len, device=gathered_soft.device)
        indices = torch.arange(target_length, device=gathered_soft.device) % self.global_bits_len
        extracted_anchor_llr.scatter_add_(0, indices, gathered_soft)
        
        dec_anchor = (extracted_anchor_llr > 0).int()
        acc = (dec_anchor == self.watermark_anchor).float().mean().item()
        return acc

    def eval_watermark(self, reversed_w_64):
        restored_w_64 = self.roar_blur(reversed_w_64)
        
        # 提取全分辨率密钥符号
        key_sign_64 = (self.key_64 * -2 + 1).float() 

        # Baseline 优先
        # 核心变动：必须先解密 (乘密钥)，再池化 (Average Pooling)
        unlocked_w_64 = restored_w_64.float() * key_sign_64
        baseline_soft = F.avg_pool2d(unlocked_w_64, kernel_size=self.patch_size)
        baseline_soft = (baseline_soft - baseline_soft.mean()) / (baseline_soft.std() + 1e-6)
        baseline_acc = self.eval_global_anchor(baseline_soft)

        if baseline_acc >= 0.85:
            refined_angle = 0.0
            best_scale = 1.0
            best_acc = baseline_acc
        else:
            search_angles = list(range(-180, 181, 5)) 
            search_scales = [0.95, 1.0, 1.05] 
            best_acc = -1
            best_angle = 0
            best_scale = 1.0
            acc_curve_for_refinement = []
            
            for scale in search_scales:
                curve = []
                for angle in search_angles:
                    aligned_w_64 = TF.affine(restored_w_64, angle=-angle, translate=[0,0], scale=scale, shear=0, interpolation=TF.InterpolationMode.BILINEAR)
                    
                    # 每次几何尝试的解码逻辑：对齐 -> 解密 -> 池化
                    unlocked_aligned_w_64 = aligned_w_64.float() * key_sign_64
                    soft_m_32 = F.avg_pool2d(unlocked_aligned_w_64, kernel_size=self.patch_size)
                    soft_m_32 = (soft_m_32 - soft_m_32.mean()) / (soft_m_32.std() + 1e-6)
                    
                    acc = self.eval_global_anchor(soft_m_32)
                    curve.append(acc)
                    if acc > best_acc:
                        best_acc = acc
                        best_angle = angle
                        best_scale = scale
                if scale == best_scale:
                    acc_curve_for_refinement = curve

            best_idx = search_angles.index(best_angle)
            refined_angle = best_angle
            if 0 < best_idx < len(search_angles) - 1:
                y_neg, y_0, y_pos = acc_curve_for_refinement[best_idx - 1], acc_curve_for_refinement[best_idx], acc_curve_for_refinement[best_idx + 1]
                denom = (y_neg - 2 * y_0 + y_pos)
                if abs(denom) > 1e-6:
                    refined_angle += ((y_neg - y_pos) / (2 * denom)) * 5.0

            logging.info(f"[Sync] 检测尺度: {best_scale} | 矫正角度: {-refined_angle:.2f}° | 锚点置信度: {best_acc:.4f}")

        # 最终输出级解码
        final_aligned_w_64 = TF.affine(restored_w_64, angle=-refined_angle, translate=[0,0], scale=best_scale, shear=0, interpolation=TF.InterpolationMode.BILINEAR)
        final_unlocked_w_64 = final_aligned_w_64.float() * key_sign_64
        final_soft_m_32 = F.avg_pool2d(final_unlocked_w_64, kernel_size=self.patch_size)
        final_soft_m_32 = (final_soft_m_32 - final_soft_m_32.mean()) / (final_soft_m_32.std() + 1e-6)

        def decode_mask_circular(mask, bits_len, target_watermark):
            spatial_indices = mask.nonzero(as_tuple=True)
            gathered_soft = []
            for c in range(self.channels):
                gathered_soft.append(final_soft_m_32[0, c, spatial_indices[0], spatial_indices[1]])
            gathered_soft = torch.cat(gathered_soft)
            
            target_length = gathered_soft.shape[0]
            
            extracted_llr = torch.zeros(bits_len, device=gathered_soft.device)
            indices = torch.arange(target_length, device=gathered_soft.device) % bits_len
            extracted_llr.scatter_add_(0, indices, gathered_soft)
            
            dec_bits = (extracted_llr > 0).int()
            acc = (dec_bits == target_watermark).float().mean().item()
            return acc

        acc_anchor = decode_mask_circular(self.mask_anchor, self.global_bits_len, self.watermark_anchor)
        acc_local1 = decode_mask_circular(self.mask_local1, self.local1_bits_len, self.watermark_local1)
        acc_local2 = decode_mask_circular(self.mask_local2, self.local2_bits_len, self.watermark_local2)

        acc_local_max = max(acc_local1, acc_local2)
        
        return acc_anchor, acc_local_max

        def decode_mask_circular(mask, bits_len, target_watermark):
            """环形软判决解码"""
            spatial_indices = mask.nonzero(as_tuple=True)
            gathered_soft = []
            for c in range(self.channels):
                gathered_soft.append(final_soft_m_32[0, c, spatial_indices[0], spatial_indices[1]])
            gathered_soft = torch.cat(gathered_soft)
            
            target_length = gathered_soft.shape[0]
            
            extracted_llr = torch.zeros(bits_len, device=gathered_soft.device)
            indices = torch.arange(target_length, device=gathered_soft.device) % bits_len
            extracted_llr.scatter_add_(0, indices, gathered_soft)
            
            dec_bits = (extracted_llr > 0).int()
            acc = (dec_bits == target_watermark).float().mean().item()
            return acc

        acc_anchor = decode_mask_circular(self.mask_anchor, self.global_bits_len, self.watermark_anchor)
        acc_local1 = decode_mask_circular(self.mask_local1, self.local1_bits_len, self.watermark_local1)
        acc_local2 = decode_mask_circular(self.mask_local2, self.local2_bits_len, self.watermark_local2)

        acc_local_max = max(acc_local1, acc_local2)
        
        return acc_anchor, acc_local_max