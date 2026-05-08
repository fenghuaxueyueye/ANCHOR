import math
import warnings
import logging

import torch
import torch.nn.functional as F
from scipy.stats import norm, truncnorm, binom
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


def _familywise_to_single_test_fpr(target_fpr: float, n_tests: int = 1) -> float:
    """
    Convert the total / family-wise FPR into the single-test FPR.

    If several candidates are tested and the best one is selected, for example
    rotation-scale synchronization candidates or multi-user payload matching,
    the single-test threshold must be tightened:

        1 - (1 - alpha) ** n_tests <= target_fpr

    For small target_fpr, this is close to alpha = target_fpr / n_tests.
    """
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1.")
    if not (0.0 < float(target_fpr) < 1.0):
        raise ValueError("target_fpr must be in (0, 1).")

    return -math.expm1(math.log1p(-float(target_fpr)) / float(n_tests))


def binomial_threshold_bits(k: int, target_fpr: float, n_tests: int = 1):
    """
    Gaussian Shading / GaussMarker style threshold.

    Null hypothesis:
        For an unwatermarked image, the extracted bits are approximately random,
        so the matching count X between extracted bits and target bits follows
        X ~ Binomial(k, 0.5).

    The returned integer tau_bits is the smallest threshold satisfying:
        P(X >= tau_bits) <= single_test_fpr

    If n_tests > 1, the threshold controls the family-wise FPR after multiple
    candidate tests.
    """
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive.")

    single_test_fpr = _familywise_to_single_test_fpr(target_fpr, n_tests)

    tau_bits = k + 1
    for t in range(0, k + 1):
        # binom.sf(t - 1, k, 0.5) = P(X >= t)
        tail_prob = float(binom.sf(t - 1, k, 0.5))
        if tail_prob <= single_test_fpr:
            tau_bits = t
            break

    if tau_bits <= k:
        actual_single_fpr = float(binom.sf(tau_bits - 1, k, 0.5))
    else:
        actual_single_fpr = 0.0

    if actual_single_fpr <= 0.0:
        actual_family_fpr = 0.0
    else:
        actual_family_fpr = -math.expm1(float(n_tests) * math.log1p(-actual_single_fpr))

    return tau_bits, tau_bits / float(k), actual_family_fpr, single_test_fpr


class Gaussian_Shading:
    def __init__(
        self,
        ch_factor=1,
        hw_factor=8,
        fpr=0.000001,
        user_number=1000000,
        anchor_search_trials=None,
        payload_search_trials=None,
        fpr_anchor=None,
        fpr_payload=None,
    ):
        self.ch_factor = ch_factor
        self.hw_factor = hw_factor
        self.fpr = float(fpr)
        self.user_number = max(1, int(user_number))

        self.patch_size = 2
        self.logical_size = 64 // self.patch_size  # 32x32 macro-pixels
        self.channels = 4
        self.logical_length = self.channels * self.logical_size * self.logical_size

        # Search space used in anchor synchronization. The threshold must count
        # all tested candidates because eval_watermark selects the best match.
        self.search_angles = list(range(-180, 181, 5))
        self.search_scales = [0.95, 1.0, 1.05]

        if anchor_search_trials is None:
            # One baseline candidate + all angle-scale candidates in the fallback scan.
            self.anchor_search_trials = 1 + len(self.search_angles) * len(self.search_scales)
        else:
            self.anchor_search_trials = max(1, int(anchor_search_trials))

        if payload_search_trials is None:
            # Current ANCHOR code tests local1 and local2 and keeps the better one.
            self.payload_search_trials = 2
        else:
            self.payload_search_trials = max(1, int(payload_search_trials))

        self.fpr_anchor = self.fpr if fpr_anchor is None else float(fpr_anchor)
        self.fpr_payload = self.fpr if fpr_payload is None else float(fpr_payload)

        # Uniform tiling masks.
        self.mask_anchor = torch.zeros((self.logical_size, self.logical_size), dtype=torch.bool).cuda()
        self.mask_local1 = torch.zeros((self.logical_size, self.logical_size), dtype=torch.bool).cuda()
        self.mask_local2 = torch.zeros((self.logical_size, self.logical_size), dtype=torch.bool).cuda()

        for i in range(4):
            for j in range(4):
                r0, r1 = i * 8, (i + 1) * 8
                c0, c1 = j * 8, (j + 1) * 8
                self.mask_anchor[r0:r0 + 4, c0:c0 + 4] = True
                self.mask_anchor[r0 + 4:r1, c0 + 4:c1] = True
                self.mask_local1[r0:r0 + 4, c0 + 4:c1] = True
                self.mask_local2[r0 + 4:r1, c0:c0 + 4] = True

        # Capacity setting.
        self.global_bits_len = 64
        self.local1_bits_len = 128
        self.local2_bits_len = 128

        self.key_32 = None
        self.key_64 = None
        self.watermark_anchor = None
        self.watermark_local1 = None
        self.watermark_local2 = None

        self.roar_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 0.5))

        # Gaussian Shading / GaussMarker style statistical thresholds.
        self.tau_anchor_bits, self.tau_anchor, self.actual_fpr_anchor, self.single_fpr_anchor = \
            binomial_threshold_bits(
                k=self.global_bits_len,
                target_fpr=self.fpr_anchor,
                n_tests=self.anchor_search_trials,
            )

        # The payload stage may compare two local blocks and N users. In the
        # current code there is only one generated watermark, but keeping
        # user_number here makes the code compatible with multi-user tracing.
        self.payload_total_tests = self.payload_search_trials * self.user_number

        self.tau_local1_bits, self.tau_local1, self.actual_fpr_local1, self.single_fpr_payload = \
            binomial_threshold_bits(
                k=self.local1_bits_len,
                target_fpr=self.fpr_payload,
                n_tests=self.payload_total_tests,
            )

        self.tau_local2_bits, self.tau_local2, self.actual_fpr_local2, _ = \
            binomial_threshold_bits(
                k=self.local2_bits_len,
                target_fpr=self.fpr_payload,
                n_tests=self.payload_total_tests,
            )

        # Backward-compatible variable names.
        self.tau_onebit = self.tau_anchor
        self.tau_bits = self.tau_local1
        self.tau_onebit_count = self.tau_anchor_bits
        self.tau_bits_count = self.tau_local1_bits
        self.tp_onebit_count = 0
        self.tp_bits_count = 0

        if self.tau_anchor_bits > self.global_bits_len:
            warnings.warn(
                f"Anchor threshold is impossible: tau_anchor_bits={self.tau_anchor_bits} > "
                f"global_bits_len={self.global_bits_len}. Increase anchor bits or relax fpr_anchor."
            )

        if self.tau_local1_bits > self.local1_bits_len or self.tau_local2_bits > self.local2_bits_len:
            warnings.warn(
                "Payload threshold is impossible. Increase payload bits, reduce user_number, "
                "or relax fpr_payload."
            )

    def get_threshold_summary(self):
        return {
            "anchor": {
                "bits": self.global_bits_len,
                "tau_bits": int(self.tau_anchor_bits),
                "tau_acc": float(self.tau_anchor),
                "target_fpr": float(self.fpr_anchor),
                "actual_fpr": float(self.actual_fpr_anchor),
                "n_tests": int(self.anchor_search_trials),
                "single_test_fpr": float(self.single_fpr_anchor),
            },
            "payload_local1": {
                "bits": self.local1_bits_len,
                "tau_bits": int(self.tau_local1_bits),
                "tau_acc": float(self.tau_local1),
                "target_fpr": float(self.fpr_payload),
                "actual_fpr": float(self.actual_fpr_local1),
                "n_tests": int(self.payload_total_tests),
                "single_test_fpr": float(self.single_fpr_payload),
            },
            "payload_local2": {
                "bits": self.local2_bits_len,
                "tau_bits": int(self.tau_local2_bits),
                "tau_acc": float(self.tau_local2),
                "target_fpr": float(self.fpr_payload),
                "actual_fpr": float(self.actual_fpr_local2),
                "n_tests": int(self.payload_total_tests),
                "single_test_fpr": float(self.single_fpr_payload),
            },
        }

    def create_watermark_and_return_w(self):
        w_64_size = self.logical_size * self.patch_size

        # High-resolution key: breaks spatial correlation at the 64x64 level.
        self.key_64 = torch.randint(
            0,
            2,
            [1, self.channels, w_64_size, w_64_size],
            dtype=torch.long,
        ).cuda()

        self.watermark_anchor = torch.randint(0, 2, [self.global_bits_len]).cuda()
        self.watermark_local1 = torch.randint(0, 2, [self.local1_bits_len]).cuda()
        self.watermark_local2 = torch.randint(0, 2, [self.local2_bits_len]).cuda()

        sd_tensor_32 = torch.zeros((1, self.channels, self.logical_size, self.logical_size), dtype=torch.long).cuda()

        def fill_mask_circular(mask, watermark_bits):
            """Spread a bit sequence circularly over the selected mask region."""
            spatial_indices = mask.nonzero(as_tuple=True)
            pixels_in_mask = spatial_indices[0].shape[0]
            target_length = pixels_in_mask * self.channels
            bits_len = watermark_bits.shape[0]

            repeats = (target_length // bits_len) + 1
            repeated = watermark_bits.repeat(repeats)[:target_length]

            idx = 0
            for c in range(self.channels):
                sd_tensor_32[0, c, spatial_indices[0], spatial_indices[1]] = repeated[idx: idx + pixels_in_mask]
                idx += pixels_in_mask

        fill_mask_circular(self.mask_anchor, self.watermark_anchor)
        fill_mask_circular(self.mask_local1, self.watermark_local1)
        fill_mask_circular(self.mask_local2, self.watermark_local2)

        # 32x32 macro-pixel payload -> 64x64 latent-resolution signs.
        m_64 = F.interpolate(sd_tensor_32.float(), scale_factor=self.patch_size, mode="nearest").long()

        # Low-frequency payload + high-frequency key = pseudo-random Gaussian signs.
        target_bits_64 = (m_64 + self.key_64) % 2
        signs_64 = (target_bits_64 * 2 - 1).half()
        z_64 = torch.randn((1, self.channels, w_64_size, w_64_size), dtype=torch.float16, device="cuda")
        w_64 = torch.abs(z_64) * signs_64

        return w_64

    def _decode_mask_circular_from_soft(self, soft_m_32, mask, bits_len, target_watermark):
        """
        Decode one circularly spread bit sequence from a soft 32x32 map.

        This implements the soft LLR accumulation used by the current ANCHOR
        code, then converts it to hard bits and computes the bit matching count.
        """
        if target_watermark is None:
            raise RuntimeError("Watermark has not been generated. Call create_watermark_and_return_w() first.")

        spatial_indices = mask.nonzero(as_tuple=True)
        gathered_soft = []
        for c in range(self.channels):
            gathered_soft.append(soft_m_32[0, c, spatial_indices[0], spatial_indices[1]])
        gathered_soft = torch.cat(gathered_soft)

        target_length = gathered_soft.shape[0]
        extracted_llr = torch.zeros(bits_len, device=gathered_soft.device)
        indices = torch.arange(target_length, device=gathered_soft.device) % bits_len
        extracted_llr.scatter_add_(0, indices, gathered_soft)

        dec_bits = (extracted_llr > 0).long()
        target_bits = target_watermark.to(gathered_soft.device).long()

        match_bits = int((dec_bits == target_bits).sum().item())
        acc = match_bits / float(bits_len)

        return {
            "acc": float(acc),
            "match_bits": match_bits,
            "bits_len": int(bits_len),
            "dec_bits": dec_bits,
            "llr": extracted_llr,
        }

    def eval_global_anchor(self, soft_m_32):
        """Backward-compatible anchor accuracy interface."""
        info = self._decode_mask_circular_from_soft(
            soft_m_32=soft_m_32,
            mask=self.mask_anchor,
            bits_len=self.global_bits_len,
            target_watermark=self.watermark_anchor,
        )
        return info["acc"]

    def eval_global_anchor_detail(self, soft_m_32):
        return self._decode_mask_circular_from_soft(
            soft_m_32=soft_m_32,
            mask=self.mask_anchor,
            bits_len=self.global_bits_len,
            target_watermark=self.watermark_anchor,
        )

    def _unlock_and_pool(self, w_64, key_sign_64):
        unlocked_w_64 = w_64.float() * key_sign_64
        soft_m_32 = F.avg_pool2d(unlocked_w_64, kernel_size=self.patch_size)
        soft_m_32 = (soft_m_32 - soft_m_32.mean()) / (soft_m_32.std() + 1e-6)
        return soft_m_32

    def eval_watermark(self, reversed_w_64):
        """
        Detect anchor and payload with Gaussian Shading / GaussMarker style
        binomial thresholds.

        Returns a dictionary instead of only two accuracies:
            result["final_watermarked"] is the final anchor AND payload decision.
            result["anchor"] contains anchor match count and threshold.
            result["payload"] contains local1/local2 payload decisions.
        """
        if self.key_64 is None:
            raise RuntimeError("Key has not been generated. Call create_watermark_and_return_w() before eval_watermark().")

        restored_w_64 = self.roar_blur(reversed_w_64)

        # Multiplying by key_sign decrypts the sign map before average pooling.
        key_sign_64 = (self.key_64 * -2 + 1).float()

        baseline_soft = self._unlock_and_pool(restored_w_64, key_sign_64)
        baseline_anchor = self.eval_global_anchor_detail(baseline_soft)

        if baseline_anchor["match_bits"] >= self.tau_anchor_bits:
            refined_angle = 0.0
            best_scale = 1.0
            best_anchor = baseline_anchor
            best_source = "baseline"
        else:
            best_anchor = None
            best_angle = 0.0
            best_scale = 1.0
            best_source = "search"
            curves_by_scale = {}

            for scale in self.search_scales:
                curve = []
                for angle in self.search_angles:
                    aligned_w_64 = TF.affine(
                        restored_w_64,
                        angle=-angle,
                        translate=[0, 0],
                        scale=scale,
                        shear=0,
                        interpolation=TF.InterpolationMode.BILINEAR,
                    )

                    soft_m_32 = self._unlock_and_pool(aligned_w_64, key_sign_64)
                    anchor_info = self.eval_global_anchor_detail(soft_m_32)
                    curve.append(anchor_info["acc"])

                    if best_anchor is None or anchor_info["match_bits"] > best_anchor["match_bits"]:
                        best_anchor = anchor_info
                        best_angle = float(angle)
                        best_scale = float(scale)

                curves_by_scale[float(scale)] = curve

            refined_angle = best_angle
            best_idx = self.search_angles.index(int(best_angle))
            best_curve = curves_by_scale.get(float(best_scale), [])
            if best_curve and 0 < best_idx < len(self.search_angles) - 1:
                y_neg = best_curve[best_idx - 1]
                y_0 = best_curve[best_idx]
                y_pos = best_curve[best_idx + 1]
                denom = y_neg - 2 * y_0 + y_pos
                if abs(denom) > 1e-6:
                    refined_angle += ((y_neg - y_pos) / (2 * denom)) * 5.0

            logging.info(
                f"[Sync] 检测尺度: {best_scale} | 矫正角度: {-refined_angle:.2f}° | "
                f"锚点匹配: {best_anchor['match_bits']}/{self.global_bits_len} "
                f"(Acc={best_anchor['acc']:.4f}, tau={self.tau_anchor_bits}/{self.global_bits_len})"
            )

        # Final aligned decoding.
        final_aligned_w_64 = TF.affine(
            restored_w_64,
            angle=-refined_angle,
            translate=[0, 0],
            scale=best_scale,
            shear=0,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        final_soft_m_32 = self._unlock_and_pool(final_aligned_w_64, key_sign_64)

        anchor_info = self._decode_mask_circular_from_soft(
            soft_m_32=final_soft_m_32,
            mask=self.mask_anchor,
            bits_len=self.global_bits_len,
            target_watermark=self.watermark_anchor,
        )
        local1_info = self._decode_mask_circular_from_soft(
            soft_m_32=final_soft_m_32,
            mask=self.mask_local1,
            bits_len=self.local1_bits_len,
            target_watermark=self.watermark_local1,
        )
        local2_info = self._decode_mask_circular_from_soft(
            soft_m_32=final_soft_m_32,
            mask=self.mask_local2,
            bits_len=self.local2_bits_len,
            target_watermark=self.watermark_local2,
        )

        anchor_pass = anchor_info["match_bits"] >= self.tau_anchor_bits
        local1_pass = local1_info["match_bits"] >= self.tau_local1_bits
        local2_pass = local2_info["match_bits"] >= self.tau_local2_bits
        payload_pass = local1_pass or local2_pass

        if local1_info["match_bits"] >= local2_info["match_bits"]:
            best_payload_name = "local1"
            best_payload_info = local1_info
            best_payload_tau_bits = self.tau_local1_bits
            best_payload_tau_acc = self.tau_local1
        else:
            best_payload_name = "local2"
            best_payload_info = local2_info
            best_payload_tau_bits = self.tau_local2_bits
            best_payload_tau_acc = self.tau_local2

        final_watermarked = bool(anchor_pass and payload_pass)

        return {
            "final_watermarked": final_watermarked,
            "sync": {
                "source": best_source,
                "angle": float(refined_angle),
                "scale": float(best_scale),
                "best_anchor_before_final_decode": best_anchor,
            },
            "anchor": {
                "passed": bool(anchor_pass),
                "acc": anchor_info["acc"],
                "match_bits": anchor_info["match_bits"],
                "bits_len": self.global_bits_len,
                "tau_bits": int(self.tau_anchor_bits),
                "tau_acc": float(self.tau_anchor),
                "target_fpr": float(self.fpr_anchor),
                "actual_fpr": float(self.actual_fpr_anchor),
                "n_tests": int(self.anchor_search_trials),
            },
            "payload": {
                "passed": bool(payload_pass),
                "best_name": best_payload_name,
                "best_acc": best_payload_info["acc"],
                "best_match_bits": best_payload_info["match_bits"],
                "best_tau_bits": int(best_payload_tau_bits),
                "best_tau_acc": float(best_payload_tau_acc),
                "target_fpr": float(self.fpr_payload),
                "n_tests": int(self.payload_total_tests),
                "local1": {
                    "passed": bool(local1_pass),
                    "acc": local1_info["acc"],
                    "match_bits": local1_info["match_bits"],
                    "bits_len": self.local1_bits_len,
                    "tau_bits": int(self.tau_local1_bits),
                    "tau_acc": float(self.tau_local1),
                    "actual_fpr": float(self.actual_fpr_local1),
                },
                "local2": {
                    "passed": bool(local2_pass),
                    "acc": local2_info["acc"],
                    "match_bits": local2_info["match_bits"],
                    "bits_len": self.local2_bits_len,
                    "tau_bits": int(self.tau_local2_bits),
                    "tau_acc": float(self.tau_local2),
                    "actual_fpr": float(self.actual_fpr_local2),
                },
            },
        }

    def eval_watermark_legacy(self, reversed_w_64):
        """Compatibility helper for older scripts expecting two float accuracies."""
        result = self.eval_watermark(reversed_w_64)
        return result["anchor"]["acc"], result["payload"]["best_acc"]
