import math
import warnings
import logging
import hashlib
import hmac
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from scipy.stats import norm, truncnorm, binom
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


@dataclass
class ProcrustesAnchorConfig:
    logical_size: int = 32
    num_landmarks: int = 64
    landmark_patch: int = 3
    min_landmarks_for_sync: int = 10
    landmark_key_mode: str = "payload_hmac"
    use_procrustes_sync: bool = True
    min_landmark_distance: int = 3
    search_radius: int = 16
    max_sync_rmse: float = 1.5
    min_inlier_ratio_for_sync: float = 0.18
    landmark_threshold_cap: float = 0.45
    use_ransac: bool = True
    ransac_trials: int = 128
    ransac_inlier_threshold: float = 2.0
    sync_angle_min: int = -180
    sync_angle_max: int = 180
    sync_angle_step: int = 5
    sync_candidate_topk: int = 16
    use_payload_sync_validation: bool = True
    secret_key: bytes = b"anchor-procrustes-v1"


def _bits_to_bytes(bits):
    """Pack a 1D bit tensor/array/list into bytes for keyed hashing."""
    if isinstance(bits, torch.Tensor):
        values = bits.detach().flatten().long().cpu().numpy().astype(np.uint8)
    else:
        values = np.asarray(bits, dtype=np.uint8).reshape(-1) & 1
    if values.size == 0:
        return b""
    packed = np.packbits(values, bitorder="big")
    return packed.tobytes()


def _int_to_bytes(value: int, length: int = 8):
    return int(value).to_bytes(length, byteorder="big", signed=False)


def generate_payload_hmac_bits(payload_bits, key, nonce, n_bits):
    """
    Generate payload-bound anchor bits with HMAC-SHA256.

    The anchor is reproducible from payload/key/nonce and therefore acts as
    both a synchronization template and a lightweight payload authenticator.
    """
    if isinstance(key, str):
        key = key.encode("utf-8")
    if isinstance(nonce, torch.Tensor):
        nonce_bytes = _bits_to_bytes(nonce)
    elif isinstance(nonce, bytes):
        nonce_bytes = nonce
    else:
        nonce_bytes = _int_to_bytes(int(nonce))

    payload_bytes = _bits_to_bytes(payload_bits)
    out = []
    counter = 0
    while len(out) < int(n_bits):
        msg = payload_bytes + nonce_bytes + _int_to_bytes(counter)
        digest = hmac.new(key, msg, hashlib.sha256).digest()
        digest_bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8), bitorder="big")
        out.extend(int(x) for x in digest_bits)
        counter += 1
    return torch.tensor(out[:int(n_bits)], dtype=torch.long)


def sample_spatially_diverse_landmarks(
    logical_size,
    num_landmarks,
    min_distance,
    seed,
    allowed_mask=None,
    patch_size=3,
):
    """
    Keyed farthest-point sampling on the logical grid.

    The first candidate is random, then each next point maximizes its minimum
    distance to previously selected points. This keeps anchors globally spread,
    which directly reduces the scale-estimation variance in the paper theory.
    """
    rng = np.random.default_rng(int(seed) % (2 ** 32))
    margin = int(patch_size) // 2
    candidates = []
    if allowed_mask is not None:
        if isinstance(allowed_mask, torch.Tensor):
            allowed = allowed_mask.detach().cpu().numpy().astype(bool)
        else:
            allowed = np.asarray(allowed_mask, dtype=bool)
    else:
        allowed = np.ones((logical_size, logical_size), dtype=bool)

    for r in range(margin, int(logical_size) - margin):
        for c in range(margin, int(logical_size) - margin):
            if not allowed[r, c]:
                continue
            patch = allowed[r - margin:r + margin + 1, c - margin:c + margin + 1]
            if patch.shape == (patch_size, patch_size) and patch.all():
                candidates.append((float(c), float(r)))

    if len(candidates) < int(num_landmarks):
        raise ValueError(
            f"Not enough candidate cells for {num_landmarks} landmarks; found {len(candidates)}."
        )

    candidates = np.asarray(candidates, dtype=np.float32)
    first = int(rng.integers(0, len(candidates)))
    selected = [first]
    selected_points = [candidates[first]]
    min_sq_dist = np.sum((candidates - candidates[first]) ** 2, axis=1)

    while len(selected) < int(num_landmarks):
        jitter = rng.random(len(candidates)) * 1e-6
        farthest = int(np.argmax(min_sq_dist + jitter))
        if math.sqrt(float(min_sq_dist[farthest])) < float(min_distance):
            break
        selected.append(farthest)
        selected_points.append(candidates[farthest])
        dist = np.sum((candidates - candidates[farthest]) ** 2, axis=1)
        min_sq_dist = np.minimum(min_sq_dist, dist)

    if len(selected_points) < int(num_landmarks):
        remaining = [i for i in range(len(candidates)) if i not in selected]
        rng.shuffle(remaining)
        for idx in remaining[: int(num_landmarks) - len(selected_points)]:
            selected_points.append(candidates[idx])

    return torch.tensor(np.asarray(selected_points[:int(num_landmarks)]), dtype=torch.float32)


def build_landmark_codebook(sd_tensor_32, landmark_points, patch_size):
    """Read local expected sign codes around each landmark from the embedded bit map."""
    half = int(patch_size) // 2
    codes = []
    for point in landmark_points:
        c = int(round(float(point[0])))
        r = int(round(float(point[1])))
        patch = sd_tensor_32[
            0,
            :,
            r - half:r + half + 1,
            c - half:c + half + 1,
        ]
        if patch.shape[-2:] != (patch_size, patch_size):
            raise ValueError("Landmark patch is out of bounds.")
        codes.append((patch.float() * 2.0 - 1.0).flatten())
    return torch.stack(codes, dim=0)


def write_landmark_codebook(sd_tensor_32, landmark_points, patch_size, code_bits):
    """Write landmark-specific local codes into the logical bit map."""
    half = int(patch_size) // 2
    n_landmarks = int(landmark_points.shape[0])
    expected = n_landmarks * sd_tensor_32.shape[1] * int(patch_size) * int(patch_size)
    if int(code_bits.numel()) < expected:
        raise ValueError("Not enough code bits for landmark codebook.")

    idx = 0
    for point in landmark_points:
        c = int(round(float(point[0])))
        r = int(round(float(point[1])))
        patch_bits = code_bits[idx:idx + sd_tensor_32.shape[1] * patch_size * patch_size]
        patch_bits = patch_bits.view(sd_tensor_32.shape[1], patch_size, patch_size).to(sd_tensor_32.device)
        sd_tensor_32[
            0,
            :,
            r - half:r + half + 1,
            c - half:c + half + 1,
        ] = patch_bits.long()
        idx += sd_tensor_32.shape[1] * patch_size * patch_size


def zero_key_on_landmarks(key_64, landmark_points, patch_size, latent_patch_size):
    """
    Reserve landmark patches from the one-time-pad sign key.

    Geometry sync must be detectable before spatial alignment is known. If the
    same high-frequency key is applied there, any scale/crop attack makes the
    pre-alignment landmark response vanish. Zeroing the key only on small,
    pseudo-random landmark patches keeps those sync codes visible while the
    payload remains keyed.
    """
    half = int(patch_size) // 2
    for point in landmark_points:
        c = int(round(float(point[0])))
        r = int(round(float(point[1])))
        r0 = (r - half) * latent_patch_size
        r1 = (r + half + 1) * latent_patch_size
        c0 = (c - half) * latent_patch_size
        c1 = (c + half + 1) * latent_patch_size
        key_64[:, :, r0:r1, c0:c1] = 0


def weighted_procrustes_2d(src_points, dst_points, weights=None, eps=1e-8):
    """
    Closed-form weighted 2D similarity alignment.

    Returns a transform mapping src_points to dst_points:
        dst ~= scale * R @ src + translation
    """
    src = torch.as_tensor(src_points, dtype=torch.float32)
    dst = torch.as_tensor(dst_points, dtype=torch.float32, device=src.device)
    if src.ndim != 2 or src.shape[1] != 2 or dst.shape != src.shape:
        raise ValueError("src_points and dst_points must both have shape [N, 2].")
    if src.shape[0] < 3:
        raise ValueError("At least 3 points are required for Procrustes alignment.")

    if weights is None:
        w = torch.ones(src.shape[0], dtype=torch.float32, device=src.device)
    else:
        w = torch.as_tensor(weights, dtype=torch.float32, device=src.device).flatten()
        if w.shape[0] != src.shape[0]:
            raise ValueError("weights must have length N.")
        w = torch.clamp(w, min=0.0)

    weight_sum = w.sum()
    if float(weight_sum.item()) <= eps:
        raise ValueError("weights must contain positive mass.")
    w_norm = w / weight_sum

    src_mean = (w_norm[:, None] * src).sum(dim=0)
    dst_mean = (w_norm[:, None] * dst).sum(dim=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    src_energy = (w * (src_centered ** 2).sum(dim=1)).sum()
    if float(src_energy.item()) <= eps:
        raise ValueError("Source landmarks are degenerate.")

    h_mat = src_centered.t().matmul(w[:, None] * dst_centered)
    u_mat, singular_values, vh_mat = torch.linalg.svd(h_mat, full_matrices=False)
    v_mat = vh_mat.t()
    det = torch.det(v_mat.matmul(u_mat.t()))
    correction = torch.diag(torch.tensor([1.0, torch.sign(det).item()], device=src.device))
    rotation = v_mat.matmul(correction).matmul(u_mat.t())
    scale = (singular_values * torch.diag(correction)).sum() / src_energy
    translation = dst_mean - scale * rotation.matmul(src_mean)

    predicted = scale * src.matmul(rotation.t()) + translation
    residual = dst - predicted
    rmse = torch.sqrt((w_norm * (residual ** 2).sum(dim=1)).sum())
    angle = torch.atan2(rotation[1, 0], rotation[0, 0]) * 180.0 / math.pi

    return {
        "scale": float(scale.item()),
        "rotation": rotation,
        "angle": float(angle.item()),
        "translation": translation,
        "rmse": float(rmse.item()),
        "singular_values": singular_values,
    }


def ransac_weighted_procrustes_2d(
    src_points,
    dst_points,
    weights=None,
    min_inliers=6,
    trials=128,
    inlier_threshold=2.0,
):
    """
    Robust Procrustes alignment with a small RANSAC loop.

    Landmark detection can produce high local correlations at wrong positions
    after scale/crop. RANSAC keeps the geometrically consistent subset before
    the final weighted least-squares fit.
    """
    src = torch.as_tensor(src_points, dtype=torch.float32)
    dst = torch.as_tensor(dst_points, dtype=torch.float32, device=src.device)
    if weights is None:
        w = torch.ones(src.shape[0], dtype=torch.float32, device=src.device)
    else:
        w = torch.as_tensor(weights, dtype=torch.float32, device=src.device).flatten()

    n_points = int(src.shape[0])
    if n_points < 3:
        raise ValueError("At least 3 points are required for RANSAC Procrustes.")

    best = None
    best_mask = None
    full_indices = torch.arange(n_points, device=src.device)

    # Include the all-point fit as a fallback candidate.
    candidate_sets = [full_indices]
    max_trials = max(0, int(trials))
    top_k = min(n_points, max(int(min_inliers) * 4, 12))
    top_indices = torch.argsort(w, descending=True)[:top_k]
    for _ in range(max_trials):
        perm = torch.randperm(top_k, device=src.device)[:3]
        candidate_sets.append(top_indices[perm])

    for subset in candidate_sets:
        try:
            estimate = weighted_procrustes_2d(src[subset], dst[subset], w[subset])
        except ValueError:
            continue

        scale = float(estimate["scale"])
        rot = estimate["rotation"].to(src.device)
        trans = estimate["translation"].to(src.device)
        pred = scale * src.matmul(rot.t()) + trans
        residual = torch.linalg.norm(dst - pred, dim=1)
        inlier_mask = residual <= float(inlier_threshold)
        inlier_count = int(inlier_mask.sum().item())
        inlier_weight = float(w[inlier_mask].sum().item()) if inlier_count > 0 else 0.0
        median_residual = float(torch.median(residual).item())
        score = (inlier_count, inlier_weight, -median_residual)

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "estimate": estimate,
                "residual": residual,
                "inlier_count": inlier_count,
                "inlier_weight": inlier_weight,
            }
            best_mask = inlier_mask

    if best is None or best_mask is None:
        raise ValueError("RANSAC Procrustes failed to find a valid candidate.")

    if int(best_mask.sum().item()) >= max(3, int(min_inliers)):
        estimate = weighted_procrustes_2d(src[best_mask], dst[best_mask], w[best_mask])
        estimate["inlier_count"] = int(best_mask.sum().item())
        estimate["inlier_ratio"] = float(best_mask.float().mean().item())
        estimate["ransac_used"] = True
        return estimate

    estimate = best["estimate"]
    estimate["inlier_count"] = int(best["inlier_count"])
    estimate["inlier_ratio"] = float(best["inlier_count"]) / float(n_points)
    estimate["ransac_used"] = True
    return estimate


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
        procrustes_config=None,
    ):
        self.ch_factor = ch_factor
        self.hw_factor = hw_factor
        self.fpr = float(fpr)
        self.user_number = max(1, int(user_number))

        self.patch_size = 2
        self.logical_size = 64 // self.patch_size  # 32x32 macro-pixels
        self.channels = 4
        self.logical_length = self.channels * self.logical_size * self.logical_size
        if procrustes_config is None:
            procrustes_config = ProcrustesAnchorConfig(logical_size=self.logical_size)
        self.procrustes_config = procrustes_config
        self.use_procrustes_sync = bool(self.procrustes_config.use_procrustes_sync)

        # Search space used in anchor synchronization. The threshold must count
        # all tested candidates because eval_watermark selects the best match.
        angle_step = max(1, int(self.procrustes_config.sync_angle_step))
        self.search_angles = list(range(
            int(self.procrustes_config.sync_angle_min),
            int(self.procrustes_config.sync_angle_max) + 1,
            angle_step,
        ))
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
        self.anchor_bit_index_32 = None
        self.local1_bit_index_32 = None
        self.local2_bit_index_32 = None
        self.anchor_nonce = None
        self.landmark_points = None
        self.landmark_codebook = None
        self.landmark_threshold = None
        self.landmark_single_fpr = None
        self.landmark_total_tests = None

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

        self.payload_combined_total_tests = self.user_number
        self.tau_payload_combined_bits, self.tau_payload_combined, self.actual_fpr_payload_combined, _ = \
            binomial_threshold_bits(
                k=self.local1_bits_len,
                target_fpr=self.fpr_payload,
                n_tests=self.payload_combined_total_tests,
            )

        cfg = self.procrustes_config
        landmark_window = (2 * int(cfg.search_radius) + 1) ** 2
        self.landmark_total_tests = max(
            1,
            int(cfg.num_landmarks) * landmark_window * max(1, len(self.search_angles)),
        )
        self.landmark_single_fpr = _familywise_to_single_test_fpr(
            self.fpr_anchor,
            self.landmark_total_tests,
        )
        landmark_code_len = self.channels * int(cfg.landmark_patch) * int(cfg.landmark_patch)
        # This is a candidate-generation threshold, not the final system
        # detector. RANSAC geometry and payload-bound bit tests provide the
        # high-precision stage; keeping this gate slightly permissive prevents
        # hard images from losing too many landmarks before geometry filtering.
        self.landmark_threshold = float(
            min(
                norm.isf(self.landmark_single_fpr) / math.sqrt(landmark_code_len),
                float(cfg.landmark_threshold_cap),
            )
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
            "procrustes_landmarks": {
                "enabled": bool(self.use_procrustes_sync),
                "num_landmarks": int(self.procrustes_config.num_landmarks),
                "patch": int(self.procrustes_config.landmark_patch),
                "min_landmarks_for_sync": int(self.procrustes_config.min_landmarks_for_sync),
                "response_threshold": float(self.landmark_threshold),
                "n_tests": int(self.landmark_total_tests),
                "single_test_fpr": float(self.landmark_single_fpr),
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
            "payload_combined": {
                "bits": self.local1_bits_len,
                "tau_bits": int(self.tau_payload_combined_bits),
                "tau_acc": float(self.tau_payload_combined),
                "target_fpr": float(self.fpr_payload),
                "actual_fpr": float(self.actual_fpr_payload_combined),
                "n_tests": int(self.payload_combined_total_tests),
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

        self.watermark_local1 = torch.randint(0, 2, [self.local1_bits_len]).cuda()
        # local1/local2 are robust replicas of the same payload. This turns the
        # two spatial partitions into soft-LLR diversity branches instead of two
        # unrelated messages, which is the statistically correct use for
        # ownership detection under crop/scale attacks.
        self.watermark_local2 = self.watermark_local1.clone()
        payload_for_anchor = torch.cat([self.watermark_local1, self.watermark_local2], dim=0)
        self.anchor_nonce = int(torch.randint(0, 2 ** 31 - 1, [1]).item())
        self.watermark_anchor = generate_payload_hmac_bits(
            payload_bits=payload_for_anchor,
            key=self.procrustes_config.secret_key,
            nonce=self.anchor_nonce,
            n_bits=self.global_bits_len,
        ).cuda()

        sd_tensor_32 = torch.zeros((1, self.channels, self.logical_size, self.logical_size), dtype=torch.long).cuda()
        self.anchor_bit_index_32 = torch.full(
            (self.channels, self.logical_size, self.logical_size),
            -1,
            dtype=torch.long,
            device="cuda",
        )
        self.local1_bit_index_32 = torch.full_like(self.anchor_bit_index_32, -1)
        self.local2_bit_index_32 = torch.full_like(self.anchor_bit_index_32, -1)

        def fill_mask_circular(mask, watermark_bits, bit_index_32):
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
                bit_index_32[c, spatial_indices[0], spatial_indices[1]] = \
                    torch.arange(idx, idx + pixels_in_mask, device=bit_index_32.device) % bits_len
                idx += pixels_in_mask

        fill_mask_circular(self.mask_anchor, self.watermark_anchor, self.anchor_bit_index_32)
        fill_mask_circular(self.mask_local1, self.watermark_local1, self.local1_bit_index_32)
        fill_mask_circular(self.mask_local2, self.watermark_local2, self.local2_bit_index_32)

        if self.use_procrustes_sync:
            cfg = self.procrustes_config
            landmark_seed_bits = generate_payload_hmac_bits(
                payload_bits=payload_for_anchor,
                key=cfg.secret_key,
                nonce=self.anchor_nonce + 1,
                n_bits=64,
            )
            landmark_seed = int.from_bytes(_bits_to_bytes(landmark_seed_bits)[:8], "big")
            self.landmark_points = sample_spatially_diverse_landmarks(
                logical_size=self.logical_size,
                num_landmarks=cfg.num_landmarks,
                min_distance=cfg.min_landmark_distance,
                seed=landmark_seed,
                allowed_mask=self.mask_anchor,
                patch_size=cfg.landmark_patch,
            ).cuda()
            landmark_code_bits = generate_payload_hmac_bits(
                payload_bits=payload_for_anchor,
                key=cfg.secret_key,
                nonce=self.anchor_nonce + 2,
                n_bits=int(cfg.num_landmarks) * self.channels * int(cfg.landmark_patch) * int(cfg.landmark_patch),
            ).cuda()
            write_landmark_codebook(
                sd_tensor_32=sd_tensor_32,
                landmark_points=self.landmark_points,
                patch_size=cfg.landmark_patch,
                code_bits=landmark_code_bits,
            )
            self._invalidate_anchor_landmark_indices()
            self.landmark_codebook = build_landmark_codebook(
                sd_tensor_32=sd_tensor_32,
                landmark_points=self.landmark_points,
                patch_size=cfg.landmark_patch,
            ).cuda()

        # 32x32 macro-pixel payload -> 64x64 latent-resolution signs.
        m_64 = F.interpolate(sd_tensor_32.float(), scale_factor=self.patch_size, mode="nearest").long()

        if self.use_procrustes_sync and self.landmark_points is not None:
            zero_key_on_landmarks(
                key_64=self.key_64,
                landmark_points=self.landmark_points,
                patch_size=self.procrustes_config.landmark_patch,
                latent_patch_size=self.patch_size,
            )

        # Low-frequency payload + high-frequency key = pseudo-random Gaussian signs.
        target_bits_64 = (m_64 + self.key_64) % 2
        signs_64 = (target_bits_64 * 2 - 1).half()
        z_64 = torch.randn((1, self.channels, w_64_size, w_64_size), dtype=torch.float16, device="cuda")
        w_64 = torch.abs(z_64) * signs_64

        return w_64

    def _invalidate_anchor_landmark_indices(self):
        """Remove overwritten landmark patches from bit-level anchor decoding."""
        if self.anchor_bit_index_32 is None or self.landmark_points is None:
            return
        half = int(self.procrustes_config.landmark_patch) // 2
        for point in self.landmark_points:
            c = int(round(float(point[0])))
            r = int(round(float(point[1])))
            self.anchor_bit_index_32[
                :,
                r - half:r + half + 1,
                c - half:c + half + 1,
            ] = -1

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

    def _decode_indexed_from_soft(
        self,
        soft_m_32,
        bit_index_32,
        bits_len,
        target_watermark,
        target_fpr=None,
        n_tests=None,
        valid_mask_32=None,
    ):
        """
        Decode bits with an explicit embedding index map.

        This keeps anchor decoding valid after landmark patches overwrite part
        of the anchor mask. Removed cells are marked as -1 and do not shift the
        circular repetition index of the remaining anchor observations.
        """
        if target_watermark is None:
            raise RuntimeError("Watermark has not been generated. Call create_watermark_and_return_w() first.")
        if bit_index_32 is None:
            raise RuntimeError("Bit index map has not been generated.")

        bit_index_32 = bit_index_32.to(soft_m_32.device)
        valid = bit_index_32 >= 0
        if valid_mask_32 is not None:
            valid_mask_32 = valid_mask_32.to(soft_m_32.device).bool()
            valid = valid & valid_mask_32.unsqueeze(0)
        if not bool(valid.any().item()):
            return {
                "acc": 0.0,
                "match_bits": 0,
                "bits_len": int(bits_len),
                "effective_bits_len": 0,
                "tau_bits_effective": int(bits_len) + 1,
                "dec_bits": torch.zeros(bits_len, dtype=torch.long, device=soft_m_32.device),
                "llr": torch.zeros(bits_len, device=soft_m_32.device),
                "counts": torch.zeros(bits_len, device=soft_m_32.device),
            }

        soft_values = soft_m_32[0][valid]
        indices = bit_index_32[valid].long()
        extracted_llr = torch.zeros(bits_len, device=soft_m_32.device)
        counts = torch.zeros(bits_len, device=soft_m_32.device)
        extracted_llr.scatter_add_(0, indices, soft_values)
        counts.scatter_add_(0, indices, torch.ones_like(soft_values))

        observed_bits = counts > 0
        dec_bits = (extracted_llr > 0).long()
        target_bits = target_watermark.to(soft_m_32.device).long()

        match_bits = int((dec_bits[observed_bits] == target_bits[observed_bits]).sum().item())
        effective_bits_len = int(observed_bits.sum().item())
        acc = match_bits / float(effective_bits_len) if effective_bits_len > 0 else 0.0
        if target_fpr is None:
            target_fpr = self.fpr_anchor
        if n_tests is None:
            n_tests = self.anchor_search_trials
        tau_bits_effective, tau_acc_effective, actual_fpr_effective, _ = binomial_threshold_bits(
            k=max(effective_bits_len, 1),
            target_fpr=target_fpr,
            n_tests=n_tests,
        )

        return {
            "acc": float(acc),
            "match_bits": match_bits,
            "bits_len": int(bits_len),
            "effective_bits_len": effective_bits_len,
            "tau_bits_effective": int(tau_bits_effective),
            "tau_acc_effective": float(tau_acc_effective),
            "actual_fpr_effective": float(actual_fpr_effective),
            "dec_bits": dec_bits,
            "llr": extracted_llr,
            "counts": counts,
        }

    def _combine_payload_replicas(self, local1_info, local2_info):
        if self.watermark_local1 is None:
            raise RuntimeError("Payload watermark has not been generated.")

        combined_llr = local1_info["llr"] + local2_info["llr"]
        combined_counts = local1_info.get("counts", torch.ones_like(combined_llr)) + \
            local2_info.get("counts", torch.ones_like(combined_llr))
        observed_bits = combined_counts > 0
        dec_bits = (combined_llr > 0).long()
        target_bits = self.watermark_local1.to(combined_llr.device).long()
        match_bits = int((dec_bits[observed_bits] == target_bits[observed_bits]).sum().item())
        bits_len = int(self.local1_bits_len)
        effective_bits_len = int(observed_bits.sum().item())
        acc = match_bits / float(effective_bits_len) if effective_bits_len > 0 else 0.0
        tau_bits, tau_acc, actual_fpr, _ = binomial_threshold_bits(
            k=max(effective_bits_len, 1),
            target_fpr=self.fpr_payload,
            n_tests=self.payload_combined_total_tests,
        )
        passed = match_bits >= int(tau_bits)
        return {
            "passed": bool(passed),
            "acc": float(acc),
            "match_bits": match_bits,
            "bits_len": bits_len,
            "effective_bits_len": effective_bits_len,
            "tau_bits": int(tau_bits),
            "tau_acc": float(tau_acc),
            "actual_fpr": float(actual_fpr),
            "dec_bits": dec_bits,
            "llr": combined_llr,
            "counts": combined_counts,
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
        if self.anchor_bit_index_32 is not None:
            return self._decode_indexed_from_soft(
                soft_m_32=soft_m_32,
                bit_index_32=self.anchor_bit_index_32,
                bits_len=self.global_bits_len,
                target_watermark=self.watermark_anchor,
            )
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

    def _raw_pool(self, w_64):
        soft_m_32 = F.avg_pool2d(w_64.float(), kernel_size=self.patch_size)
        soft_m_32 = (soft_m_32 - soft_m_32.mean()) / (soft_m_32.std() + 1e-6)
        return soft_m_32

    def _detect_landmarks_from_soft(self, soft_m_32):
        if self.landmark_points is None or self.landmark_codebook is None:
            return {
                "passed": False,
                "num_detected": 0,
                "src_points": None,
                "dst_points": None,
                "weights": None,
                "mean_score": 0.0,
                "threshold": float(self.landmark_threshold),
            }

        cfg = self.procrustes_config
        half = int(cfg.landmark_patch) // 2
        radius = int(cfg.search_radius)
        height = soft_m_32.shape[-2]
        width = soft_m_32.shape[-1]
        device = soft_m_32.device

        points = self.landmark_points.to(device).float()
        codebook = self.landmark_codebook.to(device).float()

        # Vectorized landmark matching. The previous implementation looped over
        # landmark x window candidates and called .item() for every patch, which
        # forced millions of tiny GPU/CPU synchronizations per image. unfold
        # builds all local patches once, then a single matrix multiply scores all
        # landmark codes against all candidate centers on GPU.
        patches = F.unfold(
            soft_m_32.float(),
            kernel_size=int(cfg.landmark_patch),
            padding=half,
        )[0]  # [C * patch * patch, H * W]
        scores_all = codebook.matmul(patches) / float(codebook.shape[1])

        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        flat_x = xx.flatten()
        flat_y = yy.flatten()
        point_x = points[:, 0:1]
        point_y = points[:, 1:2]

        search_mask = (
            (torch.abs(flat_x.unsqueeze(0) - point_x) <= float(radius))
            & (torch.abs(flat_y.unsqueeze(0) - point_y) <= float(radius))
            & (flat_x.unsqueeze(0) >= float(half))
            & (flat_x.unsqueeze(0) <= float(width - half - 1))
            & (flat_y.unsqueeze(0) >= float(half))
            & (flat_y.unsqueeze(0) <= float(height - half - 1))
        )
        scores_all = scores_all.masked_fill(~search_mask, -1e9)
        best_scores, best_indices = torch.max(scores_all, dim=1)
        keep = best_scores >= float(self.landmark_threshold)

        num_detected = int(keep.sum().item())
        src_points = points[keep]
        dst_points = torch.stack(
            [flat_x[best_indices[keep]], flat_y[best_indices[keep]]],
            dim=1,
        ) if num_detected > 0 else None
        weights = torch.clamp(best_scores[keep] - float(self.landmark_threshold), min=1e-3) \
            if num_detected > 0 else None
        mean_score = float(best_scores[keep].mean().item()) if num_detected > 0 else 0.0

        passed = num_detected >= int(cfg.min_landmarks_for_sync)
        return {
            "passed": bool(passed),
            "num_detected": int(num_detected),
            "src_points": src_points if num_detected > 0 else None,
            "dst_points": dst_points if num_detected > 0 else None,
            "weights": weights if num_detected > 0 else None,
            "mean_score": mean_score,
            "threshold": float(self.landmark_threshold),
        }

    def _estimate_procrustes_from_soft(self, soft_m_32):
        detected = self._detect_landmarks_from_soft(soft_m_32)
        if not detected["passed"]:
            return {
                "passed": False,
                "reason": "not_enough_landmarks",
                "landmark_detection": detected,
            }

        try:
            if self.procrustes_config.use_ransac:
                estimate = ransac_weighted_procrustes_2d(
                    detected["src_points"],
                    detected["dst_points"],
                    detected["weights"],
                    min_inliers=self.procrustes_config.min_landmarks_for_sync,
                    trials=self.procrustes_config.ransac_trials,
                    inlier_threshold=self.procrustes_config.ransac_inlier_threshold,
                )
            else:
                estimate = weighted_procrustes_2d(
                    detected["src_points"],
                    detected["dst_points"],
                    detected["weights"],
                )
        except ValueError as exc:
            return {
                "passed": False,
                "reason": str(exc),
                "landmark_detection": detected,
            }

        inlier_count = int(estimate.get("inlier_count", detected["num_detected"]))
        inlier_ratio = float(estimate.get("inlier_ratio", 1.0))
        passed = (
            estimate["rmse"] <= float(self.procrustes_config.max_sync_rmse)
            and inlier_count >= int(self.procrustes_config.min_landmarks_for_sync)
            and inlier_ratio >= float(self.procrustes_config.min_inlier_ratio_for_sync)
        )
        return {
            "passed": bool(passed),
            "reason": "ok" if passed else "high_rmse",
            "landmark_detection": detected,
            "scale": float(estimate["scale"]),
            "angle": float(estimate["angle"]),
            "translation": [
                float(estimate["translation"][0].item()),
                float(estimate["translation"][1].item()),
            ],
            "rmse": float(estimate["rmse"]),
            "inlier_count": int(inlier_count),
            "inlier_ratio": float(inlier_ratio),
            "ransac_used": bool(estimate.get("ransac_used", False)),
        }

    def _procrustes_sync_search(self, restored_w_64, key_sign_64):
        """
        Estimate sync by scanning only rotation and solving scale/translation
        in closed form with Procrustes landmarks.
        """
        zero_angle_soft = self._raw_pool(restored_w_64)
        zero_angle_estimate = self._estimate_procrustes_from_soft(zero_angle_soft)
        strong_zero_angle = (
            zero_angle_estimate.get("passed", False)
            and int(zero_angle_estimate.get("inlier_count", 0)) >= max(18, 2 * int(self.procrustes_config.min_landmarks_for_sync))
            and float(zero_angle_estimate.get("rmse", 1e9)) <= 1.0
            and abs(float(zero_angle_estimate.get("angle", 0.0))) <= 10.0
        )
        if strong_zero_angle:
            return {
                "passed": True,
                "source": "procrustes",
                "angle": float(zero_angle_estimate["angle"]),
                "coarse_angle": 0.0,
                "residual_angle": float(zero_angle_estimate["angle"]),
                "scale": float(zero_angle_estimate["scale"]),
                "translation": zero_angle_estimate["translation"],
                "rmse": float(zero_angle_estimate["rmse"]),
                "inlier_count": int(zero_angle_estimate.get("inlier_count", 0)),
                "inlier_ratio": float(zero_angle_estimate.get("inlier_ratio", 0.0)),
                "ransac_used": bool(zero_angle_estimate.get("ransac_used", False)),
                "landmarks": zero_angle_estimate["landmark_detection"],
                "fast_zero_angle": True,
            }, zero_angle_soft

        candidates = []
        for coarse_angle in self.search_angles:
            angle_aligned = TF.affine(
                restored_w_64,
                angle=-float(coarse_angle),
                translate=[0, 0],
                scale=1.0,
                shear=0,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            # Landmark patches are deliberately left unmasked by key_64, so
            # they can be detected before the unknown scale/translation is
            # estimated. Payload decoding still uses _unlock_and_pool later.
            soft_m_32 = self._raw_pool(angle_aligned)
            estimate = self._estimate_procrustes_from_soft(soft_m_32)
            detected = estimate.get("landmark_detection", {})
            score = (
                int(estimate.get("passed", False)),
                int(estimate.get("inlier_count", 0)),
                float(estimate.get("inlier_ratio", 0.0)),
                -float(estimate.get("rmse", 1e9)),
                detected.get("mean_score", 0.0),
            )
            candidates.append({
                "score": score,
                "coarse_angle": float(coarse_angle),
                "estimate": estimate,
                "soft": soft_m_32,
            })

        if not candidates:
            return {
                "passed": False,
                "source": "procrustes",
                "reason": "no_candidates",
            }, None

        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        if self.procrustes_config.use_payload_sync_validation:
            topk = max(1, int(self.procrustes_config.sync_candidate_topk))
            best = self._select_sync_candidate_by_payload(
                restored_w_64=restored_w_64,
                key_sign_64=key_sign_64,
                candidates=candidates[:topk],
            )

        estimate = best["estimate"]
        if not estimate.get("passed", False):
            return {
                "passed": False,
                "source": "procrustes",
                "coarse_angle": float(best["coarse_angle"]),
                "reason": estimate.get("reason", "failed"),
                "landmarks": estimate.get("landmark_detection", {}),
            }, best.get("soft", None)

        total_angle = float(best["coarse_angle"]) + float(estimate["angle"])
        sync_result = {
            "passed": True,
            "source": "procrustes",
            "angle": total_angle,
            "coarse_angle": float(best["coarse_angle"]),
            "residual_angle": float(estimate["angle"]),
            "scale": float(estimate["scale"]),
            "translation": estimate["translation"],
            "rmse": float(estimate["rmse"]),
            "inlier_count": int(estimate.get("inlier_count", 0)),
            "inlier_ratio": float(estimate.get("inlier_ratio", 0.0)),
            "ransac_used": bool(estimate.get("ransac_used", False)),
            "landmarks": estimate["landmark_detection"],
        }
        if "payload_validation" in best:
            sync_result["payload_validation"] = best["payload_validation"]
        return sync_result, best.get("soft", None)

    def _make_sync_result_from_candidate(self, candidate):
        estimate = candidate["estimate"]
        total_angle = float(candidate["coarse_angle"]) + float(estimate.get("angle", 0.0))
        return {
            "passed": bool(estimate.get("passed", False)),
            "source": "procrustes",
            "angle": total_angle,
            "coarse_angle": float(candidate["coarse_angle"]),
            "residual_angle": float(estimate.get("angle", 0.0)),
            "scale": float(estimate.get("scale", 1.0)),
            "translation": estimate.get("translation", [0.0, 0.0]),
            "rmse": float(estimate.get("rmse", 1e9)),
            "inlier_count": int(estimate.get("inlier_count", 0)),
            "inlier_ratio": float(estimate.get("inlier_ratio", 0.0)),
            "ransac_used": bool(estimate.get("ransac_used", False)),
            "landmarks": estimate.get("landmark_detection", {}),
        }

    def _select_sync_candidate_by_payload(self, restored_w_64, key_sign_64, candidates):
        """
        Use payload soft-LLR as a second-stage synchronization validator.

        Under heavy scale/crop, RANSAC can find a geometrically consistent but
        wrong small subset of landmarks. The correct transform should also make
        the payload replicas agree with the expected payload bits, so we score a
        few top geometric candidates by their combined payload match count.
        """
        best = None
        for candidate in candidates:
            if not candidate["estimate"].get("passed", False):
                validation = {
                    "match_bits": -1,
                    "acc": 0.0,
                    "effective_bits_len": 0,
                }
            else:
                sync_result = self._make_sync_result_from_candidate(candidate)
                try:
                    aligned_w_64, valid_mask_32 = self._align_latent_with_procrustes(
                        restored_w_64,
                        sync_result,
                    )
                    soft_m_32 = self._unlock_and_pool(aligned_w_64, key_sign_64)
                    local1_info = self._decode_indexed_from_soft(
                        soft_m_32=soft_m_32,
                        bit_index_32=self.local1_bit_index_32,
                        bits_len=self.local1_bits_len,
                        target_watermark=self.watermark_local1,
                        target_fpr=self.fpr_payload,
                        n_tests=self.payload_total_tests,
                        valid_mask_32=valid_mask_32,
                    )
                    local2_info = self._decode_indexed_from_soft(
                        soft_m_32=soft_m_32,
                        bit_index_32=self.local2_bit_index_32,
                        bits_len=self.local2_bits_len,
                        target_watermark=self.watermark_local2,
                        target_fpr=self.fpr_payload,
                        n_tests=self.payload_total_tests,
                        valid_mask_32=valid_mask_32,
                    )
                    combined = self._combine_payload_replicas(local1_info, local2_info)
                    eff_n = max(int(combined["effective_bits_len"]), 1)
                    z_score = (float(combined["match_bits"]) - 0.5 * eff_n) / math.sqrt(0.25 * eff_n)
                    validation = {
                        "match_bits": int(combined["match_bits"]),
                        "acc": float(combined["acc"]),
                        "effective_bits_len": int(combined["effective_bits_len"]),
                        "z_score": float(z_score),
                    }
                except Exception as exc:
                    validation = {
                        "match_bits": -1,
                        "acc": 0.0,
                        "effective_bits_len": 0,
                        "z_score": -1e9,
                        "error": str(exc),
                    }

            candidate["payload_validation"] = validation
            estimate = candidate["estimate"]
            validation_score = (
                float(validation.get("z_score", -1e9)),
                float(validation.get("acc", 0.0)),
                int(estimate.get("inlier_count", 0)),
                validation["match_bits"],
                -float(estimate.get("rmse", 1e9)),
            )
            if best is None or validation_score > best["validation_score"]:
                best = {
                    "candidate": candidate,
                    "validation_score": validation_score,
                }

        return best["candidate"] if best is not None else candidates[0]

    def _fallback_scan_sync(self, restored_w_64, key_sign_64):
        best_anchor = None
        best_angle = 0.0
        best_scale = 1.0
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

        return {
            "passed": best_anchor is not None,
            "source": "fallback_scan",
            "angle": float(refined_angle),
            "scale": float(best_scale),
            "translation": [0.0, 0.0],
            "best_anchor_before_final_decode": best_anchor,
        }

    def _align_latent_with_procrustes(self, w_64, sync_result):
        """
        Resample attacked latent back to the canonical latent grid using the
        Procrustes transform estimated on 32x32 landmark coordinates.

        weighted_procrustes_2d estimates:
            observed_macro ~= scale * R @ canonical_macro + translation

        Important: when coarse rotation search is used, the Procrustes estimate
        is computed after applying TF.affine(..., angle=-coarse_angle). The
        residual transform therefore lives in that coarse-aligned coordinate
        system, not in the original attacked latent coordinate system.
        """
        if sync_result.get("source") != "procrustes":
            raise ValueError("Procrustes alignment requires a procrustes sync result.")

        batch, channels, height, width = w_64.shape
        device = w_64.device
        dtype = torch.float32

        coarse_angle = float(sync_result.get("coarse_angle", 0.0))
        if abs(coarse_angle) > 1e-6:
            coarse_aligned = TF.affine(
                w_64,
                angle=-coarse_angle,
                translate=[0, 0],
                scale=1.0,
                shear=0,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            coarse_valid = TF.affine(
                torch.ones((batch, 1, height, width), dtype=torch.float32, device=device),
                angle=-coarse_angle,
                translate=[0, 0],
                scale=1.0,
                shear=0,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
        else:
            coarse_aligned = w_64
            coarse_valid = torch.ones((batch, 1, height, width), dtype=torch.float32, device=device)

        estimated_scale = max(float(sync_result.get("scale", 1.0)), 1e-6)
        residual_angle = float(sync_result.get("residual_angle", sync_result.get("angle", 0.0)))
        angle_rad = math.radians(residual_angle)
        cos_v = math.cos(angle_rad)
        sin_v = math.sin(angle_rad)
        translation = sync_result.get("translation", [0.0, 0.0])
        tx = float(translation[0])
        ty = float(translation[1])

        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )

        # Convert latent pixel coordinates to logical macro-grid coordinates.
        x_macro = xs / float(self.patch_size)
        y_macro = ys / float(self.patch_size)

        obs_x_macro = estimated_scale * (cos_v * x_macro - sin_v * y_macro) + tx
        obs_y_macro = estimated_scale * (sin_v * x_macro + cos_v * y_macro) + ty

        obs_x = obs_x_macro * float(self.patch_size)
        obs_y = obs_y_macro * float(self.patch_size)

        grid_x = 2.0 * obs_x / max(width - 1, 1) - 1.0
        grid_y = 2.0 * obs_y / max(height - 1, 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(batch, 1, 1, 1)
        valid_latent = (
            (grid_x >= -1.0)
            & (grid_x <= 1.0)
            & (grid_y >= -1.0)
            & (grid_y <= 1.0)
        )
        valid_macro = F.avg_pool2d(
            valid_latent.float().view(1, 1, height, width),
            kernel_size=self.patch_size,
        )[0, 0] > 0.999

        aligned = F.grid_sample(
            coarse_aligned.float(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).to(w_64.dtype)

        coarse_valid_sampled = F.grid_sample(
            coarse_valid,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        coarse_valid_macro = F.avg_pool2d(
            coarse_valid_sampled,
            kernel_size=self.patch_size,
        )[0, 0] > 0.999

        return aligned, (valid_macro & coarse_valid_macro)

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

        anchor_info = self.eval_global_anchor_detail(final_soft_m_32)
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

        anchor_tau_bits = int(anchor_info.get("tau_bits_effective", self.tau_anchor_bits))
        anchor_tau_acc = float(anchor_info.get("tau_acc_effective", self.tau_anchor))
        anchor_actual_fpr = float(anchor_info.get("actual_fpr_effective", self.actual_fpr_anchor))
        anchor_pass = anchor_info["match_bits"] >= anchor_tau_bits
        local1_pass = local1_info["match_bits"] >= int(local1_info.get("tau_bits_effective", self.tau_local1_bits))
        local2_pass = local2_info["match_bits"] >= int(local2_info.get("tau_bits_effective", self.tau_local2_bits))
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
                    "effective_bits_len": int(local1_info.get("effective_bits_len", self.local1_bits_len)),
                    "tau_bits": int(local1_info.get("tau_bits_effective", self.tau_local1_bits)),
                    "tau_acc": float(local1_info.get("tau_acc_effective", self.tau_local1)),
                    "actual_fpr": float(local1_info.get("actual_fpr_effective", self.actual_fpr_local1)),
                },
                "local2": {
                    "passed": bool(local2_pass),
                    "acc": local2_info["acc"],
                    "match_bits": local2_info["match_bits"],
                    "bits_len": self.local2_bits_len,
                    "effective_bits_len": int(local2_info.get("effective_bits_len", self.local2_bits_len)),
                    "tau_bits": int(local2_info.get("tau_bits_effective", self.tau_local2_bits)),
                    "tau_acc": float(local2_info.get("tau_acc_effective", self.tau_local2)),
                    "actual_fpr": float(local2_info.get("actual_fpr_effective", self.actual_fpr_local2)),
                },
            },
        }

    def eval_watermark(self, reversed_w_64):
        """
        Detect anchor and payload with FPR-controlled thresholds.

        This definition intentionally overrides the legacy implementation above.
        It first tries payload-bound Procrustes landmarks to estimate scale and
        translation in closed form, then falls back to the old angle-scale scan.
        """
        if self.key_64 is None:
            raise RuntimeError("Key has not been generated. Call create_watermark_and_return_w() before eval_watermark().")

        restored_w_64 = self.roar_blur(reversed_w_64)
        key_sign_64 = (self.key_64 * -2 + 1).float()

        baseline_soft = self._unlock_and_pool(restored_w_64, key_sign_64)
        baseline_anchor = self.eval_global_anchor_detail(baseline_soft)

        if baseline_anchor["match_bits"] >= self.tau_anchor_bits:
            sync_result = {
                "passed": True,
                "source": "baseline",
                "angle": 0.0,
                "scale": 1.0,
                "translation": [0.0, 0.0],
                "best_anchor_before_final_decode": baseline_anchor,
            }
        else:
            sync_result = {"passed": False}
            if self.use_procrustes_sync:
                sync_result, _ = self._procrustes_sync_search(restored_w_64, key_sign_64)
            if not sync_result.get("passed", False):
                fallback_result = self._fallback_scan_sync(restored_w_64, key_sign_64)
                fallback_result["procrustes_attempt"] = sync_result
                sync_result = fallback_result

            logging.info(
                f"[Sync] source={sync_result.get('source')} | "
                f"scale={sync_result.get('scale', 1.0):.4f} | "
                f"angle={-sync_result.get('angle', 0.0):.2f} | "
                f"translation={sync_result.get('translation', [0.0, 0.0])} | "
                f"landmarks={sync_result.get('landmarks', {}).get('num_detected', 0) if isinstance(sync_result.get('landmarks', {}), dict) else 0} | "
                f"inliers={sync_result.get('inlier_count', 0)} | "
                f"rmse={sync_result.get('rmse', None)} | "
                f"sync_payload={sync_result.get('payload_validation', {}).get('match_bits', None)}/"
                f"{sync_result.get('payload_validation', {}).get('effective_bits_len', None)} | "
                f"sync_z={sync_result.get('payload_validation', {}).get('z_score', None)}"
            )

        refined_angle = float(sync_result.get("angle", 0.0))
        estimated_scale = max(float(sync_result.get("scale", 1.0)), 1e-3)
        alignment_scale = estimated_scale
        translation = sync_result.get("translation", [0.0, 0.0])
        translate_latent = [
            int(round(-float(translation[0]) * self.patch_size)),
            int(round(-float(translation[1]) * self.patch_size)),
        ]

        if sync_result.get("source") == "procrustes":
            final_aligned_w_64, valid_mask_32 = self._align_latent_with_procrustes(restored_w_64, sync_result)
        else:
            final_aligned_w_64 = TF.affine(
                restored_w_64,
                angle=-refined_angle,
                translate=translate_latent,
                scale=alignment_scale,
                shear=0,
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            valid_mask_32 = torch.ones(
                (self.logical_size, self.logical_size),
                dtype=torch.bool,
                device=restored_w_64.device,
            )
        final_soft_m_32 = self._unlock_and_pool(final_aligned_w_64, key_sign_64)

        anchor_info = self._decode_indexed_from_soft(
            soft_m_32=final_soft_m_32,
            bit_index_32=self.anchor_bit_index_32,
            bits_len=self.global_bits_len,
            target_watermark=self.watermark_anchor,
            target_fpr=self.fpr_anchor,
            n_tests=self.anchor_search_trials,
            valid_mask_32=valid_mask_32,
        )
        local1_info = self._decode_indexed_from_soft(
            soft_m_32=final_soft_m_32,
            bit_index_32=self.local1_bit_index_32,
            bits_len=self.local1_bits_len,
            target_watermark=self.watermark_local1,
            target_fpr=self.fpr_payload,
            n_tests=self.payload_total_tests,
            valid_mask_32=valid_mask_32,
        )
        local2_info = self._decode_indexed_from_soft(
            soft_m_32=final_soft_m_32,
            bit_index_32=self.local2_bit_index_32,
            bits_len=self.local2_bits_len,
            target_watermark=self.watermark_local2,
            target_fpr=self.fpr_payload,
            n_tests=self.payload_total_tests,
            valid_mask_32=valid_mask_32,
        )

        anchor_tau_bits = int(anchor_info.get("tau_bits_effective", self.tau_anchor_bits))
        anchor_tau_acc = float(anchor_info.get("tau_acc_effective", self.tau_anchor))
        anchor_actual_fpr = float(anchor_info.get("actual_fpr_effective", self.actual_fpr_anchor))
        bit_anchor_pass = anchor_info["match_bits"] >= anchor_tau_bits
        landmark_anchor_pass = (
            sync_result.get("source") == "procrustes"
            and bool(sync_result.get("passed", False))
            and int(sync_result.get("inlier_count", 0)) >= int(self.procrustes_config.min_landmarks_for_sync)
            and float(sync_result.get("rmse", 1e9)) <= float(self.procrustes_config.max_sync_rmse)
        )
        anchor_pass = bool(bit_anchor_pass or landmark_anchor_pass)
        local1_pass = local1_info["match_bits"] >= self.tau_local1_bits
        local2_pass = local2_info["match_bits"] >= self.tau_local2_bits
        combined_payload_info = self._combine_payload_replicas(local1_info, local2_info)
        payload_pass = bool(combined_payload_info["passed"])
        payload_anchor_consistency_pass = True

        best_payload_name = "combined"
        best_payload_info = combined_payload_info
        best_payload_tau_bits = self.tau_payload_combined_bits
        best_payload_tau_acc = self.tau_payload_combined

        final_watermarked = bool(anchor_pass and payload_pass and payload_anchor_consistency_pass)
        landmarks = sync_result.get("landmarks", {})
        if not isinstance(landmarks, dict):
            landmarks = {}

        return {
            "final_watermarked": final_watermarked,
            "sync": {
                "source": sync_result.get("source", "unknown"),
                "passed": bool(sync_result.get("passed", False)),
                "angle": float(refined_angle),
                "scale": float(estimated_scale),
                "alignment_scale": float(alignment_scale),
                "translation": [
                    float(translation[0]),
                    float(translation[1]),
                ],
                "translate_latent": translate_latent,
                "rmse": sync_result.get("rmse", None),
                "inlier_count": int(sync_result.get("inlier_count", 0)),
                "inlier_ratio": float(sync_result.get("inlier_ratio", 0.0)),
                "ransac_used": bool(sync_result.get("ransac_used", False)),
                "num_landmarks": int(landmarks.get("num_detected", 0)),
                "sync_score": float(landmarks.get("mean_score", 0.0)),
                "landmark_threshold": float(self.landmark_threshold),
                "best_anchor_before_final_decode": sync_result.get("best_anchor_before_final_decode", None),
                "procrustes": sync_result if sync_result.get("source") == "procrustes" else sync_result.get("procrustes_attempt", None),
                "payload_validation": sync_result.get("payload_validation", None),
            },
            "anchor": {
                "passed": bool(anchor_pass),
                "acc": anchor_info["acc"],
                "match_bits": anchor_info["match_bits"],
                "bits_len": self.global_bits_len,
                "effective_bits_len": int(anchor_info.get("effective_bits_len", self.global_bits_len)),
                "tau_bits": int(anchor_tau_bits),
                "tau_acc": float(anchor_tau_acc),
                "target_fpr": float(self.fpr_anchor),
                "actual_fpr": float(anchor_actual_fpr),
                "n_tests": int(self.anchor_search_trials),
                "payload_anchor_consistency_pass": bool(payload_anchor_consistency_pass),
                "bit_anchor_pass": bool(bit_anchor_pass),
                "landmark_anchor_pass": bool(landmark_anchor_pass),
            },
            "payload": {
                "passed": bool(payload_pass),
                "best_name": best_payload_name,
                "best_acc": best_payload_info["acc"],
                "best_match_bits": best_payload_info["match_bits"],
                "best_tau_bits": int(best_payload_tau_bits),
                "best_tau_acc": float(best_payload_tau_acc),
                "target_fpr": float(self.fpr_payload),
                "n_tests": int(self.payload_combined_total_tests),
                "combined": {
                    "passed": bool(combined_payload_info["passed"]),
                    "acc": combined_payload_info["acc"],
                    "match_bits": combined_payload_info["match_bits"],
                    "bits_len": combined_payload_info["bits_len"],
                    "effective_bits_len": int(combined_payload_info["effective_bits_len"]),
                    "tau_bits": int(combined_payload_info["tau_bits"]),
                    "tau_acc": float(combined_payload_info["tau_acc"]),
                    "actual_fpr": float(combined_payload_info["actual_fpr"]),
                },
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
