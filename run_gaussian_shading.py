import argparse
import copy
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *
from attacks import build_advanced_attackers
import os
import logging


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(f"{args.output_path}/w", exist_ok=True)
    if args.save_distorted_images:
        os.makedirs(f"{args.output_path}/attacked", exist_ok=True)

    log_file = os.path.join(args.output_path, "gaussian_shading_test.log")
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("========== 开始新实验 ==========")
    logging.info(f"参数配置: {args}")
    logging.info(f"评估模式: {args.eval_mode}")
    logging.info(f"图像攻击配置: {get_attack_summary(args)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16,
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    advanced_attackers = build_advanced_attackers(args, device=device)
    if advanced_attackers:
        logging.info(f"已启用高级攻击模块: {list(advanced_attackers.keys())}")

    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device,
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    dataset, prompt_key = get_dataset(args)

    procrustes_config = ProcrustesAnchorConfig(
        logical_size=32,
        num_landmarks=args.num_landmarks,
        landmark_patch=args.landmark_patch,
        min_landmarks_for_sync=args.min_landmarks_for_sync,
        landmark_key_mode="payload_hmac",
        use_procrustes_sync=not args.disable_procrustes_sync,
        min_landmark_distance=args.min_landmark_distance,
        search_radius=args.landmark_search_radius,
        max_sync_rmse=args.max_sync_rmse,
        min_inlier_ratio_for_sync=args.min_inlier_ratio_for_sync,
        landmark_threshold_cap=args.landmark_threshold_cap,
        use_ransac=not args.disable_landmark_ransac,
        ransac_trials=args.landmark_ransac_trials,
        ransac_inlier_threshold=args.landmark_ransac_inlier_threshold,
        sync_angle_min=args.sync_angle_min,
        sync_angle_max=args.sync_angle_max,
        sync_angle_step=args.sync_angle_step,
        sync_candidate_topk=args.sync_candidate_topk,
        use_payload_sync_validation=not args.disable_payload_sync_validation,
    )

    watermark = Gaussian_Shading(
        args.channel_copy,
        args.hw_copy,
        args.fpr,
        args.user_number,
        anchor_search_trials=args.anchor_search_trials,
        payload_search_trials=args.payload_search_trials,
        fpr_anchor=args.fpr_anchor,
        fpr_payload=args.fpr_payload,
        procrustes_config=procrustes_config,
    )

    logging.info(f"统计检测阈值: {watermark.get_threshold_summary()}")

    tester_prompt = ""
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # vanilla 模式用于无水印图像误检率（FPR）测试：
    # 先固定一套检测端 reference watermark/key，但生成图像时不用它。
    if args.eval_mode == "vanilla":
        _ = watermark.create_watermark_and_return_w()
        logging.info("Vanilla FPR 测试：已固定一套检测用 watermark/key，图像将使用标准高斯随机 latent 生成。")

    results_anchor_acc = []
    results_anchor_pass = []
    results_payload_acc = []
    results_payload_pass = []
    results_final_pass = []
    results_best_payload_name = []

    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        set_random_seed(seed)

        if args.eval_mode == "watermarked":
            # 正样本 TPR 测试：使用带水印 latent 生成图像
            init_latents_w = watermark.create_watermark_and_return_w()
        elif args.eval_mode == "vanilla":
            # 无水印 FPR 测试：使用标准高斯 latent 生成图像
            # 注意：检测端 watermark/key 已在循环前固定
            init_latents_w = torch.randn(
                (1, 4, 64, 64),
                dtype=torch.float16,
                device=device,
            )
        else:
            raise ValueError(f"Unknown eval_mode: {args.eval_mode}")

        outputs = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
        )
        image_w = outputs.images[0]
        image_w.save(f"{args.output_path}/w/{args.eval_mode}_image_{i}.png")

        image_w_distortion = image_distortion(
            image_w,
            seed,
            args,
            advanced_attackers=advanced_attackers,
            prompt=current_prompt,
        )
        if args.save_distorted_images:
            image_w_distortion.save(f"{args.output_path}/attacked/{args.eval_mode}_attacked_{i}.png")

        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        detection = watermark.eval_watermark(reversed_latents_w)

        anchor_acc = detection["anchor"]["acc"]
        payload_acc = detection["payload"]["best_acc"]
        anchor_pass = detection["anchor"]["passed"]
        payload_pass = detection["payload"]["passed"]
        final_pass = detection["final_watermarked"]
        best_payload_name = detection["payload"]["best_name"]

        results_anchor_acc.append(anchor_acc)
        results_anchor_pass.append(anchor_pass)
        results_payload_acc.append(payload_acc)
        results_payload_pass.append(payload_pass)
        results_final_pass.append(final_pass)
        results_best_payload_name.append(best_payload_name)

        if args.eval_mode == "vanilla":
            image_decision_name = "FalsePositive"
        else:
            image_decision_name = "Final"

        logging.info(
            f"Image {i} | "
            f"Anchor: {detection['anchor']['match_bits']}/"
            f"{detection['anchor'].get('effective_bits_len', detection['anchor']['bits_len'])} "
            f"Acc={anchor_acc:.4f} Pass={anchor_pass} | "
            f"Payload({best_payload_name}): {detection['payload']['best_match_bits']}/"
            f"{detection['payload'][best_payload_name].get('effective_bits_len', detection['payload'][best_payload_name]['bits_len'])} "
            f"Acc={payload_acc:.4f} Pass={payload_pass} | "
            f"{image_decision_name}={final_pass}"
        )

    # ==========================================
    # === Final evaluation report ===
    # ==========================================
    total_images = len(results_anchor_acc)

    anchor_bits = watermark.global_bits_len
    local1_bits = watermark.local1_bits_len
    local2_bits = watermark.local2_bits_len
    total_bits = anchor_bits + local1_bits + local2_bits

    avg_anchor_acc = sum(results_anchor_acc) / total_images
    avg_payload_acc = sum(results_payload_acc) / total_images

    anchor_detected_count = sum(1 for x in results_anchor_pass if x)
    payload_detected_count = sum(1 for x in results_payload_pass if x)
    final_detected_count = sum(1 for x in results_final_pass if x)

    anchor_rate = anchor_detected_count / total_images
    payload_rate = payload_detected_count / total_images
    final_rate = final_detected_count / total_images

    if args.eval_mode == "vanilla":
        rate_name = "FPR"
        report_title = "最终无水印误检率评估报告"
        system_section = "[系统级误检能力]"
        final_label = "Final FPR = Anchor AND Payload"
        anchor_label = "Anchor FPR"
        payload_label = "Payload FPR"
    else:
        rate_name = "TPR"
        report_title = "最终有水印检出率评估报告"
        system_section = "[系统级检测能力]"
        final_label = "Final TPR = Anchor AND Payload"
        anchor_label = "Anchor TPR"
        payload_label = "Payload TPR"

    latent_dim = 4 * 64 * 64
    hgs_redundancy = latent_dim / total_bits

    logging.info("\n" + "=" * 50)
    logging.info(f"============= {report_title} =============")
    logging.info(f"评估图像总数: {total_images}")

    logging.info("-" * 50)
    logging.info("[水印容量分布 (Payload Capacity)]")
    logging.info(f"总净容量 (Total Net Payload): {total_bits} bits")
    logging.info(f" ├─ 全局同步锚点: {anchor_bits} bits (全息扩频冗余: {hgs_redundancy:.1f}倍)")
    logging.info(f" ├─ 局部分包 1: {local1_bits} bits (全息扩频冗余: {hgs_redundancy:.1f}倍)")
    logging.info(f" └─ 局部分包 2: {local2_bits} bits (全息扩频冗余: {hgs_redundancy:.1f}倍)")

    logging.info("-" * 50)
    logging.info(f"[Gaussian Shading / GaussMarker 式统计检测阈值]")
    logging.info(
        f"Anchor 阈值: >= {watermark.tau_anchor_bits}/{anchor_bits} bits "
        f"(Acc >= {watermark.tau_anchor:.4f}), "
        f"目标FPR={watermark.fpr_anchor}, 实际FPR≈{watermark.actual_fpr_anchor:.3e}, "
        f"候选检验次数={watermark.anchor_search_trials}"
    )
    logging.info(
        f"Payload local1 阈值: >= {watermark.tau_local1_bits}/{local1_bits} bits "
        f"(Acc >= {watermark.tau_local1:.4f}), "
        f"目标FPR={watermark.fpr_payload}, 实际FPR≈{watermark.actual_fpr_local1:.3e}, "
        f"候选检验次数={watermark.payload_total_tests}"
    )
    logging.info(
        f"Payload local2 阈值: >= {watermark.tau_local2_bits}/{local2_bits} bits "
        f"(Acc >= {watermark.tau_local2:.4f}), "
        f"目标FPR={watermark.fpr_payload}, 实际FPR≈{watermark.actual_fpr_local2:.3e}, "
        f"候选检验次数={watermark.payload_total_tests}"
    )

    logging.info("-" * 50)
    logging.info(system_section)
    logging.info(f"{anchor_label}: {anchor_rate * 100:.4f}% ({anchor_detected_count}/{total_images})")
    logging.info(f"{payload_label}: {payload_rate * 100:.4f}% ({payload_detected_count}/{total_images})")
    logging.info(f"{final_label}: {final_rate * 100:.4f}% ({final_detected_count}/{total_images})")

    logging.info("-" * 50)
    logging.info("[信息提取保真度 (Message Fidelity)]")
    logging.info(f"平均全局锚点准确率 (Mean Anchor Acc): {avg_anchor_acc * 100:.2f}%")
    logging.info(f"平均Payload最佳准确率 (Mean Payload Best Acc): {avg_payload_acc * 100:.2f}%")
    logging.info("==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian Shading / ANCHOR")
    parser.add_argument("--num", default=1000, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--num_inversion_steps", default=None, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--channel_copy", default=1, type=int)
    parser.add_argument("--hw_copy", default=8, type=int)
    parser.add_argument("--user_number", default=1000000, type=int)
    parser.add_argument("--fpr", default=0.000001, type=float)
    parser.add_argument(
        "--eval_mode",
        default="watermarked",
        choices=["watermarked", "vanilla"],
        help="watermarked: 测有水印图像 TPR；vanilla: 测无水印图像 FPR",
    )
    parser.add_argument("--fpr_anchor", default=None, type=float)
    parser.add_argument("--fpr_payload", default=None, type=float)
    parser.add_argument("--anchor_search_trials", default=None, type=int)
    parser.add_argument("--payload_search_trials", default=None, type=int)
    parser.add_argument("--disable_procrustes_sync", action="store_true")
    parser.add_argument("--num_landmarks", default=64, type=int)
    parser.add_argument("--landmark_patch", default=3, type=int)
    parser.add_argument("--min_landmarks_for_sync", default=10, type=int)
    parser.add_argument("--min_landmark_distance", default=3, type=int)
    parser.add_argument("--landmark_search_radius", default=16, type=int)
    parser.add_argument("--max_sync_rmse", default=1.5, type=float)
    parser.add_argument("--min_inlier_ratio_for_sync", default=0.18, type=float)
    parser.add_argument("--landmark_threshold_cap", default=0.45, type=float)
    parser.add_argument("--disable_landmark_ransac", action="store_true")
    parser.add_argument("--landmark_ransac_trials", default=128, type=int)
    parser.add_argument("--landmark_ransac_inlier_threshold", default=2.0, type=float)
    parser.add_argument("--sync_angle_min", default=-180, type=int)
    parser.add_argument("--sync_angle_max", default=180, type=int)
    parser.add_argument("--sync_angle_step", default=5, type=int)
    parser.add_argument("--sync_candidate_topk", default=16, type=int)
    parser.add_argument("--disable_payload_sync_validation", action="store_true")
    parser.add_argument("--output_path", default="./output/")
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--reference_model_pretrain", default=None)
    parser.add_argument("--dataset_path", default="/root/autodl-tmp/sd-prompts")
    parser.add_argument("--model_path", default="/root/autodl-tmp/sd-2-1-base")

    parser.add_argument("--jpeg_ratio", default=None, type=int)
    parser.add_argument("--random_crop_ratio", default=None, type=float)
    parser.add_argument("--random_drop_ratio", default=None, type=float)
    parser.add_argument("--gaussian_blur_r", default=None, type=int)
    parser.add_argument("--median_blur_k", default=None, type=int)
    parser.add_argument("--resize_ratio", default=None, type=float)
    parser.add_argument("--gaussian_std", default=None, type=float)
    parser.add_argument("--sp_prob", default=None, type=float)
    parser.add_argument("--brightness_factor", default=None, type=float)
    parser.add_argument("--rotation_degree", default=None, type=float, help="单独旋转攻击，单位为度；保留旧参数兼容。")

    # -----------------------------
    # RST geometric attacks
    # -----------------------------
    parser.add_argument("--rst_rotation_degree", default=None, type=float, help="RST固定旋转角度，单位为度。例如 15 表示旋转15度。")
    parser.add_argument("--rst_random_rotation_degree", default=None, type=float, help="RST随机旋转最大角度；实际角度从[-value, value]采样。")
    parser.add_argument("--rst_scale", default=None, type=float, help="RST固定尺度缩放，1.0表示不缩放，>1表示放大，<1表示缩小。")
    parser.add_argument("--rst_scale_min", default=None, type=float, help="RST随机尺度缩放下限，需要和 --rst_scale_max 同时使用。")
    parser.add_argument("--rst_scale_max", default=None, type=float, help="RST随机尺度缩放上限，需要和 --rst_scale_min 同时使用。")
    parser.add_argument("--rst_translate_x", default=0.0, type=float, help="RST水平平移比例，相对图像宽度。例如0.05表示右移5%宽度。")
    parser.add_argument("--rst_translate_y", default=0.0, type=float, help="RST垂直平移比例，相对图像高度。例如-0.05表示上移5%高度。")
    parser.add_argument("--rst_shear", default=0.0, type=float, help="RST剪切角度，默认0。")
    parser.add_argument("--rst_resized_crop_ratio", default=None, type=float, help="Crop-and-scale面积比例。例如0.75表示裁剪75%面积后缩放回原尺寸，接近GaussMarker/MaXsive设置。")
    parser.add_argument("--rst_crop_side_ratio", default=None, type=float, help="Crop-and-scale边长比例。例如0.75表示裁剪75%边长的正方形后缩放回原尺寸。")
    parser.add_argument("--rst_order", default="affine_then_crop", choices=["affine_then_crop", "crop_then_affine"], help="RST组合攻击顺序。")
    parser.add_argument("--rst_fill", default=0, type=int, help="仿射变换后空白区域填充值，默认0为黑色。")
    # -----------------------------
    # Advanced attacks: VAE compression and diffusion regeneration
    # -----------------------------
    parser.add_argument("--vae_attack", action="store_true", help="启用 CompressAI VAE/神经压缩攻击。")
    parser.add_argument(
        "--vae_model",
        default="bmshj2018-factorized",
        choices=["bmshj2018-factorized", "bmshj2018-hyperprior", "mbt2018-mean", "mbt2018", "cheng2020-anchor"],
        help="CompressAI 压缩模型名称，参考 MaXsive/GaussMarker 的 VAE compression attack 设置。",
    )
    parser.add_argument("--vae_quality", default=1, type=int, help="CompressAI quality，通常 1 最强压缩，数值越大质量越高。")

    parser.add_argument("--regen_attack", action="store_true", help="启用 diffusion regeneration / img2img 再生成攻击。")
    parser.add_argument("--regen_model_path", default=None, help="再生成攻击使用的 SD img2img 模型路径；默认复用 --model_path。")
    parser.add_argument("--regen_noise_step", default=60, type=int, help="MaXsive式 regeneration 加噪时间步。越大加噪越强，攻击越强。常用20/40/60/80/100。")
    parser.add_argument("--regen_denoise_steps", default=50, type=int, help="regeneration 去噪步数；通常保持50。")
    parser.add_argument("--regen_guidance_scale", default=7.5, type=float, help="regeneration guidance scale；为对齐MaXsive默认使用7.5。")
    parser.add_argument("--regen_start_step_mode", default="maxsive", choices=["maxsive", "nearest"], help="去噪起点计算方式。maxsive使用 denoise_steps - max(noise_step//20,1)；nearest选择最接近noise_step的scheduler步。")
    parser.add_argument("--regen_use_prompt", action="store_true", help="再生成攻击使用原始 prompt；默认使用空 prompt。")
    parser.add_argument("--regen_dtype", default="float16", choices=["float16", "float32"], help="再生成攻击模型精度。")
    parser.add_argument("--regen_seed", default=1024, type=int, help="再生成攻击随机种子。")
    parser.add_argument(
        "--advanced_attack_order",
        default="vae_then_regen",
        choices=["vae_then_regen", "regen_then_vae"],
        help="当 VAE 与 Regeneration 同时启用时的执行顺序。",
    )

    parser.add_argument("--save_distorted_images", action="store_true", help="保存攻击后的图像到 output_path/attacked，用于人工检查攻击是否生效。")

    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
