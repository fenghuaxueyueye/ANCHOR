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
import os
import logging
from scipy.stats import binom

def main(args):
    
    os.makedirs(args.output_path, exist_ok=True)
    
    log_file = os.path.join(args.output_path, 'gaussian_shading_test.log')
    logging.basicConfig(
        filename=log_file,          
        filemode='a',               
        level=logging.INFO,         
        format='%(asctime)s - %(message)s', 
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logging.info(f"========== 开始新实验==========")
    logging.info(f"参数配置: {args}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model,
                                                                                  pretrained=args.reference_model_pretrain,
                                                                                  device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    dataset, prompt_key = get_dataset(args)

    watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    os.makedirs(args.output_path, exist_ok=True)

    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    results_anchor_acc = []
    results_local_max_acc = []

    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        set_random_seed(seed)
        init_latents_w = watermark.create_watermark_and_return_w()
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
        image_w.save(f"{args.output_path}/w/image_{i}.png")

        image_w_distortion = image_distortion(image_w, seed, args)

        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        acc_anchor, acc_local_max = watermark.eval_watermark(reversed_latents_w)
        
        results_anchor_acc.append(acc_anchor)
        results_local_max_acc.append(acc_local_max)
        
        logging.info(f"Image {i} | Global Anchor: {acc_anchor:.4f} | Local Max: {acc_local_max:.4f}")

    # ==========================================
    # === 外部循环：输出最终评估报告 ===
    # ==========================================
    total_images = len(results_anchor_acc)
    
    anchor_bits = watermark.global_bits_len
    local1_bits = watermark.local1_bits_len
    local2_bits = watermark.local2_bits_len
    total_bits = anchor_bits + local1_bits + local2_bits

    avg_anchor_acc = sum(results_anchor_acc) / total_images
    avg_local_max_acc = sum(results_local_max_acc) / total_images

    target_fpr = 1e-6
    threshold_bits = binom.isf(target_fpr, anchor_bits, 0.5) 
    tau_acc = threshold_bits / anchor_bits

    detected_count = sum(1 for acc in results_anchor_acc if acc >= tau_acc)
    tpr = detected_count / total_images

    logging.info("\n" + "="*50)
    logging.info("============= 最终测试评估报告 =============")
    logging.info(f"评估图像总数: {total_images}")
    
    # 动态计算全息扩频冗余 (HGS Redundancy)
    latent_dim = 4 * 64 * 64  # 16384 个潜空间维度
    hgs_redundancy = latent_dim / total_bits  # 均摊到每个 bit 上的扩频投影维度数

    logging.info("-" * 50)
    logging.info(f"[水印容量分布 (Payload Capacity)]")
    logging.info(f"总净容量 (Total Net Payload) : {total_bits} bits")
    logging.info(f" ├─ 全局同步锚点: {anchor_bits} bits (全息扩频冗余: {hgs_redundancy:.1f}倍)")
    logging.info(f" ├─ 局部分包 1: {local1_bits} bits (全息扩频冗余: {hgs_redundancy:.1f}倍)")
    logging.info(f" └─ 局部分包 2: {local2_bits} bits (全息扩频冗余: {hgs_redundancy:.1f}倍)")
    
    logging.info("-" * 50)
    logging.info(f"[系统级检测能力 (Detection @ FPR={target_fpr})]")
    logging.info(f"数学检测阈值: 需匹配 >= {int(threshold_bits)} / {anchor_bits} bits (Acc >= {tau_acc:.4f})")
    logging.info(f"True Positive Rate (TPR): {tpr * 100:.2f}%  ({detected_count}/{total_images})")
    
    logging.info("-" * 50)
    logging.info(f"[信息提取保真度 (Message Fidelity)]")
    logging.info(f"平均全局锚点准确率 (Mean Anchor Acc): {avg_anchor_acc * 100:.2f}%")
    logging.info(f"平均局部最高准确率 (Mean Local Max Acc): {avg_local_max_acc * 100:.2f}%")
    logging.info("==================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='/root/autodl-tmp/sd-prompts')
    parser.add_argument('--model_path', default='/root/autodl-tmp/sd-2-1-base')

    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--random_crop_ratio', default=None, type=float)
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rotation_degree', default=None, type=float)

    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)