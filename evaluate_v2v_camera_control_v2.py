import argparse
import json
import os

import numpy as np
import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, PNDMScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import BertModel, BertTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection, T5EncoderModel, T5Tokenizer

from easyanimate.models import name_to_autoencoder_magvit, name_to_transformer3d

# from easyanimate.pipeline.pipeline_easyanimate_inpaint import EasyAnimateInpaintPipeline
# from easyanimate.pipeline.pipeline_easyanimate_multi_text_encoder_inpaint import EasyAnimatePipeline_Multi_Text_Encoder_Inpaint
from easyanimate.pipeline.pipeline_easyanimate_camera_control_v1 import EasyAnimatePipelineCameraControl
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import get_image_to_video_latent, get_video_to_video_latent, save_videos_grid, get_evaluation_model_input
from easyanimate.utils.fp8_optimization import convert_weight_dtype_wrapper
from easyanimate.models.pose_encoder import CameraPoseEncoderCameraCtrl, VideoFrameTokenization


def parse_args():
    parser = argparse.ArgumentParser(description="EasyAnimate 视频生成参数配置")

    parser.add_argument("--assets_json_path", type=str, default="asset/evaluate_data.json", help="资产评估数据的JSON文件路径")
    parser.add_argument("--GPU_memory_mode", type=str, default="model_cpu_offload", choices=["model_cpu_offload", "model_cpu_offload_and_qfloat8", "sequential_cpu_offload"])
    parser.add_argument("--config_path", type=str, default="config/easyanimate_video_v5_magvit_camera_control.yaml", help="配置文件的路径")
    parser.add_argument("--model_name", type=str, default="models/Diffusion_Transformer/EasyAnimateV5-7b-zh-CameraControl", help="模型名称或路径")
    parser.add_argument("--transformer_model_name", type=str, default="output_dir_20241211/checkpoint-latest", help="Transformer模型的名称或路径")
    parser.add_argument("--sampler_name", type=str, default="DDIM", choices=["Euler", "Euler A", "DPM++", "PNDM", "DDIM"])
    parser.add_argument("--transformer_path", type=str, default=None, help="预训练Transformer模型的路径")
    parser.add_argument("--motion_module_path", type=str, default=None, help="运动模块的路径")
    parser.add_argument("--vae_path", type=str, default=None, help="VAE模型的路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA模型的路径")
    parser.add_argument("--sample_width", type=int, default=384, help="样本宽度（像素）")
    parser.add_argument("--sample_height", type=int, default=672, help="样本高度（像素）")
    parser.add_argument("--video_length", type=int, default=49, help="视频长度（帧数）")
    parser.add_argument("--video_sample_stride", type=int, default=3, help="视频帧率（FPS）")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="权重的数据类型：[float32, float16, bfloat16]")
    parser.add_argument("--denoise_strength", type=float, default=1.0, help="去噪强度")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="引导规模")
    parser.add_argument("--seed", type=int, default=43, help="随机种子")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--lora_weight", type=float, default=0.55, help="LoRA权重")
    parser.add_argument("--save_path", type=str, default="samples/easyanimate-videos_v2v", help="生成视频的保存路径")

    args = parser.parse_args()

    # 处理 sample_size 作为列表
    args.sample_size = [args.sample_width, args.sample_height]

    # 处理 weight_dtype 转换为 torch dtype
    dtype_mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    args.weight_dtype = dtype_mapping.get(args.weight_dtype, torch.bfloat16)

    return args


def main(args):

    with open(args.assets_json_path, 'r', encoding='utf-8') as file:
        all_data = json.load(file)

    # # If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
    # prompt = asset_data['text']
    # negative_prompt = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
    # validation_image = asset_data['image_path']
    # validation_video = asset_data['video_path']
    # validation_camera_pose = asset_data['pose_file_path']
    # predict_type = asset_data['type']  # image2video, video2video, text2video, textimage2video

    config = OmegaConf.load(args.config_path)

    # Get Transformer
    Choosen_Transformer3DModel = name_to_transformer3d[config['transformer_additional_kwargs'].get('transformer_type', 'Transformer3DModel')]

    transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
    if args.weight_dtype == torch.float16:
        transformer_additional_kwargs["upcast_attention"] = True

    pose_encoder = CameraPoseEncoderCameraCtrl(**config['pose_encoder_kwargs'])
    pose_proj = VideoFrameTokenization()

    transformer = Choosen_Transformer3DModel.from_pretrained_2d(
        # model_name,
        args.transformer_model_name,
        subfolder="transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
        torch_dtype=torch.float8_e4m3fn if args.GPU_memory_mode == "model_cpu_offload_and_qfloat8" else args.weight_dtype,
        low_cpu_mem_usage=True,
        pose_encoder=pose_encoder,
        pose_proj=pose_proj,
    )

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if args.motion_module_path is not None:
        print(f"From Motion Module: {args.motion_module_path}")
        if args.motion_module_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(args.motion_module_path)
        else:
            state_dict = torch.load(args.motion_module_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}, {u}")

    # Get Vae
    Choosen_AutoencoderKL = name_to_autoencoder_magvit[config['vae_kwargs'].get('vae_type', 'AutoencoderKL')]
    vae = Choosen_AutoencoderKL.from_pretrained(args.model_name, subfolder="vae", vae_additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])).to(args.weight_dtype)
    if config['vae_kwargs'].get('vae_type', 'AutoencoderKL') == 'AutoencoderKLMagvit' and args.weight_dtype == torch.float16:
        vae.upcast_vae = True

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
        tokenizer = BertTokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
        tokenizer_2 = T5Tokenizer.from_pretrained(args.model_name, subfolder="tokenizer_2")
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
        tokenizer_2 = None

    if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
        text_encoder = BertModel.from_pretrained(args.model_name, subfolder="text_encoder", torch_dtype=args.weight_dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(args.model_name, subfolder="text_encoder_2", torch_dtype=args.weight_dtype)
    else:
        text_encoder = T5EncoderModel.from_pretrained(args.model_name, subfolder="text_encoder", torch_dtype=args.weight_dtype)
        text_encoder_2 = None

    if transformer.config.in_channels != vae.config.latent_channels and config['transformer_additional_kwargs'].get('enable_clip_in_inpaint', True):
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.model_name, subfolder="image_encoder").to("cuda", args.weight_dtype)
        clip_image_processor = CLIPImageProcessor.from_pretrained(args.model_name, subfolder="image_encoder")
    else:
        clip_image_encoder = None
        clip_image_processor = None

    # Get Scheduler
    Choosen_Scheduler = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM": DDIMScheduler,
    }[args.sampler_name]

    scheduler = Choosen_Scheduler.from_pretrained(args.model_name, subfolder="scheduler")

    pipeline = EasyAnimatePipelineCameraControl.from_pretrained(
        args.model_name,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=args.weight_dtype,
        clip_image_encoder=clip_image_encoder,
        clip_image_processor=clip_image_processor,
    )

    if args.GPU_memory_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload()
    elif args.GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        pipeline.enable_model_cpu_offload()
        convert_weight_dtype_wrapper(pipeline.transformer, args.weight_dtype)
    else:
        pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    if args.lora_path is not None:
        pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, "cuda")

    if vae.cache_mag_vae:
        video_length = int((video_length - 1) // vae.mini_batch_encoder * vae.mini_batch_encoder) + 1 if video_length != 1 else 1
    else:
        video_length = int(video_length // vae.mini_batch_encoder * vae.mini_batch_encoder) if video_length != 1 else 1

    for data in all_data:
        # If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
        predict_type = data['type']  # realestate_5dimension, kubric_low_level
        if predict_type == 'realestate_5dimension':
            prompt = data['text']
            negative_prompt = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
            clip_video_path = None
            groud_truth_path = data['groud_truth_path']
            pose_file_path = data['pose_file_path']
        elif predict_type == 'kubric_low_level':
            prompt = ''
            negative_prompt = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
            clip_video_path = data['clip_video_path']
            groud_truth_path = data['groud_truth_path']
            pose_file_path = data['pose_file_path']

        input_video, input_video_mask, clip_images, plucker_embedding = get_evaluation_model_input(
            groud_truth_path,
            clip_video_path=clip_video_path,
            pose_file_path=pose_file_path,
            video_length=video_length,
            fps=args.video_sample_stride,
            sample_size=args.sample_size,
        )

        # # realestate_5dimension, kubric_low_level
        # if predict_type in ['image2video', 'textimage2video']:
        #     input_video, input_video_mask, clip_images, ori_h, ori_w = get_image_to_video_latent(validation_image, None, video_length=video_length, sample_size=args.sample_size)
        # elif predict_type == 'video2video':
        #     input_video, input_video_mask, clip_images, ori_h, ori_w = get_video_to_video_latent(
        #         validation_video, video_length=video_length, fps=args.fps, sample_size=args.sample_size
        #     )
        # elif predict_type == 'text2video':
        #     input_video, input_video_mask, clip_images, ori_h, ori_w = get_image_to_video_latent(None, None, video_length=video_length, sample_size=args.sample_size)

        # plucker_embedding = get_plucker_embedding(validation_camera_pose, video_length, args.sample_size, ori_h, ori_w)
        # plucker_embedding = plucker_embedding.unsqueeze(0)  # torch.Size([1, 49, 6, 384, 672])

        with torch.no_grad():
            sample = pipeline(
                prompt,
                video_length=video_length,
                negative_prompt=negative_prompt,
                height=args.sample_size[0],
                width=args.sample_size[1],
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                video=input_video,
                mask_video=input_video_mask,
                clip_images=clip_images,
                strength=args.denoise_strength,
                plucker_embedding=plucker_embedding,
            ).videos

        if args.lora_path is not None:
            pipeline = unmerge_lora(pipeline, args.lora_path, args.lora_weight, "cuda")

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path, exist_ok=True)

        index = len([path for path in os.listdir(args.save_path)]) + 1
        prefix = str(index).zfill(8)

        if video_length == 1:
            save_sample_path = os.path.join(args.save_path, prefix + f".png")

            image = sample[0, :, 0]
            image = image.transpose(0, 1).transpose(1, 2)
            image = (image * 255).numpy().astype(np.uint8)
            image = Image.fromarray(image)
            image.save(save_sample_path)
        else:
            video_path = os.path.join(args.save_path, prefix + ".mp4")
            save_videos_grid(sample, video_path, fps=args.fps)


if __name__ == "__main__":
    args = parse_args()

    main(args)
