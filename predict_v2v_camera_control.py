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
from easyanimate.pipeline.pipeline_easyanimate_camera_control import EasyAnimatePipelineCameraControl
from easyanimate.utils.lora_utils import merge_lora, unmerge_lora
from easyanimate.utils.utils import get_image_to_video_latent, get_video_to_video_latent, save_videos_grid
from easyanimate.utils.fp8_optimization import convert_weight_dtype_wrapper
from easyanimate.models.pose_encoder import CameraPoseEncoderCameraCtrl, VideoFrameTokenization
from packaging import version as pver


def get_camera_from_dir(pose_file_dir):
    whole_camera_para = []

    try:
        with open(pose_file_dir, 'r', encoding='utf-8') as file:
            # 读取所有行
            lines = file.readlines()

            # 确保文件至少有两行
            if len(lines) < 2:
                print("文件内容不足两行，无法读取数据。")
                return whole_camera_para

            # 跳过第一行，从第二行开始处理
            for idx, line in enumerate(lines[1:], start=2):
                # 去除首尾空白字符并按空格分割
                parts = line.strip().split()

                # 检查每行是否有19个数字
                if len(parts) != 19:
                    print(f"警告：第 {idx} 行的数字数量不是19，跳过该行。")
                    continue

                try:
                    # 将字符串转换为浮点数
                    numbers = [float(part) for part in parts]
                    whole_camera_para.append(numbers)
                except ValueError as ve:
                    print(f"警告：第 {idx} 行包含非数字字符，跳过该行。错误详情: {ve}")
                    continue

    except FileNotFoundError:
        print(f"错误：文件 '{pose_file_dir}' 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

    return whole_camera_para


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        try:
            self.c2w_mat = np.linalg.inv(w2c_mat_4x4)
        except:
            self.c2w_mat = np.linalg.pinv(w2c_mat_4x4)


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype), torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype))
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1
    fx += 1e-10
    fy += 1e-10
    zs = torch.ones_like(i)  # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)  # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

    target_cam_c2w = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def compute_plucker(cam_params, video_sample_size, ori_h, ori_w):

    cam_params = [Camera(cam_param) for cam_param in cam_params]

    ori_wh_ratio = ori_w / ori_h
    sample_wh_ratio = video_sample_size[1] / video_sample_size[0]
    if ori_wh_ratio > sample_wh_ratio:  # rescale fx
        resized_ori_w = video_sample_size[0] * ori_wh_ratio
        for cam_param in cam_params:
            cam_param.fx = resized_ori_w * cam_param.fx / video_sample_size[1]
    else:  # rescale fy
        resized_ori_h = video_sample_size[1] / ori_wh_ratio
        for cam_param in cam_params:
            cam_param.fy = resized_ori_h * cam_param.fy / video_sample_size[0]

    intrinsics = np.asarray(
        [
            [cam_param.fx * video_sample_size[1], cam_param.fy * video_sample_size[0], cam_param.cx * video_sample_size[1], cam_param.cy * video_sample_size[0]]
            for cam_param in cam_params
        ],
        dtype=np.float32,
    )

    intrinsics = torch.as_tensor(intrinsics)[None]  # [1, n_frame, 4]
    c2w_poses = get_relative_pose(cam_params)
    c2w = torch.as_tensor(c2w_poses)[None]  # [1, n_frame, 4, 4]
    plucker_embedding = ray_condition(intrinsics, c2w, video_sample_size[0], video_sample_size[1], device='cpu')[0].permute(0, 3, 1, 2).contiguous()

    return plucker_embedding


def get_plucker_embedding(camera_pose, video_length, sample_size, ori_h, ori_w):
    camera_para = get_camera_from_dir(camera_pose)[:video_length]
    plucker_embedding = compute_plucker(camera_para, sample_size, ori_h, ori_w)

    return plucker_embedding


def get_empty_plucker_embedding(pixel_values_input, direction_number=6):
    shape = list(pixel_values_input.shape)
    shape[1] = direction_number
    shape = tuple(shape)
    dtype = pixel_values_input.dtype
    device = pixel_values_input.device
    plucker_in = torch.zeros(shape, device=device, dtype=dtype)
    # plucker_out = torch.zeros(shape, device=device, dtype=dtype)

    return plucker_in


def main(asset_data):
    # GPU memory mode, which can be choosen in [model_cpu_offload, model_cpu_offload_and_qfloat8, sequential_cpu_offload].
    # model_cpu_offload means that the entire model will be moved to the CPU after use, which can save some GPU memory.
    #
    # model_cpu_offload_and_qfloat8 indicates that the entire model will be moved to the CPU after use,
    # and the transformer model has been quantized to float8, which can save more GPU memory.
    #
    # sequential_cpu_offload means that each layer of the model will be moved to the CPU after use,
    # resulting in slower speeds but saving a large amount of GPU memory.
    GPU_memory_mode = "model_cpu_offload"

    # Config and model path
    config_path = "config/easyanimate_video_v5_magvit_camera_control.yaml"
    model_name = "models/Diffusion_Transformer/EasyAnimateV5-7b-zh-CameraControl"
    transformer_model_name = "output_dir_20241205_test/checkpoint-106"
    # pose_encoder_pretrained = "models/Camera_Pose/CameraCtrl_svd.ckpt"

    # Choose the sampler in "Euler" "Euler A" "DPM++" "PNDM" and "DDIM"
    # EasyAnimateV1, V2 and V3 cannot use DDIM.
    # EasyAnimateV4 and V5 support DDIM.
    sampler_name = "DDIM"

    # Load pretrained model if need
    transformer_path = None
    # Only V1 does need a motion module
    motion_module_path = None
    vae_path = None
    lora_path = None

    # Other params
    sample_size = [384, 672]
    # In EasyAnimateV1, the video_length of video is 40 ~ 80.
    # In EasyAnimateV2, V3, V4, the video_length of video is 1 ~ 144.
    # In EasyAnimateV5, the video_length of video is 1 ~ 49.
    # If u want to generate a image, please set the video_length = 1.
    video_length = 49
    fps = 8

    # Use torch.float16 if GPU does not support torch.bfloat16
    # ome graphics cards, such as v100, 2080ti, do not support torch.bfloat16
    weight_dtype = torch.bfloat16
    # If you want to generate from text, please set the validation_image_start = None and validation_image_end = None
    prompt = asset_data['text']
    negative_prompt = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
    validation_image = asset_data['image_path']
    validation_video = asset_data['video_path']
    validation_camera_pose = asset_data['pose_file_path']
    predict_type = asset_data['type']  # image2video, video2video, text2video, textimage2video
    denoise_strength = 0.70

    # EasyAnimateV1, V2 and V3 support English.
    # EasyAnimateV4 and V5 support English and Chinese.
    # 使用更长的neg prompt如"模糊，突变，变形，失真，画面暗，文本字幕，画面固定，连环画，漫画，线稿，没有主体。"，可以增加稳定性
    # 在neg prompt中添加"安静，固定"等词语可以增加动态性。

    #
    # Using longer neg prompt such as "Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art." can increase stability
    # Adding words such as "quiet, solid" to the neg prompt can increase dynamism.
    # prompt                  = "A cute cat is playing the guitar. "
    # negative_prompt         = "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code."
    guidance_scale = 6.0
    seed = 43
    num_inference_steps = 50
    lora_weight = 0.55
    save_path = "samples/easyanimate-videos_v2v"

    config = OmegaConf.load(config_path)

    # Get Transformer
    Choosen_Transformer3DModel = name_to_transformer3d[config['transformer_additional_kwargs'].get('transformer_type', 'Transformer3DModel')]

    transformer_additional_kwargs = OmegaConf.to_container(config['transformer_additional_kwargs'])
    if weight_dtype == torch.float16:
        transformer_additional_kwargs["upcast_attention"] = True

    pose_encoder = CameraPoseEncoderCameraCtrl(**config['pose_encoder_kwargs'])
    pose_proj = VideoFrameTokenization()

    transformer = Choosen_Transformer3DModel.from_pretrained_2d(
        # model_name,
        transformer_model_name,
        subfolder="transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
        torch_dtype=torch.float8_e4m3fn if GPU_memory_mode == "model_cpu_offload_and_qfloat8" else weight_dtype,
        low_cpu_mem_usage=True,
        pose_encoder=pose_encoder,
        pose_proj=pose_proj,
    )

    if transformer_path is not None:
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if motion_module_path is not None:
        print(f"From Motion Module: {motion_module_path}")
        if motion_module_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(motion_module_path)
        else:
            state_dict = torch.load(motion_module_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}, {u}")

    # Get Vae
    Choosen_AutoencoderKL = name_to_autoencoder_magvit[config['vae_kwargs'].get('vae_type', 'AutoencoderKL')]
    vae = Choosen_AutoencoderKL.from_pretrained(model_name, subfolder="vae", vae_additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])).to(weight_dtype)
    if config['vae_kwargs'].get('vae_type', 'AutoencoderKL') == 'AutoencoderKLMagvit' and weight_dtype == torch.float16:
        vae.upcast_vae = True

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open

            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
        tokenizer = BertTokenizer.from_pretrained(model_name, subfolder="tokenizer")
        tokenizer_2 = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    else:
        tokenizer = T5Tokenizer.from_pretrained(model_name, subfolder="tokenizer")
        tokenizer_2 = None

    if config['text_encoder_kwargs'].get('enable_multi_text_encoder', False):
        text_encoder = BertModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=weight_dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder_2", torch_dtype=weight_dtype)
    else:
        text_encoder = T5EncoderModel.from_pretrained(model_name, subfolder="text_encoder", torch_dtype=weight_dtype)
        text_encoder_2 = None

    if transformer.config.in_channels != vae.config.latent_channels and config['transformer_additional_kwargs'].get('enable_clip_in_inpaint', True):
        clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_name, subfolder="image_encoder").to("cuda", weight_dtype)
        clip_image_processor = CLIPImageProcessor.from_pretrained(model_name, subfolder="image_encoder")
    else:
        clip_image_encoder = None
        clip_image_processor = None

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM": DDIMScheduler,
    }[sampler_name]

    scheduler = Choosen_Scheduler.from_pretrained(model_name, subfolder="scheduler")

    pipeline = EasyAnimatePipelineCameraControl.from_pretrained(
        model_name,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
        clip_image_encoder=clip_image_encoder,
        clip_image_processor=clip_image_processor,
    )

    if GPU_memory_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload()
    elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
        pipeline.enable_model_cpu_offload()
        convert_weight_dtype_wrapper(pipeline.transformer, weight_dtype)
    else:
        pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight, "cuda")

    if vae.cache_mag_vae:
        video_length = int((video_length - 1) // vae.mini_batch_encoder * vae.mini_batch_encoder) + 1 if video_length != 1 else 1
    else:
        video_length = int(video_length // vae.mini_batch_encoder * vae.mini_batch_encoder) if video_length != 1 else 1

    # image2video, video2video, text2video, textimage2video
    if predict_type in ['image2video', 'textimage2video']:
        input_video, input_video_mask, clip_images, ori_h, ori_w = get_image_to_video_latent(validation_image, None, video_length=video_length, sample_size=sample_size)
    elif predict_type == 'video2video':
        input_video, input_video_mask, clip_images, ori_h, ori_w = get_video_to_video_latent(validation_video, video_length=video_length, fps=fps, sample_size=sample_size)
    elif predict_type == 'text2video':
        input_video, input_video_mask, clip_images, ori_h, ori_w = get_image_to_video_latent(None, None, video_length=video_length, sample_size=sample_size)

    plucker_embedding = get_plucker_embedding(validation_camera_pose, video_length, sample_size, ori_h, ori_w)
    plucker_embedding = plucker_embedding.unsqueeze(0)

    with torch.no_grad():
        sample = pipeline(
            prompt,
            video_length=video_length,
            negative_prompt=negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            video=input_video,
            mask_video=input_video_mask,
            clip_images=clip_images,
            strength=denoise_strength,
            plucker_embedding=plucker_embedding,
        ).videos

    if lora_path is not None:
        pipeline = unmerge_lora(pipeline, lora_path, lora_weight, "cuda")

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    index = len([path for path in os.listdir(save_path)]) + 1
    prefix = str(index).zfill(8)

    if video_length == 1:
        save_sample_path = os.path.join(save_path, prefix + f".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(save_sample_path)
    else:
        video_path = os.path.join(save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=fps)


if __name__ == "__main__":
    assets_json_path = "asset/predict_data.json"
    with open(assets_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    main(data[0])
