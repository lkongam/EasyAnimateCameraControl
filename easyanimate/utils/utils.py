import gc
import os

import cv2
import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from packaging import version as pver


def get_batch_index(length_of_video, video_sample_n_frames, video_sample_stride):
    if length_of_video >= video_sample_n_frames:

        def compute_stride(num_frames, desired_stride, num_samples):
            max_possible_stride = num_frames // num_samples
            if max_possible_stride == 0:
                return 1  # 最小步长为1
            return min(desired_stride, max_possible_stride)

        # 计算步长
        stride = compute_stride(length_of_video, video_sample_stride, video_sample_n_frames)

        def get_indices(start, num_frames, stride, total_length):
            indices = [start + i * stride for i in range(num_frames)]
            # 确保索引不超过总长度
            if indices[-1] >= total_length:
                # 如果最后一个索引超出范围，调整起始点
                start = total_length - (num_frames * stride)
                start = max(start, 0)
                indices = [start + i * stride for i in range(num_frames)]
            return indices

        # 获取输出部分的帧索引
        batch_index_output = get_indices(start=0, num_frames=video_sample_n_frames, stride=stride, total_length=length_of_video)
    else:
        # 当视频长度小于样本帧数时，均匀重复帧
        interval = length_of_video / video_sample_n_frames
        batch_index_output = [min(int(i * interval), length_of_video - 1) for i in range(video_sample_n_frames)]

    # 输入部分的帧索引与输出部分相同，全部为第一个索引
    batch_index_input = [batch_index_output[0]] * video_sample_n_frames

    return batch_index_input, batch_index_output


def get_video_from_dir(video_dir):
    cap = cv2.VideoCapture(video_dir)

    if not cap.isOpened():
        print(f"Cannot Open the Video File: {video_dir}")
        return []

    whole_video = []
    frame_count = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        whole_video.append(frame_pil)
        frame_count += 1

    cap.release()

    return whole_video, frame_height, frame_width


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

    if ori_h and ori_w:
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
            [
                cam_param.fx * video_sample_size[1],
                cam_param.fy * video_sample_size[0],
                cam_param.cx * video_sample_size[1],
                cam_param.cy * video_sample_size[0],
            ]
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


def get_width_and_height_from_image_and_base_resolution(image, base_resolution):
    target_pixels = int(base_resolution) * int(base_resolution)
    original_width, original_height = Image.open(image).size
    ratio = (target_pixels / (original_width * original_height)) ** 0.5
    width_slider = round(original_width * ratio)
    height_slider = round(original_height * ratio)
    return height_slider, width_slider


def color_transfer(sc, dc):
    """
    Transfer color distribution from of sc, referred to dc.

    Args:
        sc (numpy.ndarray): input image to be transfered.
        dc (numpy.ndarray): reference image

    Returns:
        numpy.ndarray: Transferred color distribution on the sc.
    """

    def get_mean_and_std(img):
        x_mean, x_std = cv2.meanStdDev(img)
        x_mean = np.hstack(np.around(x_mean, 2))
        x_std = np.hstack(np.around(x_std, 2))
        return x_mean, x_std

    sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
    return dst


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=12, imageio_backend=True, color_transfer_post_process=False):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1 / fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)


def get_image_to_video_latent(validation_image_start, validation_image_end, video_length, sample_size):
    if validation_image_start is not None and validation_image_end is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = Image.open(validation_image_start).convert("RGB")
            ori_h = image_start.size[1]
            ori_w = image_start.size[0]
            image_start = image_start.resize([sample_size[1], sample_size[0]])
        else:
            image_start = validation_image_start
            ori_h = image_start.size[1]
            ori_w = image_start.size[0]
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]

        if type(validation_image_end) is str and os.path.isfile(validation_image_end):
            image_end = Image.open(validation_image_end).convert("RGB")
            image_end = image_end.resize([sample_size[1], sample_size[0]])
        else:
            image_end = validation_image_end
            image_end = [_image_end.resize([sample_size[1], sample_size[0]]) for _image_end in image_end]

        if type(image_start) is list:
            start_video = torch.cat([torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], dim=2)
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, : len(image_start)] = start_video

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start) :] = 255
        else:
            input_video = torch.tile(torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), [1, 1, video_length, 1, 1])
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, 1:] = 255

        if type(image_end) is list:
            image_end = [_image_end.resize(image_start[0].size if type(image_start) is list else image_start.size) for _image_end in image_end]
            end_video = torch.cat([torch.from_numpy(np.array(_image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_end in image_end], dim=2)
            input_video[:, :, -len(end_video) :] = end_video

            input_video_mask[:, :, -len(image_end) :] = 0
        else:
            image_end = image_end.resize(image_start[0].size if type(image_start) is list else image_start.size)
            input_video[:, :, -1:] = torch.from_numpy(np.array(image_end)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0)
            input_video_mask[:, :, -1:] = 0

        input_video = input_video / 255

    elif validation_image_start is not None:
        if type(validation_image_start) is str and os.path.isfile(validation_image_start):
            image_start = Image.open(validation_image_start).convert("RGB")
            ori_h = image_start.size[1]
            ori_w = image_start.size[0]
            image_start = image_start.resize([sample_size[1], sample_size[0]])
        else:
            image_start = validation_image_start
            ori_h = image_start.size[1]
            ori_w = image_start.size[0]
            image_start = [_image_start.resize([sample_size[1], sample_size[0]]) for _image_start in image_start]
        image_end = None

        if type(image_start) is list:
            start_video = torch.cat([torch.from_numpy(np.array(_image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0) for _image_start in image_start], dim=2)
            input_video = torch.tile(start_video[:, :, :1], [1, 1, video_length, 1, 1])
            input_video[:, :, : len(image_start)] = start_video
            input_video = input_video / 255

            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[:, :, len(image_start) :] = 255
        else:
            input_video = torch.tile(torch.from_numpy(np.array(image_start)).permute(2, 0, 1).unsqueeze(1).unsqueeze(0), [1, 1, video_length, 1, 1]) / 255
            input_video_mask = torch.zeros_like(input_video[:, :1])
            input_video_mask[
                :,
                :,
                1:,
            ] = 255

    else:
        image_start = None
        image_end = None
        ori_h = None
        ori_w = None
        input_video = torch.zeros([1, 3, video_length, sample_size[0], sample_size[1]])
        input_video_mask = torch.ones([1, 1, video_length, sample_size[0], sample_size[1]]) * 255

    # clip_images 的形状是 [bs, 49, 384, 672, 3]
    clip_images = input_video.permute(0, 2, 3, 4, 1).contiguous()
    clip_images = (clip_images * 0.5 + 0.5) * 255

    del image_start
    del image_end
    gc.collect()

    return input_video, input_video_mask, clip_images, ori_h, ori_w


def get_video_to_video_latent(input_video_path, video_length, sample_size, fps=None, validation_video_mask=None, ref_image=None):
    if isinstance(input_video_path, str):
        cap = cv2.VideoCapture(input_video_path)
        input_video = []

        # original_fps = cap.get(cv2.CAP_PROP_FPS)
        # frame_skip = 1 if fps is None else int(original_fps // fps)

        frame_skip = 1
        # 获取视频的宽度和高度
        ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frame = cv2.resize(frame, (sample_size[1], sample_size[0]))
                input_video.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            frame_count += 1

        cap.release()
    else:
        input_video = input_video_path

    input_video = torch.from_numpy(np.array(input_video))[:video_length]
    input_video = input_video.permute([3, 0, 1, 2]).unsqueeze(0) / 255

    if ref_image is not None:
        ref_image = Image.open(ref_image)
        ref_image = torch.from_numpy(np.array(ref_image))
        ref_image = ref_image.unsqueeze(0).permute([3, 0, 1, 2]).unsqueeze(0) / 255

    if validation_video_mask is not None:
        validation_video_mask = Image.open(validation_video_mask).convert('L').resize((sample_size[1], sample_size[0]))
        input_video_mask = np.where(np.array(validation_video_mask) < 240, 0, 255)

        input_video_mask = torch.from_numpy(np.array(input_video_mask)).unsqueeze(0).unsqueeze(-1).permute([3, 0, 1, 2]).unsqueeze(0)
        input_video_mask = torch.tile(input_video_mask, [1, 1, input_video.size()[2], 1, 1])
        input_video_mask = input_video_mask.to(input_video.device, input_video.dtype)
    else:
        input_video_mask = torch.zeros_like(input_video[:, :1])
        input_video_mask[:, :, 1:] = 255

    clip_images = input_video.permute(0, 2, 3, 4, 1).contiguous()
    clip_images = (clip_images * 0.5 + 0.5) * 255

    return input_video, input_video_mask, clip_images, ori_h, ori_w


def get_evaluation_model_input(groud_truth_path, clip_video_path, pose_file_path, video_length, video_sample_stride, sample_size):
    GT_video, ori_h, ori_w = get_video_from_dir(groud_truth_path)
    whole_camera_para = get_camera_from_dir(pose_file_path)
    assert len(GT_video) == len(whole_camera_para), "the length of video must be euqal with that of camera parameter."

    if clip_video_path is not None:
        pass
    else:
        batch_index_input, batch_index_output = get_batch_index(len(GT_video), video_length, video_sample_stride)
        plucker_embedding_input, plucker_embedding_output = get_plucker_embedding(
            whole_camera_para,
            batch_index_input,
            batch_index_output,
            sample_size,
            ori_h,
            ori_w,
        )
        # pixel_values_input, pixel_values_output = get_pixel_value(GT_video, batch_index_input, batch_index_output, video_transforms)
        # input_video

    # return input_video, input_video_mask, clip_images, plucker_embedding
