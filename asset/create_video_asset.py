import cv2
import sys


def extract_frames(input_path, output_path, start_frame, num_frames):
    # 打开输入视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {input_path}")
        sys.exit(1)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"总帧数: {total_frames}")

    # 检查是否有足够的帧
    if start_frame + num_frames > total_frames:
        print(f"请求的帧范围超出视频总帧数。视频总帧数: {total_frames}, 请求到的最后一帧: {start_frame + num_frames -1}")
        cap.release()
        sys.exit(1)

    # 获取视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"帧率: {fps}, 分辨率: {width}x{height}")

    # 定义视频编写器，使用与输入视频相同的编解码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID', 'avc1' 等
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 设置开始读取的帧位置（OpenCV使用0基索引，所以第73帧对应索引72）
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    current_frame = start_frame
    frames_extracted = 0

    print(f"开始提取帧 {start_frame} 到 {start_frame + num_frames -1}")

    while frames_extracted < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("意外地未能读取到所有请求的帧。")
            break
        out.write(frame)
        frames_extracted += 1
        current_frame += 1
        if frames_extracted % 10 == 0 or frames_extracted == num_frames:
            print(f"已提取 {frames_extracted}/{num_frames} 帧")

    # 释放资源
    cap.release()
    out.release()
    print(f"帧提取完成，保存到 {output_path}")


if __name__ == "__main__":
    # 示例用法
    input_video = "asset/20240719024331_view_0_1.mp4"  # 输入视频文件路径
    output_video = "asset/20240719024331_view_0_1_out.mp4"  # 输出视频文件路径
    start = 61  # 开始帧（从1开始计数）
    num = 49  # 要提取的帧数

    extract_frames(input_video, output_video, start, num)
