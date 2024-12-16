export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5-7b-zh-CameraControl"
export POSE_ADAPTOR_CKPT="models/Camera_Pose/CameraCtrl_svdxt.ckpt"
export DATASET_NAME="/mnt/chenyang_lei/Datasets/easyanimate_dataset"
export DATASET_META_NAME="/mnt/chenyang_lei/Datasets/easyanimate_dataset/metadata_4000.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json \
  --deepspeed_multinode_launcher standard \
  --main_process_port 29502 \
  scripts/train_v2v_camera_control_V2.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --pose_adaptor_ckpt=$POSE_ADAPTOR_CKPT \
  --config_path "config/easyanimate_video_v5_magvit_camera_control_v2.yaml" \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --checkpointing_epochs=1 \
  --checkpoints_total_limit=1 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="output_dir_20241211" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=5e-3 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --not_sigma_loss \
  --uniform_sampling \
  --use_deepspeed \
  --train_mode="CameraControl" \
  --trainable_modules "." \
  --tracker_project_name="v2v-camera-control-finetune"