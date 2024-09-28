python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py --detector_path ./training/config/detector/clip.yaml --train_dataset "FaceForensics++" --test_dataset "Celeb-DF-v1" "Celeb-DF-v2"--task_target "clip" --no-save_feat --ddp

python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py --detector_path ./training/config/detector/xclip.yaml --train_dataset "FaceForensics++" --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" --task_target "xclip" --no-save_feat --ddp

python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py --detector_path ./training/config/detector/clip_wavelet.yaml --train_dataset "FaceForensics++" --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" --task_target "clip_wavelet" --no-save_feat --ddp

python3 -m torch.distributed.launch --nproc_per_node=4 training/train.py --detector_path ./training/config/detector/xclip_wavelet.yaml --train_dataset "FaceForensics++" --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" --task_target "xclip_wavelet" --no-save_feat --ddp