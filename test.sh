echo "*******************executing for xception*******************"
python3 training/test.py --detector_path ./training/config/detector/xception.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/xception_best.pth
echo "*********executing for xception completed*******************"

echo "*******************executing for capsule_net*******************"
python3 training/test.py --detector_path ./training/config/detector/capsule_net.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/capsule_best.pth
echo "*********executing for capsule_net completed*******************"

echo "*******************executing for core*******************"
python3 training/test.py --detector_path ./training/config/detector/core.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/core_best.pth
echo "*********executing for core completed*******************"

echo "*******************executing for f3net*******************"
python3 training/test.py --detector_path ./training/config/detector/f3net.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/f3net_best.pth
echo "*********executing for f3net completed*******************"

echo "*******************executing for meso4*******************"
python3 training/test.py --detector_path ./training/config/detector/meso4.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/meso4_best.pth
echo "*********executing for meso4 completed*******************"

echo "*******************executing for recce*******************"
python3 training/test.py --detector_path ./training/config/detector/recce.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/recce_best.pth
echo "*********executing for recce completed*******************"

echo "*******************executing for srm*******************"
python3 training/test.py --detector_path ./training/config/detector/srm.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/srm_best.pth
echo "*********executing for srm completed*******************"

echo "*******************executing for ffd*******************"
python3 training/test.py --detector_path ./training/config/detector/ffd.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/ffd_best.pth
echo "*********executing for ffd completed*******************"

echo "*******************executing for meso4Inception*******************"
python3 training/test.py --detector_path ./training/config/detector/meso4Inception.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/meso4Incep_best.pth
echo "*********executing for meso4Inception completed*******************"

echo "*******************executing for spsl*******************"
python3 training/test.py --detector_path ./training/config/detector/spsl.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/spsl_best.pth
echo "*********executing for spsl completed*******************"


echo "*******************executing for clip*******************"
python3 training/test.py --detector_path ./training/config/detector/clip.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/clip_best.pth
echo "*********executing for clip completed*******************"


echo "*******************executing for clip_wavelet*******************"
python3 training/test.py --detector_path ./training/config/detector/clip_wavelet.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/clip_wavelet_best.pth
echo "*********executing for clip_wavelet completed*******************"


echo "*******************executing for xclip*******************"
python3 training/test.py --detector_path ./training/config/detector/xclip.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/xclip_best.pth
echo "*********executing for xclip completed*******************"

echo "*******************executing for xclip_wavelet*******************"
python3 training/test.py --detector_path ./training/config/detector/xclip_wavelet.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2" "FaceShifter" --weights_path ./training/weights/xclip_wavelet_best.pth
echo "*********executing for xclip_wavelet completed*******************"




# python3 training/test.py --detector_path ./training/config/detector/clip.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2"  "FaceShifter" --weights_path /home/lalith.bharadwaj/DeepfakeBench/logs/training/clip_Clip_wave_head_2024-09-05-22-29-36/test/avg/ckpt_best.pth

# python3 training/test.py --detector_path ./training/config/detector/xclip.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2"  "FaceShifter" --weights_path /home/lalith.bharadwaj/DeepfakeBench/logs/training/xclip_xclip_2024-09-04-22-07-29/test/avg/ckpt_best.pth

# python3 training/test.py --detector_path ./training/config/detector/xclip.yaml --test_dataset "Celeb-DF-v1" "Celeb-DF-v2"  "FaceShifter" --weights_path /home/lalith.bharadwaj/DeepfakeBench/logs/training/xclip_xclip_wave_head_2024-09-09-00-31-20/test/avg/ckpt_best.pth

