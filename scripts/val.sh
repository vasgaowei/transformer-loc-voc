#! /bin/sh
export CUDA_VISIBLE_DEVICES=7

cd ../
python ./tools_cam/test_cam.py --config_file configs/ILSVRC/deit_cam_small_patch16_224.ymal --resume ./ckpt/ImageNet/deit_cam_small_patch16_224_CAM-NORMAL_SEED26_CAM-THR0.3_BS2048_2021-03-14-21-35/ckpt/model_best_top1_loc.pth  MODEL.CAM_THR 0.25
