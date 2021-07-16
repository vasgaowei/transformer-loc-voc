#! /bin/sh
export CUDA_VISIBLE_DEVICES=7

cd ../
python ./tools_cam/test_cam.py --config_file configs/CUB/deit_cam_small_patch16_224.ymal --resume ./ckpt/CUB/weigao/model_epoch60.pth  MODEL.CAM_THR 0.1
