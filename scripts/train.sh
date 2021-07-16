#! /bin/sh

cd ../
python ./tools_cam/train_cam.py --config_file ./configs/ILSVRC/deit_cam_small_patch16_224.ymal --lr 5e-4
