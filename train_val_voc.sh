GPU_ID=$1
NET=${2}
SIZE=${3}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python ./tools_cam/train_cam_cor_loc.py --config_file ./configs/VOC/deit_cam_${NET}_patch16_${SIZE}.ymal --lr 5e-5 MODEL.CAM_THR 0.2