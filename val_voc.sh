GPU_ID=$1
NET=${2}
SIZE=${3}
SAVE_PATH=${4}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

python ./tools_cam/test_cam_cor_loc.py --config_file configs/VOC/deit_cam_${NET}_patch16_${SIZE}.ymal --resume ${SAVE_PATH} TEST.SAVE_BOXED_IMAGE True MODEL.CAM_THR 0.2