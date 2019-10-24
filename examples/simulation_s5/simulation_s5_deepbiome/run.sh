export CUDA_VISIBLE_DEVICES=3
echo $CUDA_VISIBLE_DEVICES

model=${PWD##*/}
echo $model

python3 ../../main.py --log_info=config/log_info.yaml --path_info=config/path_info.cfg --network_info=config/network_info.cfg  --max_queue_size=50 --workers=10 --use_multiprocessing=False
