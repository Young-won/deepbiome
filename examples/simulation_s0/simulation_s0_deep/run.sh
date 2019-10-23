export CUDA_VISIBLE_DEVICES=2
echo $CUDA_VISIBLE_DEVICES

model=${PWD##*/}
echo $model

python3 ../../main.py --kfold=5 --log_info=config/log_info.yaml --path_info=config/path_info.cfg --network_info=config/network_info.cfg  --max_queue_size=20 --workers=1 --use_multiprocessing=False
