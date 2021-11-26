export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
export PORT=14122
export MYPATH=/workspace/mmcd


# python -c "import time; time.sleep(int(4.1*60*60))"

cd ${MYPATH}

bash ./tools/dist_train.sh configs/cd_stb/upernet_hr40_576x576_stb.py ${NUM_GPUS} ${PORT} --work-dir ./work_dirs/cd_stb/upernet_hr40_576x576_stb

