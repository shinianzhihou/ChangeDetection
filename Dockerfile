FROM registry-vpc.cn-hangzhou.aliyuncs.com/carryhjr/rsipac:0.1

COPY . /workspace/
COPY run.py /workspace/
WORKDIR /workspace

RUN apt-get update && pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html
RUN ls ./ && ls /workspace
RUN pip install -r mmcd/requirements.txt


CMD ["python","run.py","/input_path","/output_path"]



