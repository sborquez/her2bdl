#!/bin/bash

docker run -it \
      --entrypoint /bin/bash \
      --name her2bdl --gpus all --rm -p 8888:8888 \
     -v "/mnt/storage-lite/archived_projects/her2bdl/scripts/datasets":/datasets \
     -v "/mnt/storage-lite/archived_projects/her2bdl/scripts/experiments":/experiments \
     -v "$(pwd)":/her2bdl_dev \
     -e HER2BDL_HOME=/her2bdl_dev \
     her2bdl2:latest