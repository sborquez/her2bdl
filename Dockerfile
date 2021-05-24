FROM  continuumio/miniconda3
LABEL Author, Sebastian Borquez

# Define a variable with an optional default value that can be override at 
# build-time with docker build.
ARG HER2BDL_BRANCH=dev
# TODO: change it when merge to master
#ARG HER2BDL_BRANCH=master 

# The environment variables set using ENV will persist when a container is run 
# from the resulting image. You can view the values using docker inspect, and 
# change them using docker run --env <key>=<value>.
ENV WAND_AND_KEY=dryrun

#---------------- Prepare the envirennment ----------------
RUN apt-get update && apt-get install gcc ffmpeg libsm6 libxext6  -y
RUN git clone -b ${HER2BDL_BRANCH} --single-branch \
    https://github.com/sborquez/her2bdl 
ENV HER2BDL_HOME /her2bdl
WORKDIR $HER2BDL_HOME

# Dataset folder mounted from host
ENV HER2BDL_DATASETS /datasets
RUN mkdir ${HER2BDL_DATASETS}
# Configurations files and runs
ENV HER2BDL_EXPERIMENTS /experiments
RUN mkdir ${HER2BDL_EXPERIMENTS}
# Host repository folder mounted from host to local development
ENV HER2BDL_DEV /her2bdl_dev
RUN mkdir ${HER2BDL_DEV}

RUN conda update --name base conda &&\
    conda env create --file environment_cpu.yml
RUN echo "conda activate her2bdl" >> ~/.bashrc
SHELL ["conda", "run", "--name", "her2bdl", "/bin/bash", "-c"]

#CMD 
# TODO: container commands
# ENTRYPOINT ["conda", "run", "--name", "her2dbl", "python", "main.py"]
