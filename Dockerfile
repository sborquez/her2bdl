FROM  continuumio/miniconda3
LABEL Author, Sebastian Borquez

# Define a variable with an optional default value that can be override at 
# build-time with docker build.
ARG HER2BDL_VERSION=dev
# TODO: change it when merge to master
#ARG HER2BDL_VERSION=master 

# The environment variables set using ENV will persist when a container is run 
# from the resulting image. You can view the values using docker inspect, and 
# change them using docker run --env <key>=<value>.
ENV WAND_AND_KEY=dryrun

#---------------- Prepare the envirennment ----------------
RUN apt-get update && apt-get install -y gcc
RUN git clone -b ${HER2BDL_VERSION} --single-branch \
    https://github.com/sborquez/her2bdl 
ENV APP_HOME /her2bdl
WORKDIR $APP_HOME

# Dataset folder mounted from host
ENV APP_DATA /datasets
RUN mkdir ${APP_DATA}
# Configurations files and runs
ENV APP_EXPERIMENTS /experiments
RUN mkdir ${APP_EXPERIMENTS}
# Host repository folder mounted from host to local development
ENV APP_DEV /her2bdl_dev
RUN mkdir ${APP_DEV}

RUN conda update --name base conda &&\
    conda env create --file environment_cpu.yml

SHELL ["conda", "run", "--name", "her2bdl", "/bin/bash", "-c"]

#CMD 
# TODO: container commands
# ENTRYPOINT ["conda", "run", "--name", "her2dbl", "python", "main.py"]
