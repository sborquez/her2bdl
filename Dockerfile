FROM  continuumio/miniconda3
LABEL Author, Sebastian Borquez

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . $APP_HOME

# The environment variables set using ENV will persist when a container is run from the resulting image. You can view the values using docker inspect, and change them using docker run --env <key>=<value>.

# TODO: Add GPU support
# ARG enable_gpu

#---------------- Prepare the envirennment
RUN conda update --name base conda &&\
    conda env create --file environment.yaml
SHELL ["conda", "run", "--name", "her2dbl", "/bin/bash", "-c"]

#CMD 
#ENTRYPOINT ["conda", "run", "--name", "her2dbl", "python", "main.py"]