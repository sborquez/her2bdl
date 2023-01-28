conda install -y python=3.8
conda install -y -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -y cuda -c nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
pip install tensorflow
conda install -y -c conda-forge openslide-python
conda install -y numpy pandas xlrd matplotlib scipy scikit-learn scikit-image plotly pyyaml seaborn IPython
pip install twine nose tqdm wandb pycm==3.0 openpyxl opencv-python