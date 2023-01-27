# conda create --name her2bdl2
# conda activate her2bdl2
conda install -y python=3.6
conda install -y -c conda-forge cudatoolkit=11.0 cudnn=8.0.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install tensorflow==2.4.1
conda install -y -c conda-forge openslide-python
conda install -y numpy pandas h5py xlrd matplotlib scipy scikit-learn scikit-image plotly pyyaml seaborn IPython
pip install twine nose tqdm wandb pycm==3.0 openpyxl opencv-python