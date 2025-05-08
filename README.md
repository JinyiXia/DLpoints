# DLpoints

This is created to test different deep learning models in separating the leaf and wood points from high-density TLS (Terrestrial Laser Scan) data.

## Windows environment
I use a Silva Lab laptop with an NVIDIA Quadro RTX 5000 GPU.

The basic structure is from the PyTorch library torch-points3d. The first important step will be installing the library.\
I'm following the steps from https://github.com/torch-points3d/torch-points3d/issues/803. Also, according to https://github.com/torch-points3d/torch-points-kernels/issues/80, downgrading to CUDA 11.1 solves the issue. 

Install visual studio 2019 (https://learn.microsoft.com/en-us/visualstudio/releases/2019/redistribution#vs2019-download) and cuda 11.1 (https://developer.nvidia.com/cuda-11.1.1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)
```
conda create -n torch-points python=3.8 vc=14.2
conda activate torch-points
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-cluster==1.5.9 torch-scatter==2.0.9 torch-sparse==0.6.11 -f https://pytorch-geometric.com/whl/torch-1.9.1%2Bcu111.html
C:\Users\xjy11\.conda\envs\torch-points\python.exe -m pip install "pip<24.0"
git clone https://github.com/torch-points3d/torch-points3d.git
cd torch-points3d
pip install -r requirements.txt
C:\Users\jinyixia\AppData\Local\anaconda3\condabin\conda.bat install -c conda-forge hdbscan==0.8.28
```

## Linux environment
I am using supercomputer HiperGator from UF:
- Number of CPU cores: 3
- Maximum memory on the compute node requested for the job in Gigabytes: 32 (Need to be more than 23 because that's how much PyTorch will ask to get allocated
- Cluster partition: gpu
- Generic Resource Request: gpu:a100:1

### 1. Check if gcc is greater than version 9:
```
gcc --version
```
If not:
```
module spider GCC #check the available version of GCC
module load gcc/9.3.0 #load the GCC
gcc --version #double check the version
```
### 2. Load conda:
```
module load conda
```
If it shows 'No ~/.condarc found, creating a new config from HPG defaults', run ```module load conda``` again. Nothing should show up after running this.\
Create a virtual environment:
```
conda create -n torch-points3d python=3.7 cudatoolkit=11.1
conda activate torch-points3d
```
### 3. Install PyTorch:
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
```
*To test if the installation is correct, use Python to test ```python -c "import torch; print(torch.cuda.is_available())"``` ```python -c "import torch_cluster; print(torch_cluster.__file__)"``` ```python -c "import torch_scatter; print(torch_scatter.__file__)"```, the result should be True
### 4. Install requirements:
```
git clone https://github.com/torch-points3d/torch-points3d.git
cd torch-points3d
pip install -r requirements.txt
```
If the error shows:
```
Collecting sklearn (from open3d==0.12.0)
  Downloading sklearn-0.0.post12.tar.gz (2.6 kB)
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [15 lines of output]
      The 'sklearn' PyPI package is deprecated, use 'scikit-learn'
      rather than 'sklearn' for pip commands.
      
      Here is how to fix this error in the main use cases:
      - use 'pip install scikit-learn' rather than 'pip install sklearn'
      - replace 'sklearn' by 'scikit-learn' in your pip requirements files
        (requirements.txt, setup.py, setup.cfg, Pipfile, etc ...)
      - if the 'sklearn' package is used by one of your dependencies,
        it would be great if you take some time to track which package uses
        'sklearn' instead of 'scikit-learn' and report it to their issue tracker
      - as a last resort, set the environment variable
        SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True to avoid this error
      
      More information is available at
      https://github.com/scikit-learn/sklearn-pypi-package
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.
```
Run the following:
```
pip install scikit-learn==1.0
pip install open3d==0.12.0
```
Then run ```pip install -r requirements.txt``` again.

*To test if the installation is correct, use Python to test ```python -c "import torch_points_kernels.points_cuda; print('torch_points_kernels.points_cuda OK')"```.
### 5. Install MinkowskiEngine and TorchSparse
```
conda install -y ninja
conda install -c conda-forge openblas-devel
pip install -U "MinkowskiEngine==0.5.4" -f https://nvidia-minkowski-engine.s3.us-west-2.amazonaws.com/torch-1.8.1-cuda111.html
pip install torch_sparse -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```
*To test if the installation is correct, use Python to test ```python -c "import MinkowskiEngine as ME; print('MinkowskiEngine OK')"``` ```python -c "import torchsparse; print('TorchSparse OK')"``` ```python -c "import torch_sparse; print('torch_sparse OK')"```.
### 6. Install torch-points3d:
```
wget https://files.pythonhosted.org/packages/c5/f1/f3af914effa74b9a20cec6d27896ade54c01af1c402b9e176de56d0150c7/torch_points3d-1.3.0-py3-none-any.whl
pip install torch_points3d-1.3.0-py3-none-any.whl
```
Can ignore the error like: 
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
```
### 7. Install the following with certain versions again:
```
pip install hydra-core==1.0.7
conda install numpy==1.19.5
pip install open3d==0.12.0
pip install wandb --upgrade
```
### 8. Install pycuda
```
pip install pycuda
```
### 9. Test:
```
python -m unittest -v
```
If the error shows ```ImportError: 'test' module incorrectly imported from '/home/jinyixia/torch-points3d/test'. Expected '/home/jinyixia/torch-points3d'. Is this module globally installed?```, try:
```
python -m unittest discover -s test -p "test_*.py" -v
```
Shall see something like "Ran 163 tests in 213.969s OK"

### 10. Comment out the trainer.py line.355:
```
#skip_first=getattr(self._cfg.training.tensorboard.pytorch_profiler, "skip_first", 10)
```
As this api is for torch>=1.9.0, and we use torch=1.8.1

### 11. Next time activate the environment:
```
module load conda
conda activate torch-points3d
module load cuda/11.1
module load gcc/9.3.0
cd torch-points3d
```
### 12. Batch job
```
#!/bin/bash
#SBATCH --job-name=kpconv-train
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out

# Load modules and activate conda environment
module load conda
source activate torch-point3d
module load cuda/11.1
module load gcc/9.3

# Set threads (optional, to avoid OMP warnings)
export OMP_NUM_THREADS=8

# Navigate to your project directory
cd /home/jinyixia/torch-points3d

# Start training
python train.py task=segmentation models=segmentation/kpconv model_name=KPDeformableConvPaper data=segmentation/shapenet training.wandb.log=false
hydra.run.dir=/orange/rcstudents/jinyixia/torch-points3d_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```
```
sbatch run_kpconv.sh
watch -n 10 squeue -u jinyixia
tail -f logs/kpconv-train_
```

Or request GPU resources by Interactive Access
```
srun -p gpu --account=rcstudents --nodes=1 --gpus=a100:2 --time=3-00:00:00 --mem=300gb --pty -u bash -i
```
