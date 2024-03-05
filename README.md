# DLpoints

This is created to test different deep learning models in seperating the leaf and wood points from high density TLS (Terrestrial Laser Scan) data.
I am using a laptop from Silva Lab with NVIDIA Quadro RTX 5000 GPU.

The basic strucure is from a pytorch library torch-points3d. The first important step will be installing the library.
I'm following the steps from, but instead of Linux system, I am using a Windows system and the cuda version is 11.5.

conda create -n torch-points python=3.8 vc=14.2
conda activate torch-points
pip install torch_cluster==1.6.0 torch_scatter==2.0.9 torch_sparse==0.6.13 -f https://pytorch-geometric.com/whl/torch-1.11.0%2Bcu115.html
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/torch-points3d/torch-points3d.git
cd torch-points3d
pip install -r requirements.txt
