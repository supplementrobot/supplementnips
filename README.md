# PRFusion

### Requirements

---

- Platform

  ```
  Ubuntu 20.04
  nvcc 11.8/11.6
  python 3.8
  ```

- PyTorch

  ```
  pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
  ```

  MinkowskiEngine 0.5.4

  ```
  conda install openblas-devel -c anaconda
  pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
  ```

  PointNet2

  ```
  cd pointnet2
  python setup.py install
  ```

  Others1

  ```
  numpy
  open3d
  opencv-python
  pandas
  Pillow
  pytorch_metric_learning==1.1
  scikit-learn
  scipy
  torch_geometric==1.7.2
  torch_scatter==2.0.9
  torch_cluster
  torch_sparse
  timm
  torchdiffeq
  tqdm
  matplotlib
  ```

  SpConv

  ```
  pip install spconv-cu118	
  ```

  SpTr

  ```
  git clone https://github.com/dvlab-research/SparseTransformer.git
  cd SparseTransformer
  python setup.py install
  ```

### Datasets and Code

***

- BenchmarkBoreasv3.zip
  https://drive.google.com/file/d/1zWF8uSmnDgzYczuuoK-w-zF_AnVV7o95/view?usp=share_link
- Logs
  https://drive.google.com/drive/folders/1dUbsEKuzkoXWISPCPsSWhNgbkI5QdM7H?usp=share_link  

