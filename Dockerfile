FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y curl git build-essential && rm -rf /var/lib/apt/lists/*
# Install system libraries required by Open3D (libGL)
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*
    
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda --version

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create --prefix .conda_env python=3.10

ENV PYTHONUNBUFFERED=1
ENV PIP_PROGRESS_BAR=raw
# RUN conda activate ./.conda_env/
# SHELL ["conda","run","-p",".conda_env","/bin/bash","-c"]
SHELL ["conda","run","--no-capture-output","-p",".conda_env","/bin/bash","-c"]

RUN pip --version

# https://github.com/Pointcept/PointTransformerV3?tab=readme-ov-file#installation

RUN pip install ninja h5py pyyaml
RUN pip install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm
RUN pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
RUN pip install torch-geometric
RUN pip install spconv-cu126
RUN pip install open3d trimesh
RUN pip install flash-attn --no-build-isolation
RUN pip install wandb peft

RUN git clone https://github.com/Pointcept/Pointcept.git
ENV TORCH_CUDA_ARCH_LIST="8.6"
RUN cd Pointcept/libs/pointops && python setup.py install

# RUN git clone https://github.com/Pointcept/PointTransformerV3.git

COPY train/s3dis/semseg-pt-v3m1-0-base--local.py /Pointcept/configs/s3dis/semseg-pt-v3m1-0-base--local.py
COPY train/zaha/semseg-pt-v3m1-0-zaha.py /Pointcept/configs/zaha/semseg-pt-v3m1-0-zaha.py
COPY train/zaha/semseg-pt-v1-zaha.py /Pointcept/configs/zaha/semseg-pt-v1-zaha.py
# COPY train/zaha/datasets/__init__.py /Pointcept/pointcept/datasets/__init__.py
# COPY train/zaha/datasets/zaha.py /Pointcept/pointcept/datasets/zaha.py
COPY train/zaha/addons/__init__.py /Pointcept/pointcept/datasets/__init__.py
COPY train/zaha/addons/zaha_dataset.py /Pointcept/pointcept/datasets/zaha_dataset.py
# COPY train/zaha/addons/height_transform.py /Pointcept/pointcept/datasets/height_transform.py
# COPY train/zaha/addons/remap_labels_transform.py /Pointcept/pointcept/datasets/remap_labels_transform.py
COPY train/zaha/addons/zaha_transforms.py /Pointcept/pointcept/datasets/zaha_transforms.py

COPY train/evaluator.py /Pointcept/pointcept/engines/hooks/evaluator.py
COPY train/test.py /Pointcept/pointcept/engines/test.py
COPY train/train.py /Pointcept/pointcept/engines/train.py

ENV WANDB_MODE=disabled

RUN pip install psutil

# CMD ["conda", "run", "-p", ".conda_env", "python", "/app/main.py"]
CMD ["conda", "run", "--no-capture-output", "-p", ".conda_env", "bash", "/train/zaha/run.sh"]
