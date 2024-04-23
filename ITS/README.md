## Architecture
![Architecture](./focalnet.svg)


## Environment

conda create -n vmam
conda activate vmam

pip install -r requirements.txt
cd kernels/selective_scan && pip install .

pip install causal-conv1d>=1.2.0

cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..