## Architecture
![Architecture](./focalnet2.svg)


## Environment
~~~
conda create -n vmam
conda activate vmam
~~~

~~~
pip install causal-conv1d>=1.2.0
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
~~~

~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~