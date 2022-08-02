# Multi-Objective Reinforcement Learning in Autonomous UAV Navigation

The implementation of the proposed framework in [this paper](https://www.example.com).

The conda environment file is [environment.yaml](./environment.yaml). Here we use PyTorch with CUDA 10.2, 
which might not work with the latest GPUs. Update to newer version of PyTorch using instructions from [PyTorch Install Guide](https://pytorch.org/get-started/locally/)

crete new environment by 

```shell
 conda env create -n <ENVNAME> --file environment.yaml
```

Before running, you might want to edit the config in the [main.py](./main.py)

Run the experiment by:

```shell
python main.py
```

or run in background with:

```shell
nohup python ./main.py > out.txt 2>&1 &
```
