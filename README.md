# Event Traffic Forecasting 

The offical implement of paper "Event Traffic Forecasting with Sparse Multimodal Data", ACM MM 24.
 
## How to Install?
We test on Ubuntu 22 + python 3.9

```shell
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

## How to Run?
The model arch is in baselines/T3/arch/t3_arch.py

```shell
python experiments/train.py -c baselines/T3/SZCEC.py
```

We will release the total project (code, data, and docs) after paper acceptance, but there are some limitations on the data due to copyright issues.

