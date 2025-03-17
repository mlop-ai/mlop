---
sidebar_position: 1
---

# Quickstart

Start logging your experiments with **MLOP** in 4 simple steps:

1. Get an account at [demo.mlop.ai](https://demo.mlop.ai/auth/sign-up)
2. Install our Python SDK. Within a Python environment, open a Terminal window and paste in the following,
```bash
pip install mlop[dev]
```
3. Log in to your [mlop.ai](https://demo.mlop.ai/o) account from within the Python client,
```python
import mlop
mlop.login()
```
4. Start logging your experiments by integrating **MLOP** to the scripts, as an example,
```python
import mlop

config = {'lr': 0.001, 'epochs': 1000}
run = mlop.init(project="title", config=config)

# insert custom model training code
for i in range(config['epochs']):
    run.log({"val/loss": 0})

run.finish()
```
And... profit! The script will redirect you to the webpage where you can view and interact with the run. The web dashboard allows you to easily compare time series data and can provide actionable insights for your training.

These steps are described in further detail in our [introductory tutorial](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/intro.ipynb).  
You may also learn more about **MLOP** by checking out our [documentation](https://mlop-ai.github.io/docs/).
