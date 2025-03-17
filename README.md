<img src="https://github.com/mlop-ai/mlop/raw/refs/heads/main/docs/static/img/logo.svg?sanitize=true" alt="logo" height="80">

<div class="markdown-google-sans">
  <h1><strong>MLOP</strong></h1>
</div>

[![stars](https://img.shields.io/github/stars/mlop-ai/mlop)](https://github.com/mlop-ai/mlop/stargazers)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/intro.ipynb)
[![pypi](https://img.shields.io/pypi/v/mlop)](https://pypi.org/project/mlop/)
[![build](https://img.shields.io/github/actions/workflow/status/mlop-ai/mlop/mlop.yml)](https://github.com/mlop-ai/mlop/actions/workflows/mlop.yml)
[![license](https://img.shields.io/github/license/mlop-ai/mlop)](https://github.com/mlop-ai/mlop/blob/main/LICENSE)

**MLOP** is a Machine Learning Operations (MLOps) framework. It provides superior experimental tracking capabilities as well as lifecycle management for training ML models. To get started, [try out our introductory notebook](https://colab.research.google.com/github/mlop-ai/mlop/blob/main/examples/intro.ipynb) or [get an account with us today](https://demo.mlop.ai/auth/sign-up)!

## Demo

The following clip demonstrates benefits of using **MLOP** (bottom left) over conventional alternative experimental tracking tooling (bottom right). **MLOP**'s lightweight and asynchronous design allows it to perform the same tasks significantly faster. Additionally, the performance does not critically suffer in a high throughput scenario either - this is, however, a common problem with other tools in this category. Supporting high data ingestion rates should be a priority to achieve efficient MLOps. It is the future trend as compute power increases.

<video loop src='https://github.com/user-attachments/assets/efd9720e-6128-4278-85ec-ee6139a851af' alt="demo" width="1200" style="display: block; margin: auto;"></video>

## Getting Started

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
