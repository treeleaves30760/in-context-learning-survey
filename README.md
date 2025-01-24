# In Context Learning On Open Source Model

## Install

```bash
conda create -n InContextLearning python==3.11.9 -y
conda activate InContextLearning

pip install -r requirements.txt
```

If you need to use cuda, install the [torch](https://pytorch.org/get-started/locally/) manually.

Create an .env file for HuggingFace Token.

## Usage

```bash
python in_context_learning.py --model "meta-llama/Llama-3.2-11B-Vision-Instruct" --train_dir "srcs/fish" --model_type "llama3"
```

## Reference