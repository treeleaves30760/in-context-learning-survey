# In Context Learning On Open Source Model

## Install

```bash
conda create -n InContextLearning python==3.11.9 -y
conda activate InContextLearning

pip install -r requirements.txt
```

If you need to use cuda, install the [torch](https://pytorch.org/get-started/locally/) manually.

## Usage

```bash
python in-context-learning.py \
    --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --provider "huggingface" \
    --train_data "train.json" \
    --test_data "test.json" \
    --output_file "results.json"
```

## Reference