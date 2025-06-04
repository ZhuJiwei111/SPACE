# SPACE

This is the official implementation of the ICML 2025 poster paper "SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model".

## Environment Setup

Key Package Version

```bash
torch==2.1.1
transformers==4.45.2
```
TBD

## Dataset

We utilize the dataset from [Basenji](https://console.cloud.google.com/storage/browser/basenji_barnyard), which is originally in TensorFlow data format and requires users to pay for download costs. We have converted the data to H5 format and made it freely available for download on Hugging Face.

TBD

# Pre-trained Model

We have uploaded our model configuration and weight files to Hugging Face Hub at: https://huggingface.co/yangyz1230/space

## Loading the Model

You can easily load the pre-trained model weights from Hugging Face Hub using the following method:

```python
from model.modeling_space import Space

model_name_or_path = "yangyz1230/space"
model = Space.from_pretrained(model_name_or_path)

## Pre-training

TBD

## Downstream Tasks

TBD

## Acknowledgments

Our implementation is based on [Enformer-Pytorch](https://github.com/lucidrains/enformer-pytorch) and [Mod-Squad](https://vis-www.cs.umass.edu/mod-squad). We thank their excellent work.

## Citation

If you find our work or code helpful, please cite:

```bibtex
TBD