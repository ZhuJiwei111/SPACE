# 🧬 SPACE 

This is the official implementation of the ICML 2025 poster paper "SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model".

# Environment Setup

You can create a conda environment using the provided `environment.yml` file to ensure all dependencies are properly installed:

```bash
conda env create -f environment.yml
conda activate space
```

## Dataset
We utilize the dataset from [Basenji](https://console.cloud.google.com/storage/browser/basenji_barnyard), which is originally in TensorFlow data format and requires users to pay for download costs. We have converted the data to H5 format and made it freely available for download on 🤗 Hugging Face: https://huggingface.co/datasets/yangyz1230/space/tree/main.

TBD: Since the dataset is very large, the upload work is still in process.

# Pre-trained Model
We have uploaded our model config and weights to 🤗 Hugging Face Hub at: https://huggingface.co/yangyz1230/space.
You can easily load the pre-trained model using the following code:
```python
from model.modeling_space import Space
model_name_or_path = "yangyz1230/space"
model = Space.from_pretrained(model_name_or_path)
```

## Pre-training

TBD

## Downstream Tasks

TBD

## Acknowledgments

Our implementation is based on [Enformer-Pytorch](https://github.com/lucidrains/enformer-pytorch) and [Mod-Squad](https://vis-www.cs.umass.edu/mod-squad). We thank their excellent work.

## Citation

If you find our work, code or released data helpful, please cite:

```bibtex
@misc{yang2025spacegenomicprofilepredictor,
      title={SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model}, 
      author={Zhao Yang and Jiwei Zhu and Bing Su},
      year={2025},
      eprint={2506.01833},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.01833}, 
}
