# 🧬 SPACE 

This is the official implementation of the ICML 2025 poster paper "SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model".

## Environment Setup

You can create a conda environment using the provided `environment.yml` file to ensure all dependencies are properly installed:

```bash
conda env create -f environment.yml
conda activate space
```

## Dataset
We utilize the dataset from [Basenji](https://console.cloud.google.com/storage/browser/basenji_barnyard), which is originally in TensorFlow data format and requires users to pay for download costs. We have converted the data to H5 format and made it freely available for download on 🤗 Hugging Face: https://huggingface.co/datasets/yangyz1230/space.

Note: The original data we provide is in compressed H5 format, which is not conducive to parallel data loading with dataloaders during training. We have provided preprocessing code in our repository to convert the compressed H5 format to byte streams: https://github.com/ZhuJiwei111/SPACE/blob/6fc1aee012229f7c3daab033584caa3e9d07485a/dataloaders/h5dataset.py#L120-L131. You can also convert the H5 data yourself to formats suitable for large-scale training, such as WebDataset.
<!--,
**Update**:
Due to potential difficulties with H5 format data in supporting parallel data loading, we have prepared a new format where each sample's genomic profile is stored as individual NumPy (.npy) files. We will upload these soon （ https://huggingface.co/datasets/yangyz1230/space_npy ） and provide the corresponding dataset implementation. (In fact, converting from H5 to .npy format is quite straightforward - if your training is bottlenecked by data loading, you may also try converting the data yourself first.)
-->

## Pre-trained Model
We have uploaded our model config and weights to 🤗 Hugging Face Hub at: https://huggingface.co/yangyz1230/space.
You can easily load the pre-trained model using the following code:
```python
from model.modeling_space import Space
model_name_or_path = "yangyz1230/space"
model = Space.from_pretrained(model_name_or_path)
```
## Try the Genomic Profile Prediction
We provide code for genomic profile prediction in both `test_Enformer.ipynb` and `test_SPACE.ipynb`.

## Quick Start for Enformer/Borzoi Training

If you are specifically interested in reproducing Enformer and similar models like Borzoi with minimal setup, please refer to our simplified repository: [Enformer_Borzoi_Training_PyTorch](https://github.com/yangzhao1230/Enformer_Borzoi_Training_Pytorch). This repository provides streamlined training scripts with minimal modifications to the original Hugging Face Trainer, making it easier to get started with genomics model training.

## SPACE Pre-training

You can train a SPACE model from scratch:
```
bash train.sh
```

## Downstream Tasks

We provide code for reproducing downstream tasks in the folder `experiments/`. This requires ensuring your downstream task data is stored in the `datasets/` directory.

For example, to reproduce NT benchmark's results:

1. Place the NT dataset in `datasets/NT/`
2. Run the following commands:
```
bash experiments/NT/expr_NT.sh
```

## Acknowledgments

Our implementation is based on [Enformer-Pytorch](https://github.com/lucidrains/enformer-pytorch) and [Mod-Squad](https://github.com/UMass-Embodied-AGI/Mod-Squad). We thank their excellent work.

## Citation

If you find our work, code, or released data helpful, please cite:

```bibtex

@inproceedings{yang2025space,
      title={{SPACE}: Your Genomic Profile Predictor is a Powerful {DNA} Foundation Model},
      author={Zhao Yang and Jiwei Zhu and Bing Su},
      booktitle={Forty-second International Conference on Machine Learning},
      year={2025},
      url={https://openreview.net/forum?id=o4L9y4Jetm}
}
