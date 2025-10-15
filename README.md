<h1 align="center">
<code>Brainteaser</code>
</h1>

<p align="center">
<strong>Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models</strong>
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-firebrick)](https://arxiv.org/abs/2505.10844)
[![OpenReview](https://img.shields.io/badge/OpenReview-eeeeee)](https://openreview.net/forum?id=3oQDkmW72a)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS_2025-purple)](https://openreview.net/pdf?id=3oQDkmW72a)
[![HuggingFace Dataset](https://img.shields.io/badge/HuggingFace_Dataset-orange)](https://huggingface.co/datasets/ChenLiu1996/Brainteaser)
[![Github Stars](https://img.shields.io/github/stars/stephenxia1/brainteasers.svg?style=social&label=Stars)](https://github.com/stephenxia1/brainteasers/)

</div>

<br>

This is the official implementation of [**Brainteaser**](https://arxiv.org/abs/2505.10844), NeurIPS 2025.

## Citation
```bibtex
@article{han2025creativity,
  title={Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models},
  author={Han, Simeng and Xia, Stephen and Zhang, Grant and Dai, Howard and Liu, Chen and Chen, Lichang and Nguyen, Hoang Huy and Mei, Hongyuan and Mao, Jiayuan and McCoy, R. Thomas},
  journal={Advances in neural information processing systems},
  year={2025}
}
```

## Usage
(Update the code below please)

```python3
python queryGPT.py \
    --name [Name of your experiment] \
    --dataset [Math/Logic] \
    --prompt ["promptInstructions.txt"] \
    --rows [Default is 1] \
    --samples [Default is 1]

# Name: the name of your experiment
# Dataset: Math or Logic
# Prompt: your instructions file
# Rows: the # of problems from the data to sample on
# Samples: the # of times to query each question
# Results will be saved in ./results
```

## Dependencies
We developed the codebase in a miniconda environment.
How we created the conda environment:
```
# Optional: Update to libmamba solver.
conda update -n base conda
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

conda create --name bt pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c anaconda -c conda-forge -y
conda activate bt
conda install scikit-learn scikit-image pandas matplotlib seaborn tqdm -c pytorch -c anaconda -c conda-forge -y

python -m pip install beautifulsoup4 requests
python -m pip install nltk

python -m pip install openai
python -m pip install google-genai
python -m pip install anthropic
```
