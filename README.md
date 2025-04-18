# brainteasers
Please use python queryGPT.py --name [Name of your experiment] --dataset [Math/Logic] --prompt ["promptInstructions.txt"] --rows [Default is 1] --samples [Default is 1]

Name is just the name of your experiment
Dataset is Math or Logic
Prompt is your instructions file
Rows is the # of problems from the data to sample on
Samples is the # of times to query each question
Results will be saved in ./results


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
```