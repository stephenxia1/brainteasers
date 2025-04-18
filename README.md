# brainteasers
Please use python queryGPT.py --name [Name of your experiment] --dataset [Math/Logic] --rows [Default is 1] --samples [Default is 1]
Name is just the name of your logic, it will search for a corresponding .txt file in prompting to give as instructions
Dataset is Math or Logic
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