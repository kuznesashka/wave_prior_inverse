# Quickstart
1. Download and install
[miniconda](https://docs.conda.io/en/latest/miniconda.html) python
distribution for your OS if not already installed. Anaconda is also fine.
2. Create virtual environment and install dependencies:
```
conda create -n wave python=3.8
conda activate wave
conda install -c conda-forge mne-base scipy matplotlib scikit-learn numpy pandas
```
3. Download the data archive from [here](https://disk.yandex.ru/d/zbNj4Qj4Q3y3ew); extract the
files into the projects' root
4. Launch the analysis from the projects' root folder with
```
python main.py
```
You should see the resulting plot with the ROC-surve in the end.
