## CS 760 Project Repo

![](https://i.imgur.com/SqbGhnJ.png)

### Check it out!

[DEMO!!!](http://104.197.241.47:8888/notebooks/demo.ipynb?token=a107b97ea8b9acfff38617881ca81b53fe9314a71a93193b)

[Final report](https://docs.google.com/document/d/1XiXpHKwEOPc9PkKmj1xGaOxLR5skX2kg2bvludoqqXg/edit?usp=sharing).

[Data](https://docs.google.com/spreadsheets/d/1r2RDiRO-6SYY2ZDEveDztFRMF0re3nCt5PwRgchQM6E/edit?usp=sharing).

[Original Kaggle Competition](https://www.kaggle.com/c/data-science-bowl-2018).

### Authors

[Kendall Park](https://github.com/KendallPark), [Will Strauss](https://github.com/willstrauss), [Xaihe Lui](https://github.com/shynehua), [Frank Zou](https://github.com/szou28), [Jon Ide](https://github.com/pastpeak), and [Zetong Qi](https://github.com/zetongqi).

### Setup

Install [pipenv](https://github.com/pypa/pipenv). This will keep our development environments the same. If you have python installed via anaconda, check out [these instructions](https://github.com/pypa/pipenv/blob/master/docs/advanced.rst#-pipenv-and-conda).

Install [Git LFS](https://git-lfs.github.com/).

Clone the repo.

``` sh
git lfs clone git@github.com:KendallPark/nuclei.git
cd nuclei
```

Install dependencies.
``` sh
pipenv install
```

Clone the Mask RCNN repo and install.
``` sh
git clone git@github.com:matterport/Mask_RCNN.git
cd Mask_RCNN
pipenv run python setup.py install
cd ..
```

Unzip all the data files.
``` sh
cd data
unzip stage1_sample_submission.csv.zip
unzip stage1_solution.csv.zip
unzip stage1_train_labels.csv.zip
unzip stage1_test.zip -d stage1_test/
unzip stage1_train.zip -d stage1_train/
cd ..
```

To access the Jupyter notebooks, run:

``` sh
pipenv run jupyter lab
```
