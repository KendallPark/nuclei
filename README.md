## CS 760 Project Repo

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

To access the Jupyter notebooks, run:

``` sh
pipenv run jupyter lab
```
