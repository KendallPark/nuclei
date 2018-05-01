cd SageMaker

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | sudo bash

sudo yum install git-lfs

git lfs clone https://github.com/KendallPark/nuclei.git

pip install pipenv

pipenv install

pipenv install tensorflow-gpu

git clone https://github.com/matterport/Mask_RCNN.git

cd Mask_RCNN

pipenv run python setup.py install

cd ..

cd data

unzip stage1_sample_submission.csv.zip
unzip stage1_solution.csv.zip
unzip stage1_train_labels.csv.zip
unzip stage1_test.zip -d stage1_test/
unzip stage1_train.zip -d stage1_train/

cd ..

# pipenv run python nucleus.py train --dataset=../data/ --subset=train --weights=none
# pipenv run python nucleus.py train --dataset=../data/ --subset=train --weights=last
