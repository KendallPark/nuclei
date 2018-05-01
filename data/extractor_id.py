import csv
import random
import os
import pandas as pd
import shutil
from sklearn.utils import shuffle

def extraction(extraction_rate=0.1, stage_1_train_classes_csv="stage1_train_classes.csv", stage1_train_folder="stage1_train"):
    if not os.path.exists(stage_1_train_classes_csv):
        raise IOError("Make sure you have stage1_train_classes.csv")
    if not os.path.exists(stage1_train_folder):
        raise IOError("Make sure you have folder: stage1_train")
    with open(stage_1_train_classes_csv) as fr:
        cr = csv.DictReader(fr)
        res = [x for x in cr if os.path.exists('{0}/{1}'.format(stage1_train_folder, x['filename'].replace('.png','')))]
    assert(res != [])
    pd_data_origin = pd.DataFrame.from_records(res)
    pd_data = shuffle(pd_data_origin, random_state = 1)
    pd_grouped = pd_data.groupby(["foreground", "background"])
    gps = pd_grouped.groups
    extracted = {}
    result = []
    for key in gps:
        ext_index = gps[key][:1 + int((len(gps[key]) - 1) * extraction_rate)]
        extracted[key] = pd_data_origin['filename'][ext_index]
        result.extend([x.replace('.png', '') for x in extracted[key]])
    return result

if __name__ == "__main__":
    print(extraction(0.1, "stage1_train_classes.csv", "stage1_train"))
