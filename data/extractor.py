import csv
import random
import os
import pandas as pd
import shutil
from sklearn.utils import shuffle

extraction_rate = 0.1

with open("stage1_train_classses.csv") as fr:
    cr = csv.DictReader(fr)
    res = [x for x in cr if os.path.exists('stage1_train/{0}'.format(x['filename'].replace('.png','')))]
assert(res != [])
pd_data_origin = pd.DataFrame.from_records(res)
pd_data = shuffle(pd_data_origin, random_state = 1)
pd_grouped = pd_data.groupby(["foreground", "background"])
gps = pd_grouped.groups

extracted = {}
for key in gps:
    ext_index = gps[key][:1 + int((len(gps[key]) - 1) * extraction_rate)]
    extracted[key] = pd_data_origin['filename'][ext_index]

if not os.path.exists('data_validation'):
    os.makedirs('data_validation')

count = 0
for key in extracted:
    for filename in extracted[key]:
        count += 1
        filename_ = filename.replace('.png', '')
        shutil.move('stage1_train/{0}'.format(filename_), 'data_validation/{0}'.format(filename_))

print("moved {0} files to data_validation folder.".format(count))