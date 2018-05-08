# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import json
import os
print(os.listdir("input"))

def get_key(key):
    try:
        return int(key)
    except ValueError:
        return key

# Any results you write to the current directory are saved as output.
def multi_label(path):
    dic = {}
    ids = []
    binary_labels = []
    with open(path, 'r') as f:
        data = json.load(f)
    max_value = 0
    for i in data["annotations"]:
        for j in range(len(i["labelId"])):
            if max_value < int(i["labelId"][j]):
                max_value = int(i["labelId"][j])
    print(max_value)
    for i in data["annotations"]:
        tmp = [0]*max_value
        for j in i["labelId"]:
            tmp[int(j)-1] = 1
        dic[i["imageId"]] = tmp
    for key, labels in sorted(dic.items(), key = lambda t: get_key(t[0])):
        print(key)
        print(labels)
        ids.append(key)
        binary_labels.append(labels)
    ids = np.array(ids)
    binary_labels = np.array(binary_labels)
    print(ids.shape)
    print(binary_labels.shape)

    return ids, binary_labels

i, l = multi_label("input/validation.json")
print(i[27])
#max_train = multi_label("input/train.json", "train_label.json")
