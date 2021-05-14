from aug_utils import *
import json

data_dir='/opt/ml/input/data/train_dataset/train_dials.json'
data = json.load(open(data_dir))

#현재 변형된 데이터들만 존재, 사용할 때는 합쳐서 사용해야함
new_data=augmentation(data)
with open("/opt/ml/input/data/train_dataset/new_dataset1.json", "w", encoding='UTF8') as json_file:

    json.dump(new_data, json_file,indent=2 ,ensure_ascii=False)