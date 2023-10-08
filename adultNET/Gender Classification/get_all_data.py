import pandas as pd
import os
import numpy as np
import cv2

classes = ["female", "male"]

img = []
img_name = []
class_names = []
speciemns = []
file_path = []
for cls in classes:
    if(cls != ".DS_Store"):
        for spec in os.listdir(f"data/{cls}/"):
            if(spec != ".DS_Store"):
                files = os.listdir(f"data/{cls}/{spec}")
                if(".DS_Store" in files): files.remove(".DS_Store")
                print(f"{cls} - {spec} - {len(files)}")
                for f in files: 
                    file_path.append(f"data/{cls}/{spec}/{f}")
                    img.append(cv2.resize(cv2.imread(f"data/{cls}/{spec}/{f}"), (224, 224)))
                    img_name.append(f)
                    class_names.append(cls)
                    if(cls == "male"): speciemns.append(int(spec.split("_")[0]))
                    else: speciemns.append(int(spec))

df = pd.DataFrame(list(zip(img_name, file_path, class_names, speciemns)), \
    columns = ["file_name", "file_path","class_name", "specimen"])

df.to_csv(os.path.join('data/classification_data_stephensiUnfed.csv'))

train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()


class_files = df.class_name.unique()

for i in range(len(class_files)):

    unique_df = df[df.class_name.isin([class_files[i]])]

    unique_files = unique_df.specimen.unique()
    train_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.60), replace=False))
    train_df = pd.concat([train_df, unique_df[unique_df.specimen.isin(train_files)]])
    rest = unique_df[~unique_df.specimen.isin(train_files)]

    unique_files = rest.specimen.unique()
    val_files = set(np.random.choice(unique_files, int(len(unique_files) * 0.50), replace=False))
    val_df = pd.concat([val_df, rest[rest.specimen.isin(val_files)]])

    test_df = pd.concat([test_df, rest[~rest.specimen.isin(val_files)]])

train_df.to_csv(f'data/train.csv', header=True, index=None)
val_df.to_csv(f'data/val.csv', header=True, index=None)
test_df.to_csv(f'data/test.csv', header=True, index=None)

print("Finished")
