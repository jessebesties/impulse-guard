import os
import pandas as pd

def explore_directory(path):
    paths = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            paths.append(os.path.join(root, file_name))
    return paths

paths = explore_directory("raw-data/KDEF")[1:]

processed_paths = []
for path in paths:
    img_file = path.split("/")[-1][:-4]
    session = img_file[0]
    gender = img_file[1]
    identity = img_file[2:4]
    expression = img_file[4:6]
    angle = img_file[6:8]
    processed_paths.append([path, session, gender, identity, expression, angle])
    
df = pd.DataFrame(processed_paths, columns=["path", "session", "gender", "identity", "expression", "angle"])
print(df)
df.to_csv('raw_data.csv', index=False)
# print(len(cut_paths))