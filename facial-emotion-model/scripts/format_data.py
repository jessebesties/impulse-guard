import pandas as pd
from PIL import Image
import os

df = pd.read_csv('raw_data.csv')
for index, row in df.iterrows():
    if row['angle'] == 'FL' or row['angle'] == 'FR':
        continue
    
    image = Image.open(row['path'])
    cropped_image = image.crop((0, 162, 562, 662))
    if row['identity'] == 7 or row['identity'] == 8:
        cropped_image.save(f'data/test/{row["expression"]}/{row["path"].split("/")[-1]}')
        # os.remove(f'data/test/{row["expression"]}/{row["path"].split("/")[-1]}')
    else:
        cropped_image.save(f'data/train/{row["expression"]}/{row["path"].split("/")[-1]}')
        # os.remove(f'data/train/{row["expression"]}/{row["path"].split("/")[-1]}')