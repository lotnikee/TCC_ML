import pandas as pd 
from tqdm import tqdm

file_path = "/Users/lotburgstra/Downloads/TrainingData/data.pckl.gz"

df = pd.read_pickle("/Users/lotburgstra/Desktop/TCC_ML/Question_4/data.pckl.gz", compression = 'gzip')

df.to_csv("training_data.csv", index = False)

chunks = []

for chunk in tqdm(pd.read_csv("training_data.csv", chunksize = 10000), desc = "Loading CSV"):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index = True)

print(df.head())

