import pandas as pd 
from tqdm import tqdm

df = pd.read_pickle("/Users/lotburgstra/Desktop/TCC_ML/Question_4/data.pckl.gz", compression = 'gzip')

df.to_csv("training_data.csv", index = False)

