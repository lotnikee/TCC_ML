import pandas as pd 

file_path = "/Users/lotburgstra/Downloads/TrainingData/data.pckl.gz"

df = pd.read_pickle(file_path, compression = 'gzip')