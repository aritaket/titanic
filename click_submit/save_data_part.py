import pandas as pd

TR_NUM = 10000
train_reader = pd.read_csv("train.csv", header=0, chunksize=TR_NUM)
train_df = train_reader.get_chunk(TR_NUM)
train_df.to_csv('train_part.csv', index=False)

TR_NUM = 10000
train_reader = pd.read_csv("test.csv", header=0, chunksize=TR_NUM)
train_df = train_reader.get_chunk(TR_NUM)
train_df.to_csv('test_part.csv', index=False)
