import datajanitor


# Get the datasets and do basic data cleaning
shoppers = datajanitor.getDataset('shoppers')
df = shoppers.getDataFrame(doOHE=True)

# Partition and transform the dataset

# Get a learner and run the dataset
