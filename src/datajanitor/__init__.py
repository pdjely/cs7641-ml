from . import shoppers, adult

"""
Factory function to generate a dataset
"""
def getDataset(dataSetName):
    datasets = {
        'shoppers': shoppers.Shoppers
    }

    return datasets[dataSetName]()
