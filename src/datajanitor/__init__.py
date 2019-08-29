from . import shoppers, adult


def getDataset(dataSetName):
    """
    Factory function to generate a dataset by name
    :param dataSetName:
    :return:
    """
    datasets = {
        'shoppers': shoppers.Shoppers
    }

    return datasets[dataSetName]()


