from . import shoppers, adult


def getDataset(dataSetName, randomState=1):
    """
    Factory function to generate a dataset by name
    :param randomState: randomState to pass to all partition functions
    :param dataSetName:
    :return:
    """
    datasets = {
        'shoppers': shoppers.Shoppers,
        'adult': adult.Adult
    }

    return datasets[dataSetName](randomState=randomState)


