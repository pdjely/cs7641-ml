from . import shoppers, adult, news, cancer, nursery


def getDataset(dataSetName, randomState=1):
    """
    Factory function to generate a dataset by name
    :param randomState: randomState to pass to all partition functions
    :param dataSetName:
    :return:
    """
    datasets = {
        'shoppers': shoppers.Shoppers,
        'adult': adult.Adult,
        'news': news.News,
        'cancer': cancer.Cancer,
        'nursery': nursery.Nursery
    }

    return datasets[dataSetName]()


