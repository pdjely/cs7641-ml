from . import shoppers, adult, news, cancer, nursery, spam, musk


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
        'nursery': nursery.Nursery,
        'spam': spam.Spam,
        'musk': musk.Musk
    }

    return datasets[dataSetName]()


