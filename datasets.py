from torchvision import datasets

class MNISTTrainDataset(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        if "download" not in kwargs:
            kwargs["download"] = True
        super().__init__(*args, train=True, **kwargs)

class MNISTTestDataset(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        if "download" not in kwargs:
            kwargs["download"] = True
        super().__init__(*args, train=False, **kwargs)
