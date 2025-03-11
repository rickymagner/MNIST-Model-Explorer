from torchvision import datasets, transforms
from torchvision import transforms as T
from torch.utils.data import random_split, DataLoader

class MNISTData:
    """
    Class to handle MNIST data loading and processing.
    """
    def __init__(self, batch_size: int = 64, normalize: bool = True, val_split: float = 0.1, data_dir: str = "data"):
        self.batch_size = batch_size
        self.normalize = normalize
        self.val_split = val_split

        self.data_dir = data_dir

        self.train_dataset, self.val_dataset, self.test_dataset = self.get_mnist_data()
        self.train_loader, self.val_loader, self.test_loader = self.make_dataloaders(self.train_dataset, self.val_dataset, self.test_dataset)

    def get_mnist_data(self):
        """
        Get the MNIST dataset, downloading if needed.
        """
        transformations = [transforms.ToTensor()]
        if self.normalize:
            # Use known MNIST normalization parameters for mean/std
            transformations.append(transforms.Normalize((0.1307,), (0.3081,)))

        transform = T.Compose(transformations)

        train_dataset_full = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)

        train_dataset, val_dataset = random_split(train_dataset_full, [1 - self.val_split, self.val_split])

        return train_dataset, val_dataset, test_dataset

    def make_dataloaders(self, train_dataset, val_dataset, test_dataset) -> (DataLoader, DataLoader, DataLoader):
        """
        Create dataloaders for the training, validation, and test sets.
        """
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader