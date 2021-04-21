from torch.utils.data import Dataset, DataLoader


class STGraphDateset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y



