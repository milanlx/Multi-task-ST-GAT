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


class TotalSTGraphDateset(Dataset):
    def __init__(self, x_bus, x_inrix, x_ele, x_weather, y):
        self.x_bus = x_bus
        self.x_inrix = x_inrix
        self.x_ele = x_ele
        self.x_weather = x_weather
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        _x_bus = self.x_bus[index]
        _x_inrix = self.x_inrix[index]
        _x_ele = self.x_ele[index]
        _x_weather = self.x_weather[index]
        _y = self.y[index]
        return _x_bus, _x_inrix, _x_ele, _x_weather, _y

