class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        print(mean)
        print(std)
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean