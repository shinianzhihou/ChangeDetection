class MultiDropout2d(nn.Module):
    """Perform `Dropout2d` in several features."""
    def __init__(self, p=0.5, dim=2, inplace=True):
        super().__init__()
        self.drop = nn.Dropout2d(p=p, inplace=inplace)
        self.dim = dim

    def forward(self, x_list):
        x = torch.cat(x_list, dim=self.dim)
        x = self.drop(x)
        split_size = int(x.shape[self.dim]/len(x_list))
        return torch.split(x, split_size, dim=self.dim)