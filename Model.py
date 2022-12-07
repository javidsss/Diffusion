import torch
import torch.nn as nn
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if not up: ##Downsampling
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        else: ##Upsampling
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        ## First conv:
        x = self.batch_norm1(self.relu(self.conv1(x)))
        ## Time embedding:
        time_emb = self.relu(self.time_mlp(t))
        ## adding two dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        ## add time channel to conv:
        x = x + time_emb
        ## next conv
        x = self.batch_norm2(self.relu(self.conv2(x)))
        ##down or upsample:
        return self.transform(x)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, time_emb_dim=32):
        super().__init__()
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)

        ## time embedding
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim),
                                      nn.Linear(time_emb_dim, time_emb_dim),
                                      nn.ReLU()
                                      )

        ## Initial projection
        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)
        ## Downsampling
        self.downs = nn.ModuleList([ConvBlock(down_channels[i], down_channels[i+1], time_emb_dim, up=False)
                                    for i in range(len(down_channels)-1)])
        ## Upsampling
        self.ups = nn.ModuleList([ConvBlock(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
                                  for i in range(len(down_channels)-1)])

        self.out = nn.Conv2d(up_channels[-1], in_channels, 1) ##Out_channel = in_channels so that the output has the
        # same channel as input

    def forward(self, x, timestep):
        ## Embedd time
        t = self.time_mlp(timestep)
        ## Initial conv
        x = self.conv0(x)
        ## Residual
        residuals = []
        for down in self.downs:
            x = down(x, t)
            ## saving the skip connections
            residuals.append(x)
        for up in self.ups:
            residual = residuals.pop()
            ## Add the skip connections to upsample
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)

        return self.out(x)


if __name__ == '__main__':
    batch_size = 5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(batch_size, 3, 64, 64)
    T = 300
    t = torch.randint(0, T, (batch_size,), device=device).long()
    model = Unet(in_channels=3, out_channels=1, time_emb_dim=32)

    y = model(x, t)
    assert y.shape == x.shape, "size doesn't match"

