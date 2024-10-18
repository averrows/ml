import torch.nn as nn
import torch
import torch.nn.functional as F
class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, timestep_channel, size):
        super().__init__()
        # we separate them because we don't want the loss from the param, get propagated
        self.out_channels = out_channels
        self.conv2d_multiplied_by_param = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, padding=1)
        self.conv2d_img = nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size = 3, padding = 1)
        
        self.dense_timestep = nn.Linear(timestep_channel, out_channels)

        self.layer_norm = nn.LayerNorm((out_channels, size, size))
    
    def forward(self, x, timestep_embedding):
        x_param = self.conv2d_multiplied_by_param(x)
        x_origin = self.conv2d_img(x)

        x_param = x_param * torch.reshape(self.dense_timestep(timestep_embedding), (-1, self.out_channels, 1, 1))
        return F.relu(self.layer_norm(x_origin + x_param))
    
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = []
        self.up = []

        self.down_x32 = UnetBlock(
            in_channels=3,
            out_channels=128,
            timestep_channel=192,
            size=32
        )
        self.down_x16 = UnetBlock(
            in_channels=128,
            out_channels=128,
            timestep_channel=192,
            size=16
        )
        self.down_x8 = UnetBlock(
            in_channels=128,
            out_channels=128,
            timestep_channel=192,
            size=8
        )
        self.down_x4 = UnetBlock(
            in_channels=128,
            out_channels=128,
            timestep_channel=192,
            size=4
        )

        self.mlp = nn.Sequential(
            # input_size = output_pixel + timestep_channel
            nn.Linear(4 * 4 * 128 + 192, 128),
            nn.LayerNorm([128]),
            nn.ReLU(),

            nn.Linear(128, 32 * 4 * 4),
            nn.LayerNorm([32 * 4 * 4]),
            nn.ReLU()
        )

        self.up_x4 = UnetBlock(
            in_channels= 32 + 128,
            out_channels=128,
            timestep_channel=192,
            size=4
        )

        self.up_x8 = UnetBlock(
            in_channels= 128 + 128,
            out_channels=128,
            timestep_channel=192,
            size=8
        )

        self.up_x16 = UnetBlock(
            in_channels= 128 + 128,
            out_channels=128,
            timestep_channel=192,
            size=16
        )

        self.up_x32 = UnetBlock(
            in_channels= 128 + 128,
            out_channels=128,
            timestep_channel=192,
            size=32
        )

        self.maxpool2d = nn.MaxPool2d(2)
        self.upsampling2d = nn.Upsample(scale_factor=2, mode="bilinear")

        self.down = [
            self.down_x32,
            self.down_x16,
            self.down_x8,
            self.down_x4
        ]

        self.up = [
            self.up_x4,
            self.up_x8,
            self.up_x16,
            self.up_x32
        ]

        self.down_output = []

        self.conv_out = nn.Conv2d(128, 3, 1)
        self.embed_timestep = nn.Sequential(
            nn.Linear(1, 192),
            nn.LayerNorm([192]),
            nn.ReLU(),
        )

    def forward(self, x, timestep):
        timestep_embedding = self.embed_timestep(timestep)
        x_down = self._forward_down(x, timestep_embedding)
        
        return self.conv_out(self._forward_up(
            x = (self.mlp(torch.cat([x_down.view(-1, 128 * 4 * 4), timestep_embedding], 1))).view(-1, 32, 4, 4),
            timestep_embedding=timestep_embedding
        ))
    
    def _forward_down(self, x, timestep_embedding):
        for idx, layer in enumerate(self.down):
            x = layer(x, timestep_embedding)
            self.down_output.append(x)
            if idx < len(self.down) - 1:
                x = self.maxpool2d(x)
        return x
    
    def _forward_up(self, x, timestep_embedding):
        for idx, layer in enumerate(self.up):
            x = layer(torch.cat([x, self.down_output[len(self.down_output) - idx - 1]], dim = 1), timestep_embedding)
            if idx < len(self.up) - 1:
                x = self.upsampling2d(x)
        return x

BATCH_SIZE = 32
def main():
    model = Unet()
    ts = torch.randint(0, 16, size=(BATCH_SIZE, 1), dtype=torch.float)
    x = torch.randn((BATCH_SIZE, 3, 32, 32))
    y = model(x, ts)
    print(y.shape)
main()
