import torch.nn as nn
import torch
import torch.nn.functional as F
class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, timestep_channel, size):
        # we separate them because we don't want the loss from the param, get propagated
        self.out_channels = out_channels
        self.conv2d_multiplied_by_param = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, padding=1)
        self.conv2d_img = nn.Conv2d(in_channels=out_channels, out_channels= out_channels, kernel_size = 3, padding = 1)
        
        self.dense_timestep = nn.Linear(timestep_channel, out_channels)

        self.layer_norm = nn.LayerNorm((out_channels, size, size))
    
    def forward(self, x, timestep_embedding):
        x_param = self.conv2d_multiplied_by_param(x)
        x_origin = self.conv2d_img(x)

        x_param = x_param * torch.reshape(self.dense_timestep(timestep_embedding), (-1, self.out_channels, 1, 1))
        return F.relu(self.layer_norm(x_origin + x_param))
