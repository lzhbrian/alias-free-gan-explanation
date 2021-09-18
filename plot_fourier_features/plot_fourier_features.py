# code is mostly modified from https://github.com/rosinality/alias-free-gan-pytorch/blob/d1a4c52ea0be9a6a853fe10e486402b276aef94b/model.py
# https://github.com/rosinality/alias-free-gan-pytorch/pull/1
# https://github.com/rosinality/alias-free-gan-pytorch/issues/24

import math

import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F


class FourierFeature(nn.Module):
    def __init__(self, size=16, margin=10, dim=512, cutoff=2, eps=1e-8):
        """
        size:   sampling rate (or feature map size)
        margin: expanded feature map margin size
        dim:    # channels
        cutoff: cutoff fc
        """
        super().__init__()

        normalized_margin = margin / size * 1
        # -0.5-m ~ 0.5+m, uniform interplate 'size' (except last one)
        # note the margin here, target canvas was -0.5~0.5, extended canvas should be larger
        coords = torch.linspace(- 0.5 - normalized_margin, 
                                  0.5 + normalized_margin, 
                                size + 2 * margin + 1)[:-1]

        # 0~fc, uniform interpolate 'dim//4'
        freqs  = torch.linspace(0, cutoff, dim // 4)

        self.register_buffer("coords", coords)
        self.register_buffer("freqs", freqs)

        self.eps = eps

    def forward(self, batch_size, affine=None):
        """
        affine: [B, 4], r_c, r_s, t_x, t_y
        """

        coord_map = torch.ger(self.freqs, self.coords) # outer product, [dim//4, size]

        size = self.coords.shape[0] # size
        coord_h = coord_map.view(self.freqs.shape[0], 1, size) # [dim//4, 1, size]
        coord_w = coord_h.transpose(1, 2)                      # [dim//4, size, 1]

        if affine is not None:
            norm = torch.norm(affine[:, :2], dim=-1, keepdim=True)
            affine = affine / (norm + self.eps)
            r_c, r_s, t_x, t_y = affine.view(
                affine.shape[0], 1, 1, 1, affine.shape[-1] # [B, 1, 1, 1, 4]
            ).unbind(-1) # each one is [B, 1, 1, 1]

            coord_h_orig = coord_h.unsqueeze(0) # [1, dim//4, 1, size]
            coord_w_orig = coord_w.unsqueeze(0) # [1, dim//4, size, 1]

            # [x] = [cos, -sin][x] + [tx] = [x*cos - y*sin + tx]
            # [y]   [sin,  cos][y]   [ty] = [x*sin + y*cos + ty]
            # lots of broadcasting here
            coord_w =  coord_w_orig * r_c - coord_h_orig * r_s + t_x  # [B, dim//4, size, size]
            coord_h =  coord_w_orig * r_s + coord_h_orig * r_c + t_y  # [B, dim//4, size, size]

            coord_w = torch.cat((torch.sin(2 * math.pi * coord_w), torch.cos(2 * math.pi * coord_w)), 1) # [B, 2*(dim//4), size, size]
            coord_h = torch.cat((torch.sin(2 * math.pi * coord_h), torch.cos(2 * math.pi * coord_h)), 1) # [B, 2*(dim//4), size, size]
            coords = torch.cat((coord_w, coord_h), 1) # [B, 4*(dim//4), size, size]
            return coords # [B, 4*(dim//4), size, size]

        else:
            # every channel (or feature map) represent some freq on h/w 's sin/cos, so we use dim//4
            # 1D -> expand to 2D
            coord_w = torch.cat((torch.sin(2 * math.pi * coord_w), torch.cos(2 * math.pi * coord_w)), 0) # [2*(dim//4), size, 1]
            coord_h = torch.cat((torch.sin(2 * math.pi * coord_h), torch.cos(2 * math.pi * coord_h)), 0) # [2*(dim//4), 1, size]
            coord_w = coord_w.expand(-1, -1, size)      # [2*(dim//4), size, size]
            coord_h = coord_h.expand(-1, size, -1)      # [2*(dim//4), size, size]
            coords = torch.cat((coord_w, coord_h), 0)   # [4*(dim//4), size, size]
            return coords.unsqueeze(0).expand(batch_size, -1, -1, -1) # [B, 4*(dim//4), size, size]

num_channel = 32
fc = 2
size = 16 # 16x16
margin = 10
normalized_margin = margin / size * 1

a = FourierFeature(size, margin, num_channel, fc)
# a.forward(1).shape
out = a.forward(1, torch.Tensor([[1,0,0,0]]))
# print(out.shape) # [1, num_channel, 26, 26] NCHW
# print(out)



def plot3d(ax, Z):
    # plot a 3D surface like in the example mplot3d/surface3d_demo
    X = np.linspace(- 0.5 - normalized_margin, 
                      0.5 + normalized_margin, 
                    size + 2 * margin + 1)[:-1]
    Y = np.linspace(- 0.5 - normalized_margin, 
                      0.5 + normalized_margin, 
                    size + 2 * margin + 1)[:-1]
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z)#, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)
    # fig.colorbar(surf, shrink=0.5, aspect=10)


## W
# sin
fig = plt.figure()
ax = fig.add_subplot(2, 4, 1, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 2 + 0].numpy())
ax = fig.add_subplot(2, 4, 2, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 2 + 1].numpy())
ax = fig.add_subplot(2, 4, 3, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 2 + 2].numpy())
ax = fig.add_subplot(2, 4, 4, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 2 + 3].numpy())
ax = fig.add_subplot(2, 4, 5, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 2 + 4].numpy())
ax = fig.add_subplot(2, 4, 6, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 2 + 5].numpy())
ax = fig.add_subplot(2, 4, 7, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 2 + 6].numpy())
ax = fig.add_subplot(2, 4, 8, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 2 + 7].numpy())
# plt.show()
plt.tight_layout()
plt.suptitle('w_sin')
plt.savefig('w_sin.png')
# cos
fig = plt.figure()
ax = fig.add_subplot(2, 4, 1, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 3 + 0].numpy())
ax = fig.add_subplot(2, 4, 2, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 3 + 1].numpy())
ax = fig.add_subplot(2, 4, 3, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 3 + 2].numpy())
ax = fig.add_subplot(2, 4, 4, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 3 + 3].numpy())
ax = fig.add_subplot(2, 4, 5, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 3 + 4].numpy())
ax = fig.add_subplot(2, 4, 6, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 3 + 5].numpy())
ax = fig.add_subplot(2, 4, 7, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 3 + 6].numpy())
ax = fig.add_subplot(2, 4, 8, projection='3d'); plot3d(ax, out[0, num_channel // 4 * 3 + 7].numpy())
# plt.show()
plt.tight_layout()
plt.suptitle('w_cos')
plt.savefig('w_cos.png')




## H
# sin
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(2, 4, 1, projection='3d'); plot3d(ax, out[0, 0].numpy())
ax = fig.add_subplot(2, 4, 2, projection='3d'); plot3d(ax, out[0, 1].numpy())
ax = fig.add_subplot(2, 4, 3, projection='3d'); plot3d(ax, out[0, 2].numpy())
ax = fig.add_subplot(2, 4, 4, projection='3d'); plot3d(ax, out[0, 3].numpy())
ax = fig.add_subplot(2, 4, 5, projection='3d'); plot3d(ax, out[0, 4].numpy())
ax = fig.add_subplot(2, 4, 6, projection='3d'); plot3d(ax, out[0, 5].numpy())
ax = fig.add_subplot(2, 4, 7, projection='3d'); plot3d(ax, out[0, 6].numpy())
ax = fig.add_subplot(2, 4, 8, projection='3d'); plot3d(ax, out[0, 7].numpy())
# plt.show()
plt.tight_layout()
plt.suptitle('h_sin')
plt.savefig('h_sin.png')

# cos
fig = plt.figure()
ax = fig.add_subplot(2, 4, 1, projection='3d'); plot3d(ax, out[0, num_channel // 4 + 0].numpy())
ax = fig.add_subplot(2, 4, 2, projection='3d'); plot3d(ax, out[0, num_channel // 4 + 1].numpy())
ax = fig.add_subplot(2, 4, 3, projection='3d'); plot3d(ax, out[0, num_channel // 4 + 2].numpy())
ax = fig.add_subplot(2, 4, 4, projection='3d'); plot3d(ax, out[0, num_channel // 4 + 3].numpy())
ax = fig.add_subplot(2, 4, 5, projection='3d'); plot3d(ax, out[0, num_channel // 4 + 4].numpy())
ax = fig.add_subplot(2, 4, 6, projection='3d'); plot3d(ax, out[0, num_channel // 4 + 5].numpy())
ax = fig.add_subplot(2, 4, 7, projection='3d'); plot3d(ax, out[0, num_channel // 4 + 6].numpy())
ax = fig.add_subplot(2, 4, 8, projection='3d'); plot3d(ax, out[0, num_channel // 4 + 7].numpy())
# plt.show()
plt.tight_layout()
plt.suptitle('h_cos')
plt.savefig('h_cos.png')



# plt.subplot(2, 4, 1); plt.imshow(out[0, 0])
# plt.subplot(2, 4, 2); plt.imshow(out[0, 1])
# plt.subplot(2, 4, 3); plt.imshow(out[0, 2])
# plt.subplot(2, 4, 4); plt.imshow(out[0, 3])
# plt.subplot(2, 4, 5); plt.imshow(out[0, 4])
# plt.subplot(2, 4, 6); plt.imshow(out[0, 5])
# plt.subplot(2, 4, 7); plt.imshow(out[0, 6])
# plt.subplot(2, 4, 8); plt.imshow(out[0, 7])
# plt.show()



