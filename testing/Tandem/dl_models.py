import pickle
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A basic 1D Convolutional Block.

    Parameters:
    -----------
    c_in : int
        Number of input channels.
    c_out : int
        Number of output channels.
    act_fn : nn.Module, optional
        Activation function to apply after the convolution. Default is ReLU.
    """
    def __init__(self, c_in, c_out, act_fn = nn.ReLU(),**kwargs):
        super().__init__()
        # Define a sequential block with a 1D convolution and an activation function.
        # The convolution has a kernel size of 3 and padding is set to 'same', ensuring
        # output dimensions match the input dimensions along the time axis.
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=3, padding='same'),
            act_fn  # Apply the specified activation function
        )

    def forward(self, x):
        # Pass the input through the sequential network (convolution + activation)
        x = self.net(x)
        return x



class ResNetBlock(nn.Module):
    """
    A 1D Residual Block with optional Batch Normalization.

    Parameters:
    -----------
    c_in : int
        Number of input channels.
    c_out : int
        Number of output channels.
    batch_norm : bool, optional
        If True, applies batch normalization after each convolution. Default is False.
    act_fn : nn.Module, optional
        Activation function to apply after each convolution and at the end. Default is ReLU.
    """
    def __init__(self, c_in, c_out, batch_norm = False, act_fn = nn.ReLU,**kwargs):
        super().__init__()

        # Define the main network block depending on whether batch normalization is used.
        if not batch_norm:
            # Sequential block without batch normalization
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=3, padding='same', stride=1),
                act_fn(),  # Apply activation function
                nn.Conv1d(c_out, c_out, kernel_size=3, padding='same', stride=1),
                act_fn(),  # Apply activation function
                nn.Conv1d(c_out, c_out, kernel_size=3, padding='same', stride=1)
            )
        else:
            # Sequential block with batch normalization
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=3, padding='same', stride=1, bias=False),
                nn.BatchNorm1d(c_out),
                act_fn(),  # Apply activation function
                nn.Conv1d(c_out, c_out, kernel_size=3, padding='same', stride=1, bias=False),
                nn.BatchNorm1d(c_out),
                act_fn(),  # Apply activation function
                nn.Conv1d(c_out, c_out, kernel_size=3, padding='same', stride=1, bias=False),
                nn.BatchNorm1d(c_out),
            )

        # Downsampling layer (only used if the input and output channels differ)
        self.downsample = nn.Conv1d(c_in, c_out, kernel_size=3, padding='same', stride=1) if c_in != c_out else None
        # Final activation function to apply after the residual connection
        self.endAct = act_fn()

    def forward(self, x):
        # Pass input through the main network block
        z = self.net(x)
        # Apply downsampling if necessary (input and output dimensions differ)
        if self.downsample is not None:
            x = self.downsample(x)
        # Add the residual (input) to the output of the main block
        out = z + x
        # Apply the final activation function
        out = self.endAct(out)
        return out

class IdMapBlock(nn.Module):
    """
    A 1D Identity Mapping Block with optional Batch Normalization.

    Parameters:
    -----------
    c_in : int
        Number of input channels.
    c_out : int
        Number of output channels.
    batch_norm : bool, optional
        If True, applies batch normalization after each convolution. Default is False.
    act_fn : nn.Module, optional
        Activation function to apply after each convolution and at the end. Default is ReLU.
    """
    def __init__(self, c_in, c_out, batch_norm = False, act_fn = nn.ReLU(),**kwargs):
        super().__init__()
        # Define the network depending on whether batch normalization is used.
        if not batch_norm:
            # Without batch normalization
            self.net = nn.Sequential(
                act_fn,  # Activation function before the first convolution
                nn.Conv1d(c_in, c_out, kernel_size=3, padding='same', stride=1),
                act_fn,
                nn.Conv1d(c_out, c_out, kernel_size=3, padding='same', stride=1),
                act_fn,
                nn.Conv1d(c_out, c_out, kernel_size=3, padding='same', stride=1)
            )
        else:
            # With batch normalization
            self.net = nn.Sequential(
                nn.BatchNorm1d(c_in),  # Batch normalization applied to input
                act_fn,  # Activation after batch normalization
                nn.Conv1d(c_in, c_out, kernel_size=3, padding='same', stride=1, bias=False),
                nn.BatchNorm1d(c_out),  # Batch normalization after convolution
                act_fn,
                nn.Conv1d(c_out, c_out, kernel_size=3, padding='same', stride=1, bias=False),
                nn.BatchNorm1d(c_out),  # Batch normalization after second convolution
                act_fn,
                nn.Conv1d(c_out, c_out, kernel_size=3, padding='same', stride=1, bias=False)
            )

        # Downsample if input and output channels differ
        self.downsample = nn.Conv1d(c_in, c_out, kernel_size=3, padding='same', stride=1) if c_in != c_out else None
        # Final activation function after the residual connection
        self.endAct = act_fn

    def forward(self, x):
        # Pass input through the network (with/without batch normalization)
        z = self.net(x)
        # Apply downsampling if input and output channels differ
        if self.downsample is not None:
            x = self.downsample(x)
        # Add residual connection (input + processed output)
        out = z + x
        # Apply final activation
        out = self.endAct(out)
        return out





class ResNeXtBlock(nn.Module):
    """
    A 1D ResNeXt Block with optional Batch Normalization.

    Parameters:
    -----------
    c_in : int
        Number of input channels.
    c_out : int
        Number of output channels.
    N_groups : int
        Number of groups for group convolution.
    N_bottleneck : float
        Bottleneck ratio (determines the bottleneck channel size as a fraction of c_out).
    batch_norm : bool, optional
        If True, applies batch normalization after each convolution. Default is False.
    act_fn : nn.Module, optional
        Activation function to apply after each convolution and at the end. Default is ReLU.
    """
    def __init__(self, c_in, c_out, N_groups = None, N_bottleneck = None, batch_norm = True, act_fn = nn.ReLU(),**kwargs):
        super().__init__()
        self.batch_norm = batch_norm
        # Compute the bottleneck channel size based on the bottleneck ratio
        c_bottleneck = int(c_out * N_bottleneck)
        # Define the main network block depending on the use of batch normalization
        if not batch_norm:
            # Without batch normalization
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_bottleneck, kernel_size=3, padding='same', stride=1),
                act_fn,  # Activation after first convolution
                nn.Conv1d(c_bottleneck, c_bottleneck, groups=c_bottleneck//N_groups, kernel_size=3, padding='same', stride=1),
                act_fn,  # Activation after group convolution
                nn.Conv1d(c_bottleneck, c_out, kernel_size=1, padding='same', stride=1)  # Pointwise convolution
            )
        else:
            # With batch normalization
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_bottleneck, kernel_size=3, padding='same', stride=1, bias=False),
                nn.BatchNorm1d(c_bottleneck),  # Batch normalization after the first convolution
                act_fn,
                nn.Conv1d(c_bottleneck, c_bottleneck, groups=c_bottleneck//N_groups, kernel_size=3, padding='same', stride=1, bias=False),
                nn.BatchNorm1d(c_bottleneck),  # Batch normalization after group convolution
                act_fn,
                nn.Conv1d(c_bottleneck, c_out, kernel_size=1, padding='same', stride=1, bias=False),
                nn.BatchNorm1d(c_out)  # Batch normalization after pointwise convolution
            )
        # Downsampling layer to match the input/output dimensions when necessary
        self.downsample = nn.Conv1d(c_in, c_out, kernel_size=3, padding='same', stride=1) if c_in != c_out else None
        # Final activation function after the residual connection
        self.endAct = act_fn

    def forward(self, x):
        z = self.net(x)
        # Apply downsampling if input and output dimensions differ
        if self.downsample is not None:
            x = self.downsample(x)
        # Apply batch normalization if batch_norm is enabled
        if self.batch_norm:
            x = nn.functional.batch_norm(x)
        # Add the residual connection (input + processed output)
        out = z + x
        # Apply final activation
        out = self.endAct(out)
        return out
    
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.activation(x + self.block(x))
    
class FullyConnectedResNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_blocks=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()  #outputs should be non-negative
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.output_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim)
        )
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.activation(self.block(x))


class FullyConnected(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_blocks=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.Sequential(*[Block(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()  #outputs should be non-negative
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.output_proj(x)
        return x
