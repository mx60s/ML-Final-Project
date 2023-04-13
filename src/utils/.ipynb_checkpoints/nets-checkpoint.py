import torch
from torch import nn
import random
import numpy as np

import utils.ciivae_etc as util

# TODO: need to implement a piVAE/conv-piVAE for tests?

# NOTE: their models are lists of modules (which are optimized seperately in train) instead of actual Pytorch models. We're ollowing that convention for now.
# Code taken from CI-iVAE repo

def CIiVAE(dim_x, dim_u,
          dim_z=16, prior_node_list=[128, 128],
          encoder_node_list=[4096, 4096],
          decoder_node_list=[4096, 4096],
          decoder_final_activation='sigmoid'):
    '''
    dim_z: dimension of representations
    prior_node_list: list of number of nodes in layers in label prior networks
    encoder_node_list: list of number of nodes in layers in encoder networks
    decoder_node_list: list of number of nodes in layers in decoder networks
    decoder_final_activation: the last activation layer in decoder. Please choose 'sigmoid' or 'None' 
    '''
    
    prior = Prior_conti(dim_z, dim_u, prior_node_list)
    encoder = Encoder(dim_x, dim_z, encoder_node_list)
    decoder = Decoder(dim_z, dim_x, decoder_node_list,
                            final_activation=decoder_final_activation)
    return [prior, encoder, decoder]

# TODO remove 
"""
def CIiVAE(nn.Module):
    def __init__(ConvPiVAE, self, dim_x, dim_u, 
            dim_z=16, prior_node_list=[128, 128],
            encoder_node_list=[4096, 4096],
            decoder_node_list=[4096, 4096],
            decoder_final_activation='sigmoid'):
        '''
        dim_z: dimension of representations
        prior_node_list: list of number of nodes in layers in label prior networks
        encoder_node_list: list of number of nodes in layers in encoder networks
        decoder_node_list: list of number of nodes in layers in decoder networks
        decoder_final_activation: the last activation layer in decoder. Please choose 'sigmoid' or 'None' 
        '''
        
        super(ConvPiVAE, self).__init__()
        
        self.prior = Prior_conti(dim_z, dim_u, prior_node_list)
        self.encoder = Encoder(dim_x, dim_z, encoder_node_list)
        self.decoder = Decoder(dim_z, dim_x, decoder_node_list,
                               final_activation=decoder_final_activation)
        
    def forward(self, x, u):
        lam_mean, lam_log_var = self.prior(u)
        z_mean, z_log_var = self.encoder(x)
        post_mean, post_log_var = util.compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
        post_sample = util.sampling(post_mean, post_log_var)
        encoded_sample = util.sampling(z_mean, z_log_var)
        
        fire_rate_post, obs_log_var = self.decoder(post_sample)
            fire_rate_encoded, _ = self.decoder(encoded_sample)
            
        return post_sample, encoded_sample, fire_rate_post, 
                obs_log_var, fire_rate_encoded
"""
# encoder changed to the cebra one, decoder changed to 2d output
# not sure what that means for the output
def ConvCIiVAE(dim_x, dim_u,
          dim_z=16, prior_node_list=[128, 128],
          encoder_node_list=[4096, 4096],
          decoder_node_list=[4096, 4096],
          decoder_final_activation='sigmoid'):
    '''
    dim_z: dimension of representations
    prior_node_list: list of number of nodes in layers in label prior networks
    encoder_node_list: list of number of nodes in layers in encoder networks
    decoder_node_list: list of number of nodes in layers in decoder networks
    decoder_final_activation: the last activation layer in decoder. Please choose 'sigmoid' or 'None' 
    '''
    
    prior = Prior_conti(dim_z, dim_u, prior_node_list)
    encoder = Encoder(dim_x, dim_z, encoder_node_list)
    decoder = Decoder(dim_z, dim_x, decoder_node_list,
                            final_activation=decoder_final_activation)
    return [prior, encoder, decoder]


class Prior_conti(nn.Module):
    def __init__(self, dim_z, dim_u, prior_node_list):
        super(Prior_conti, self).__init__()
        
        self.dim_z, self.dim_u = dim_z, dim_u
        self.prior_node_list = prior_node_list
        
        self.mu_net = nn.ModuleList()
        self.log_var_net = nn.ModuleList()
        
        # input dimension is dim_u
        self.mu_net.append(nn.Linear(self.dim_u, self.prior_node_list[0]))
        self.mu_net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.log_var_net.append(nn.Linear(self.dim_u, self.prior_node_list[0]))
        self.log_var_net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        if len(self.prior_node_list) > 1:
            for i in range(len(self.prior_node_list)-1):
                self.mu_net.append(nn.Linear(self.prior_node_list[i],
                                             self.prior_node_list[i+1]))
                self.mu_net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                self.log_var_net.append(nn.Linear(self.prior_node_list[i],
                                                  self.prior_node_list[i+1]))
                self.log_var_net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                
            del(i)
            
        self.mu_net.append(nn.Linear(self.prior_node_list[-1], self.dim_z))
        self.log_var_net.append(nn.Linear(self.prior_node_list[-1], self.dim_z))
        
    def forward(self, u_input):
        h_mu, h_log_var = self.mu_net[0](u_input), self.log_var_net[0](u_input)
        if len(self.mu_net) > 1:
            for i in range(len(self.mu_net)-1):
                h_mu = self.mu_net[i+1](h_mu)
                h_log_var = self.log_var_net[i+1](h_log_var)
            del(i)
        return h_mu, h_log_var

    
class Encoder(nn.Module):
    def __init__(self, dim_x, dim_z, encoder_node_list):
        super(Encoder, self).__init__()
        
        self.dim_x, self.dim_z = dim_x, dim_z
        self.encoder_node_list = encoder_node_list
        
        self.main = nn.ModuleList()
        
        # input dimension is dim_x
        self.main.append(nn.Linear(self.dim_x, self.encoder_node_list[0]))
        self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        if len(self.encoder_node_list) > 1:
            for i in range(len(self.encoder_node_list)-1):
                self.main.append(nn.Linear(self.encoder_node_list[i],
                                          self.encoder_node_list[i+1]))
                self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            del(i)
        
        # input dimension is gen_nodes
        self.mu_net = nn.Linear(self.encoder_node_list[-1], self.dim_z)
        self.log_var_net = nn.Linear(self.encoder_node_list[-1], self.dim_z)
    
    def forward(self, x_input):
        h = self.main[0](x_input)
        
        if len(self.main) > 1:
            for i in range(len(self.main)-1):
                h = self.main[i+1](h)
                
            del(i)
            
        mu, log_var = self.mu_net(h), self.log_var_net(h)
        
        return mu, log_var
    
# receptive field of 10 samples
class CebraConvEncoder(nn.Module):
    def __init__(self, dim_x, dim_z, encoder_node_list):
        super(Encoder, self).__init__()
        
        self.dim_x, self.dim_z = dim_x, dim_z
        self.encoder_node_list = encoder_node_list
        
        self.main = nn.ModuleList()

        self.main.append(nn.Linear(self.dim_x, self.encoder_node_list[0]))
        self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        if len(self.encoder_node_list) > 1:
            for i in range(len(self.encoder_node_list)-1):
                self.main.append(nn.Linear(self.encoder_node_list[i],
                                          self.encoder_node_list[i+1]))
                self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                
            del(i)
        
        # input dimension is gen_nodes
        self.mu_net = nn.Linear(self.encoder_node_list[-1], self.dim_z)
        self.log_var_net = nn.Linear(self.encoder_node_list[-1], self.dim_z)
    
    def forward(self, x_input):
        h = self.main[0](x_input)
        
        if len(self.main) > 1:
            for i in range(len(self.main)-1):
                h = self.main[i+1](h)
                
            del(i)
            
        mu, log_var = self.mu_net(h), self.log_var_net(h)
        
        return mu, log_var
        
        
class Decoder(nn.Module):
    def __init__(self, dim_z, dim_x, decoder_node_list, final_activation=None):
        super(Decoder, self).__init__()
        
        self.dim_z, self.dim_x = dim_z, dim_x
        self.decoder_node_list = decoder_node_list
        
        self.main = nn.ModuleList()
        # input dimension is dim_z
        
        self.main.append(nn.Linear(self.dim_z, self.decoder_node_list[0]))
        self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        if len(self.decoder_node_list) > 1:
            for i in range(len(self.decoder_node_list)-1):
                self.main.append(nn.Linear(self.decoder_node_list[i],
                                          self.decoder_node_list[i+1]))
                self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                
            del(i)
        
        # input dimension is gen_nodes
        if final_activation == 'sigmoid':
            self.mu_net = nn.Sequential(
                            nn.Linear(self.decoder_node_list[-1], self.dim_x),
                            nn.Sigmoid()
                            )
        elif final_activation == None:
            self.mu_net = nn.Sequential(
                            nn.Linear(self.decoder_node_list[-1], self.dim_x)
                            )
            
        self.obs_log_var_net = nn.Linear(1, self.dim_x)
    
    def forward(self, z_input):
        h = self.main[0](z_input)
        
        if len(self.main) > 1:
            for i in range(len(self.main)-1):
                h = self.main[i+1](h)
                
            del(i)
            
        o = self.mu_net(h)
        
        device = z_input.get_device()
        if device == -1:
            one_tensor = torch.ones((1,1))
            obs_log_var = self.obs_log_var_net(one_tensor)
            return o, obs_log_var
        else:
            one_tensor = torch.ones((1,1)).cuda()
            obs_log_var = self.obs_log_var_net(one_tensor)
            return o, obs_log_var
