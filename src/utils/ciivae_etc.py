import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
import math

# adapting from https://discuss.pytorch.org/t/1d-gaussian-kernel/160899/2
# https://stackoverflow.com/questions/3149279/optimal-sigma-for-gaussian-filtering-of-an-image
# sigma=(k-1)/6
# TODO make sure that this makes any sense lol
# kernel should be shape [120,120,10]
def gaussian_conv(x, sigma, device='cpu'):
    # not sure if it should be an arange or just ones or what. can't find anything on that.
    #weights = torch.arange(1, 11, dtype=torch.float, device=x.device)
    #weights = torch.unsqueeze(weights, 0)
    #weights = torch.unsqueeze(weights, 2)

    weights = torch.ones(1, 10, 1)
    kernel = gaussian_filter1d(weights, sigma)
    
    return torch.nn.functional.conv1d(x, torch.from_numpy(kernel).to(device))

def compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var):
    # q(z) = q(z|x)p(z|u) = N((mu1*var2+mu2*var1)/(var1+var2), var1*var2/(var1+var2));
    
    post_mean = (z_mean/(1+torch.exp(z_log_var-lam_log_var))) + (lam_mean/(1+torch.exp(lam_log_var-z_log_var)))
    post_log_var = z_log_var + lam_log_var - torch.log(torch.exp(z_log_var) + torch.exp(lam_log_var))
    
    return post_mean, post_log_var

def sampling(mean, log_var):
    device = mean.get_device()
    if device == -1:
        epsilon = torch.randn(mean.shape)
        return mean + torch.exp(0.5 * log_var) * epsilon
    else:
        epsilon = torch.randn(mean.shape).cuda()
        return mean + torch.exp(0.5 * log_var).cuda() * epsilon

# Ref: https://github.com/edenton/svg/blob/master/train_svg_lp.py#L131-#L138
def kl_criterion(mu1, log_var1, mu2, log_var2):
    sigma1 = log_var1.mul(0.5).exp() 
    sigma2 = log_var2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(log_var1) + (mu1 - mu2)**2)/(2*torch.exp(log_var2)) - 1/2
    return torch.mean(kld, dim=-1)

def extract_feature(result_path, x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    saved_model = torch.load('%s/model.pth' % result_path)
    prior, encoder, decoder = saved_model['prior'], saved_model['encoder'], saved_model['decoder']
    prior.eval(); encoder.eval(); decoder.eval()
    
    if device == 'cuda':
        z_mean, z_log_var = encoder(x.cuda())
    elif device == 'cpu':
        z_mean, z_log_var = encoder(x)
    z_sample = sampling(z_mean, z_log_var)
    return z_sample

def generate_z(result_path, u):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    saved_model = torch.load('%s/model.pth' % result_path)
    prior, encoder, decoder = saved_model['prior'], saved_model['encoder'], saved_model['decoder']
    prior.eval(); encoder.eval(); decoder.eval()
    
    u = u.cuda() if device == 'cuda' else u
    z_mean, z_log_var = prior(u.cuda())
    return z_mean, z_log_var
    