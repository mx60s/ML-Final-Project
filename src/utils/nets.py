import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import random
import sys
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder
# import progressbar
import tqdm

import utils.ciivae_etc as UTIL
from scipy.ndimage import gaussian_filter1d

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


# encoder changed to the cebra one, decoder changed to 2d output
# not sure what that means for the output
def ConvCIiVAE(dim_x, dim_u,
          dim_z=16,
          decoder_final_activation='sigmoid',
          time_window=10,
          gen_nodes=60):
    '''
    dim_z: dimension of representations
    prior_node_list: list of number of nodes in layers in label prior networks
    encoder_node_list: list of number of nodes in layers in encoder networks
    decoder_node_list: list of number of nodes in layers in decoder networks
    decoder_final_activation: the last activation layer in decoder. Please choose 'sigmoid' or 'None' 
    '''
    
    prior = Prior_conti(dim_z, dim_u)
    encoder = CebraConvEncoder(dim_x, dim_z, time_window, gen_nodes)
    decoder = Decoder(dim_z, dim_x, final_activation=decoder_final_activation)
    return [prior, encoder, decoder]

class Encoder(nn.Module):
    def __init__(self, dim_x, dim_z, encoder_node_list):
        super(Encoder, self).__init__()
        
        self.dim_x, self.dim_z = dim_x, dim_z
        self.encoder_node_list = encoder_node_list # hidden dims
        
        self.main = nn.ModuleList()
        
        # input dimension is dim_x
        self.main.append(nn.Linear(self.dim_x, self.encoder_node_list[0]))
        self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        if len(self.encoder_node_list) > 1:
            for i in range(len(self.encoder_node_list)-1):
                self.main.append(nn.Linear(self.encoder_node_list[i],
                                          self.encoder_node_list[i+1]))
                self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # input dimension is gen_nodes
        self.mu_net = nn.Linear(self.encoder_node_list[-1], self.dim_z)
        self.log_var_net = nn.Linear(self.encoder_node_list[-1], self.dim_z)
    
    def forward(self, x_input):
        h = self.main[0](x_input)
        
        if len(self.main) > 1:
            for i in range(len(self.main)-1):
                h = self.main[i+1](h)

        mu, log_var = self.mu_net(h), self.log_var_net(h)
        
        return mu, log_var


def fit(model, dataloader_dict,
        num_epoch=100, seed=0,
        beta=0.01, Adam_beta1=0.5, Adam_beta2=0.999, weight_decay=5e-6,
        init_lr=5e-5, lr_milestones=[25, 50, 75], lr_gamma=0.5,
        dtype=torch.float32, M=50, alpha_step=0.025,
        fix_alpha=None, result_path=None, sigma=0.01, batch_size=200,
        num_worker=1):
    '''
    num_epoch: the number of epoch
    seed: the random seed number
    beta: the coefficient of KL-penalty term in ELBOs
    Adam_beta1: beta1 for Adam optimizer
    Adam_beta2: beta2 for Adam optimizer
    weight_decay: the coefficient of the half of L2 penalty term
    init_lr: the initial learning rate
    lr_milestones: the epochs to reduce the learning rate
    lr_gamma: the multiplier for each time learning rate is reduced
    dtype: the data type
    M: the number of MC samples to approximate skew KL-divergences
    alpha_step: the distance between each grid points in finding samplewise optimal alpha
    fix_alpha: If it is None, the objective function is supremum of ELBO(\alpha) over \alpha. If it is a real number in [0, 1], the objective function is the ELBO(\alpha). For example, when it is 0.0, the objective is the ELBO of iVAEs.
    result_path: the directory where results are saved
    '''
    
    # declare basic variables
    prior, encoder, decoder = model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    beta_kl_post_prior = beta
    beta_kl_encoded_prior = beta
    Adam_betas = (Adam_beta1, Adam_beta2)
    if result_path is None:
        now = datetime.datetime.now()
        result_path = './results/ci_ivae-time=%d-%d-%d-%d-%d' % (now.month, now.day, now.hour, now.minute, now.second)
    os.makedirs(result_path, exist_ok=True)
    
    # lines for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # define optimizers and schedulers
    enc_optimizer = torch.optim.Adam(encoder.parameters(),
                                     betas=Adam_betas,
                                     lr=init_lr,
                                     weight_decay=weight_decay)
    gen_optimizer = torch.optim.Adam(list(decoder.parameters())
                                     +list(prior.parameters()),
                                     betas=Adam_betas,
                                     lr=init_lr,
                                     weight_decay=weight_decay)

    enc_scheduler = torch.optim.lr_scheduler.MultiStepLR(enc_optimizer,
                                                         milestones=lr_milestones,
                                                         gamma=lr_gamma)
    gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(gen_optimizer,
                                                         milestones=lr_milestones,
                                                         gamma=lr_gamma)

    # define training log    
    loss_names = ['loss', 'recon_loss_post', 'kl_post_prior',
                  'recon_loss_encoded', 'kl_encoded_prior', 'l2_penalty']
    logs = {}
    for datasetname in ['train', 'val']:
        logs[datasetname] = {}
        for loss_name in loss_names:
            logs[datasetname][loss_name] = []
            
    summary_stats = []
    
    # training part
    mse_criterion = torch.nn.MSELoss()
    #if device == 'cuda':
    prior.to(device)
    encoder.to(device)
    decoder.to(device)
    mse_criterion.to(device)

    for epoch in range(1, num_epoch+1):
        num_batch = 0
        #maggie = 0
        for x_batch, u_batch in tqdm.tqdm(dataloader_dict['train'],
                                         desc='[Epoch %d/%d] Training' % (epoch, num_epoch)):
            #if maggie == 10:
            #  break
            #else:
            #  maggie += 1
            
            x_batch, u_batch = x_batch.to(device), u_batch.to(device)
                
            # why are they adding what I assume to be noise?
            x_batch += torch.randn_like(x_batch)*1e-2

            prior.train()
            encoder.train()
            decoder.train()

            enc_optimizer.zero_grad()
            gen_optimizer.zero_grad()

            # forward step
            lam_mean, lam_log_var = prior(u_batch)
            
            #print('x_batch sample')
            #print(x_batch.shape)
            x_batch_reshape = torch.transpose(x_batch, 1, 2)
            x_batch_fire_rate = torch.squeeze(UTIL.gaussian_conv(x_batch_reshape, sigma, device=device))
            #print('x_batch_fire_rate shape', x_batch_fire_rate.shape)
            #print('x_batch_fire_rate sample')
            #print(x_batch_fire_rate[0][:30])
            
            z_mean, z_log_var = encoder(x_batch_fire_rate)
            
            z_mean = torch.squeeze(z_mean)
            z_log_var = torch.squeeze(z_log_var)
            
            post_mean, post_log_var = UTIL.compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
            post_sample = UTIL.sampling(post_mean, post_log_var)
            encoded_sample = UTIL.sampling(z_mean, z_log_var)

            epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M), device=device)
                
            post_sample = post_sample.to(device)
            encoded_sample = encoded_sample.to(device)
                
            fire_rate_post = decoder(post_sample)
            fire_rate_encoded = decoder(encoded_sample)
            
            obs_loglik_post = -torch.mean((fire_rate_post - x_batch_fire_rate)**2, dim=1)
            obs_loglik_encoded = -torch.mean((fire_rate_encoded - x_batch_fire_rate)**2, dim=1)
            
            # TODO remove after debugging
            #obs_loglik_encoded = torch.mean(obs_loglik_encoded, dim=1)
            #print('obs_loglik_encoded', obs_loglik_encoded.shape)
            #obs_loglik_post = torch.mean(obs_loglik_post, dim=1)
            
            kl_post_prior = UTIL.kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
            kl_encoded_prior = UTIL.kl_criterion(z_mean, z_log_var, lam_mean, lam_log_var)
            
            elbo_post = obs_loglik_post - beta_kl_post_prior*kl_post_prior
            elbo_encoded = obs_loglik_encoded - beta_kl_encoded_prior*kl_encoded_prior

            z_mean_tiled = torch.tile(torch.unsqueeze(z_mean, 2), [1, 1, M])
            z_log_var_tiled = torch.tile(torch.unsqueeze(z_log_var, 2), [1, 1, M])
            z_sample_tiled = z_mean_tiled + torch.exp(0.5 * z_log_var_tiled) * epsilon

            post_mean_tiled = torch.tile(torch.unsqueeze(post_mean, 2), [1, 1, M])
            post_log_var_tiled = torch.tile(torch.unsqueeze(post_log_var, 2), [1, 1, M])
            post_sample_tiled = post_mean_tiled + torch.exp(0.5 * post_log_var_tiled) * epsilon

            log_z_density_with_post_sample = -torch.sum((post_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
            log_post_density_with_post_sample = -torch.sum((post_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
            log_z_density_with_z_sample = -torch.sum((z_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
            log_post_density_with_z_sample = -torch.sum((z_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)

            if fix_alpha is not None:
                if fix_alpha == 0.0:
                    loss = torch.mean(-elbo_post)
                elif fix_alpha == 1.0:
                    loss = torch.mean(-elbo_encoded)
                else:
                    ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                    ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                    skew_kl_post = torch.log(1.0/(fix_alpha*ratio_z_over_post_with_post_sample+(1.0-fix_alpha)))
                    skew_kl_post = torch.abs(torch.mean(skew_kl_post, dim=-1))
                    skew_kl_encoded = torch.log(1.0/(fix_alpha+(1.0-fix_alpha)*ratio_post_over_z_with_z_sample))
                    skew_kl_encoded = torch.abs(torch.mean(skew_kl_encoded, dim=-1))
                    loss = -fix_alpha*elbo_encoded-(1.0-fix_alpha)*elbo_post+fix_alpha*skew_kl_encoded+(1.0-fix_alpha)*skew_kl_post
            else:
                alpha_list = np.arange(alpha_step, 1.0, alpha_step)
                loss = torch.zeros((elbo_post.shape[0], len(alpha_list)))
                i = 0
                for alpha in alpha_list:
                    ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                    ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                    skew_kl_post = torch.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                    skew_kl_post = torch.abs(torch.mean(skew_kl_post, dim=-1))
                    skew_kl_encoded = torch.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                    skew_kl_encoded = torch.abs(torch.mean(skew_kl_encoded, dim=-1))
                    loss[:, i] = -alpha*elbo_encoded-(1.0-alpha)*elbo_post+alpha*skew_kl_encoded+(1.0-alpha)*skew_kl_post
                    i += 1
                loss, _ = torch.min(loss, dim = 1)
            loss = torch.mean(loss)
            #print("train loss", loss)
            # backward step
            loss.backward()

            enc_optimizer.step()
            gen_optimizer.step()
        
        prior.eval()
        encoder.eval()
        decoder.eval()
        for datasetname in ['train', 'val']:
            loss_cumsum, sample_size = 0.0, 0
            obs_loglik_post_cumsum, kl_post_prior_cumsum = 0.0, 0.0
            obs_loglik_encoded_cumsum, kl_encoded_prior_cumsum = 0.0, 0.0
            #maggie = 0
            for x_batch, u_batch in tqdm.tqdm(dataloader_dict[datasetname],
                                         desc='[Epoch %d/%d] Computing loss terms on %s' % (epoch, num_epoch, datasetname)):
                
                #if maggie == 10:
                #  break
                #else:
                #  maggie += 1

                x_batch, u_batch = x_batch.to(device), u_batch.to(device)

                # forward step
                lam_mean, lam_log_var = prior(u_batch)
                
                #print('x_batch', x_batch.shape)
                x_batch_reshape = torch.transpose(x_batch, 1, 2) # idk maybe this reshaping shouldn't happen
                x_batch_fire_rate = torch.squeeze(UTIL.gaussian_conv(x_batch_reshape, sigma, device=device))
                z_mean, z_log_var = encoder(x_batch_fire_rate)
                
                z_mean = torch.squeeze(z_mean)
                z_log_var = torch.squeeze(z_log_var)

                post_mean, post_log_var = UTIL.compute_posterior(z_mean, z_log_var, lam_mean, lam_log_var)
                post_sample = UTIL.sampling(post_mean, post_log_var)
                encoded_sample = UTIL.sampling(z_mean, z_log_var)

                epsilon = torch.randn((z_mean.shape[0], z_mean.shape[1], M))
                #if device == 'cuda':
                # TODO this is not efficient at all
                post_sample = post_sample.to(device)
                encoded_sample = encoded_sample.to(device)
                epsilon = epsilon.to(device)

                fire_rate_post= decoder(post_sample) # torch.clip(decoder(post_sample), min_value=1e-7, max_value=1e7)
                fire_rate_encoded = decoder(encoded_sample) #torch.clip(decoder(encoded_sample) #, min_value=1e-7, max_value=1-1e-7)
                
                obs_loglik_post = -torch.mean((fire_rate_post - x_batch_fire_rate)**2, dim=1)
                obs_loglik_encoded = -torch.mean((fire_rate_encoded - x_batch_fire_rate)**2, dim=1)
                
                kl_post_prior = UTIL.kl_criterion(post_mean, post_log_var, lam_mean, lam_log_var)
                kl_encoded_prior = UTIL.kl_criterion(z_mean, z_log_var, lam_mean, lam_log_var)
                
                elbo_post = obs_loglik_post - beta_kl_post_prior*kl_post_prior
                elbo_encoded = obs_loglik_encoded - beta_kl_encoded_prior*kl_encoded_prior

                z_mean_tiled = torch.tile(torch.unsqueeze(z_mean, 2), [1, 1, M])
                z_log_var_tiled = torch.tile(torch.unsqueeze(z_log_var, 2), [1, 1, M])
                z_sample_tiled = z_mean_tiled + torch.exp(0.5 * z_log_var_tiled) * epsilon

                post_mean_tiled = torch.tile(torch.unsqueeze(post_mean, 2), [1, 1, M])
                post_log_var_tiled = torch.tile(torch.unsqueeze(post_log_var, 2), [1, 1, M])
                post_sample_tiled = post_mean_tiled + torch.exp(0.5 * post_log_var_tiled) * epsilon

                log_z_density_with_post_sample = -torch.sum((post_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                log_post_density_with_post_sample = -torch.sum((post_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)
                log_z_density_with_z_sample = -torch.sum((z_sample_tiled - z_mean_tiled)**2/(2*torch.exp(z_log_var_tiled))+(z_log_var_tiled/2), dim=1)
                log_post_density_with_z_sample = -torch.sum((z_sample_tiled - post_mean_tiled)**2/(2*torch.exp(post_log_var_tiled))+(post_log_var_tiled/2), dim=1)

                if fix_alpha is not None:
                    if fix_alpha == 0.0:
                        loss = torch.mean(-elbo_post)
                    elif fix_alpha == 1.0:
                        loss = torch.mean(-elbo_encoded)
                    else:
                        ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                        ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                        skew_kl_post = torch.log(1.0/(fix_alpha*ratio_z_over_post_with_post_sample+(1.0-fix_alpha)))
                        skew_kl_post = torch.abs(torch.mean(skew_kl_post, dim=-1))
                        skew_kl_encoded = torch.log(1.0/(fix_alpha+(1.0-fix_alpha)*ratio_post_over_z_with_z_sample))
                        skew_kl_encoded = torch.abs(torch.mean(skew_kl_encoded, dim=-1))
                        loss = -fix_alpha*elbo_encoded-(1.0-fix_alpha)*elbo_post+fix_alpha*skew_kl_encoded+(1.0-fix_alpha)*skew_kl_post
                else:
                    alpha_list = np.arange(alpha_step, 1.0, alpha_step)
                    loss = torch.zeros((elbo_post.shape[0], len(alpha_list)))
                    i = 0
                    for alpha in alpha_list:
                        ratio_z_over_post_with_post_sample = torch.exp(log_z_density_with_post_sample-log_post_density_with_post_sample)
                        ratio_post_over_z_with_z_sample = torch.exp(log_post_density_with_z_sample-log_z_density_with_z_sample)
                        skew_kl_post = torch.log(1.0/(alpha*ratio_z_over_post_with_post_sample+(1.0-alpha)))
                        skew_kl_post = torch.abs(torch.mean(skew_kl_post, dim=-1))
                        skew_kl_encoded = torch.log(1.0/(alpha+(1.0-alpha)*ratio_post_over_z_with_z_sample))
                        skew_kl_encoded = torch.abs(torch.mean(skew_kl_encoded, dim=-1))
                        loss[:, i] = -alpha*elbo_encoded-(1.0-alpha)*elbo_post+alpha*skew_kl_encoded+(1.0-alpha)*skew_kl_post
                        i += 1
                    loss, _ = torch.min(loss, dim = 1)
                loss = torch.mean(loss) # that being said isn't this wrong? why mean twice?
                
                loss_cumsum += loss.item()*np.shape(x_batch)[0]
                obs_loglik_post_cumsum += torch.mean(obs_loglik_post).item()*np.shape(x_batch)[0]
                kl_post_prior_cumsum += torch.mean(kl_post_prior).item()*np.shape(x_batch)[0]
                obs_loglik_encoded_cumsum += torch.mean(obs_loglik_encoded).item()*np.shape(x_batch)[0]
                kl_encoded_prior_cumsum += torch.mean(kl_encoded_prior).item()*np.shape(x_batch)[0]
                sample_size += np.shape(x_batch)[0]

            l2_penalty = 0.0
            for networks in [prior, encoder, decoder]:
                for name, m in networks.named_parameters():
                    if 'weight' in name:
                        l2_penalty += 0.5*torch.sum(m**2)
            #print("appending val loss")
            logs[datasetname]['loss'].append(loss_cumsum/sample_size)
            logs[datasetname]['recon_loss_post'].append(-obs_loglik_post_cumsum/sample_size)
            logs[datasetname]['kl_post_prior'].append(kl_post_prior_cumsum/sample_size)
            logs[datasetname]['recon_loss_encoded'].append(-obs_loglik_encoded_cumsum/sample_size)
            logs[datasetname]['kl_encoded_prior'].append(kl_encoded_prior_cumsum/sample_size)
            logs[datasetname]['l2_penalty'].append(l2_penalty.item())
        
        # save loss curves
        linestyles = ['solid', 'dashed']
        i = 0
        for dataset_name in ['train', 'val']:
            plt.plot(logs[dataset_name]['loss'][:], linestyle=linestyles[i],
                     label=dataset_name)
            i += 1
        if epoch == 1:
            plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('%s/loss_curves.pdf' % (result_path), dpi=600)
        
        # update models and logs if the best validation loss is updated
        current_val_loss = logs['val']['loss'][-1]
        print('epoch', epoch)
        best_val_loss = current_val_loss if epoch == 1 else np.minimum(best_val_loss, current_val_loss)
        print('current_val_loss', current_val_loss)
        print('best_val_loss', best_val_loss)
        if best_val_loss == current_val_loss:
            # update model and logs
            #print("yes")
            best_val_epoch = epoch
            os.makedirs('%s/' % result_path, exist_ok=True)
            torch.save({'prior': prior,
                        'encoder': encoder,
                        'decoder': decoder,
                        'logs': logs,
                        'num_epoch': num_epoch,
                        'batch_size': batch_size,
                        'num_worker': num_worker,
                        'seed': seed,
                        'beta': beta,
                        'Adam_beta1': Adam_beta1,
                        'Adam_beta2': Adam_beta2,
                        'weight_decay': weight_decay,
                        'init_lr': init_lr,
                        'lr_milestones': lr_milestones,
                        'lr_gamma': lr_gamma,
                        'dtype': dtype,
                        'M': M,
                        'alpha_step': alpha_step,
                        'fix_alpha': fix_alpha,
                        'result_path': result_path},
                       '%s/model.pth' % result_path)
        if epoch == num_epoch:
            # update logs
            saved_model = torch.load('%s/model.pth' % result_path)
            saved_model['logs'] = logs
            torch.save(saved_model, '%s/model.pth' % result_path)

        current_summary_stats_row = {}

        current_summary_stats_row['epoch'] = epoch
        current_summary_stats_row['best_val_epoch'] = best_val_epoch
        current_summary_stats_row['train_loss'] = logs['train']['loss'][-1]
        current_summary_stats_row['val_loss'] = logs['val']['loss'][-1]
        current_summary_stats_row['train_recon_loss_post'] = logs['train']['recon_loss_post'][-1]
        current_summary_stats_row['val_recon_loss_post'] = logs['val']['recon_loss_post'][-1]
        current_summary_stats_row['train_kl_post_prior'] = logs['train']['kl_post_prior'][-1]
        current_summary_stats_row['val_kl_post_prior'] = logs['val']['kl_post_prior'][-1]
        current_summary_stats_row['train_recon_loss_encoded'] = logs['train']['recon_loss_encoded'][-1]
        current_summary_stats_row['val_recon_loss_encoded'] = logs['val']['recon_loss_encoded'][-1]
        current_summary_stats_row['train_kl_encoded_prior'] = logs['train']['kl_encoded_prior'][-1]
        current_summary_stats_row['val_kl_encoded_prior'] = logs['val']['kl_encoded_prior'][-1]
        current_summary_stats_row['l2_penalty'] = logs['train']['l2_penalty'][-1]
        
        summary_stats.append(current_summary_stats_row)
        pd.DataFrame(summary_stats).to_csv('%s/summary_stats.csv' % result_path, index=False)  
    
    return None


class Prior_conti(nn.Module):
    def __init__(self, dim_z, dim_u, prior_node_list=[128,128]):
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
            
        self.mu_net.append(nn.Linear(self.prior_node_list[-1], self.dim_z))
        self.log_var_net.append(nn.Linear(self.prior_node_list[-1], self.dim_z))
        
    def forward(self, u_input):
        h_mu, h_log_var = self.mu_net[0](u_input), self.log_var_net[0](u_input)
        if len(self.mu_net) > 1:
            for i in range(len(self.mu_net)-1):
                h_mu = self.mu_net[i+1](h_mu)
                h_log_var = self.log_var_net[i+1](h_log_var)

        return h_mu, h_log_var
    

# receptive field of 10 samples
# For the model with receptive field of 10, a convolutional network with five time convolutional layers was used. The first layer had kernel size 2, the next three layers had kernel size 3 and used skip connections. The final layer had kernel size 3 and mapped the hidden dimensions to the output dimension.

# so input is of size (10, 120); output of (2,) each
# how does that get rid of the 10
class ConvEncoder(nn.Module):
    def __init__(self, dim_x, dim_z, time_window, hid_dim=60):
        super(CebraConvEncoder, self).__init__()
        self.dim_x, self.dim_z = dim_x, dim_z
        #self.hid_dim = hid_dim
        self.window = time_window # need?
        
        self.conv1 = nn.Conv1d(
            in_channels=dim_x, out_channels=dim_x, kernel_size=2, padding='same')
        self.conv2 = nn.Conv1d(
            in_channels=dim_x, out_channels=dim_x, kernel_size=3, padding='same')
        self.conv3 = nn.Conv1d(
            in_channels=dim_x, out_channels=dim_x, kernel_size=3, padding='same')
        
        self.conv4 = nn.Conv1d(
            in_channels=dim_x, out_channels=dim_x, kernel_size=3)#, padding='same')
        
        # have to decide if this will be a conv or linear layer
        # is it basically a linear with kernel size 1 lol
        self.conv_mu = nn.Conv1d(
            in_channels=dim_x, out_channels=dim_z, kernel_size=1, padding='same')
        self.conv_log_var = nn.Conv1d(
            in_channels=dim_x, out_channels=dim_z, kernel_size=1, padding='same')
        
        
        self.ff1 = nn.Linear(dim_x, hid_dim)
        self.ff2 = nn.Linear(hid_dim, hid_dim)
        self.ff3 = nn.Linear(hid_dim, hid_dim)
        self.ff4 = nn.Linear(hid_dim, hid_dim)
        
        # self.conv_mu = nn.Linear(hid_dim, dim_z)
        # self.conv_log_var
        
        # only use if not MSE
        #self.bn_mu = nn.BatchNorm1d(num_features=dim_z)
        #self.bn_log_var = nn.BatchNorm1d(num_features=dim_z)
        
        self.gelu =  nn.GELU()
    
    def forward(self, x):
        x = self.conv1(x)
        #print("shape after conv1", x.shape)
        
        ident = x
        
        x = self.conv2(x)
        #print("shape after conv2", x.shape)
        #x += ident # skip connections before or after?
        x = self.conv3(x)
        #print("shape after conv3", x.shape)
        #x += ident
        x = self.conv4(x)
        #print("shape after conv4", x.shape)
        #x += ident
            
        mu, log_var = self.conv_mu(x), self.conv_log_var(x) # not sure if correct
        
        # this unless MSE is used: need to check
        #mu = self.bn_mu(mu)
        #log_var = self.bn_log_var(log_var)
        
        return mu, log_var
        
        
class Decoder(nn.Module):
    def __init__(self, dim_z, dim_x=120, decoder_node_list=[4096, 4096], final_activation=None, time_window=10):
        super(Decoder, self).__init__()
        
        self.dim_z, self.dim_x = dim_z, dim_x
        self.decoder_node_list = decoder_node_list
        self.time_window = time_window
        
        self.main = nn.ModuleList()
        # input dimension is dim_z
        
        self.main.append(nn.Linear(self.dim_z, self.decoder_node_list[0]))
        self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        if len(self.decoder_node_list) > 1:
            for i in range(len(self.decoder_node_list)-1):
                self.main.append(nn.Linear(self.decoder_node_list[i],
                                          self.decoder_node_list[i+1]))
                self.main.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        # input dimension is gen_nodes
        # output needs to be (10,120)
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

        o = self.mu_net(h)
        
        return o