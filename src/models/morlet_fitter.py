import sys
sys.path.append('/Users/xz498/Desktop/ultrasound project/data analysis/ultrasonic_ml/src')
sys.path.append('/Users/xz498/Desktop/ultrasound project/data analysis/M3Learning-Util/src')
sys.path.append('/Users/xz498/Desktop/ultrasound project/data analysis/AutoPhysLearn/src')
import os

from random import shuffle
from m3util.util.IO import make_folder
from m3util.ml.regularization import LN_loss, ContrastiveLoss, DivergenceLoss, Sparse_Max_Loss

from autophyslearn.spectroscopic.nn import Multiscale1DFitter, Conv_Block, FC_Block, block_factory


import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from tqdm import tqdm
import wandb
import numpy as np
import h5py

class morlet_1D_fitters_real():
    def __init__(self, limits=[1,1,975], device='cpu'):
        self.limits = limits
        self.omega = 4.5e-03# 2.25 MHz # TODO: make this an init parameter
    
    def scale_parameters(self, embedding):
        a = self.limits[0] * embedding[..., 0] # amplitude
        mu = self.limits[1] * embedding[..., 1] # mean
        sigma = self.limits[2] * embedding[..., 2] # standard deviation
        omega = (self.limits[3] * embedding[..., 3] + 1) * self.omega # +-1% of the angular frequency
        
        return torch.stack([a,mu,sigma,omega],axis=2)

    def apply_activations(self, embedding):
        '''This function takes an embedding and scales it to the limits of the parameters
        
        This function implements the Pseudo-Voigt profile as described in: (but not exactly anymore)
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330705/
        
        Args:
            embedding (torch.Tensor): Tensor of shape (batch_size, num_fits, 4) containing:
                - A: Area under curve (index 0)
             - x: Mean position (index 1)
                - w: Full Width at Half Maximum (FWHM) (index 2)
                - nu: Lorentzian character fraction (index 3)
            limits (list): Scale factors for [A, x, w]. Defaults to [1, 1, 975]
        '''
        a = nn.Tanh()(embedding[..., 0])/2 + 0.5 # amplitude [0,1]
        mu = torch.clamp(nn.Tanh()(embedding[..., 1])/2 + 0.5, min=1e-10) # mean [1e-10,1]
        sigma = torch.clamp(nn.Tanh()(embedding[..., 2])/2 + 0.5, min=0.1) # stdv [0.1,1]
        omega = nn.Tanh()(embedding[..., 3]) # angular freq [-1,1]
        
        return torch.stack([a,mu,sigma,omega],axis=2)
    
    def generate_fit(self, embedding, spec_len, **kwargs, ):
        '''Generate 1D Morlet profiles from embedding parameters.
        # H2O: 1.5 MRayl (specific acoustic impedance), 1500 m/s -> TT= 13,333 ns
        impedance, loss coefficient, and travel time of the layer
        mode (str) : 'echo', 'transmission', 'both' - the acoustic signal type to generate

        not compatible with cwt: pi**-0.25 * (exp(1j*w*(x - mu)) - exp(-0.5*(w**2))) * exp(-0.5*(x - mu)**2)
        compatible with cwt: exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)

        Args:
            embedding (torch.Tensor): Tensor of shape (batch_size, num_fits, 4) containing:
                - a: Amplitude (index 0)
                - mu: Center frequency (index 1)
                - sigma: Standard deviation (index 2)
                - omega: Angular frequency fraction deviation from 2.25 MHz (index 3)
            spec_len (int): Length of the spectrum
        '''
        device = embedding.device
        # Unpack embedding tensor along last dimension (shape: [..., 4])
        a = embedding[..., 0].unsqueeze(-1)  # amplitude
        mu = embedding[..., 1].unsqueeze(-1)  # center frequency
        sigma = embedding[..., 2].unsqueeze(-1)  # standard deviation of gaussian window
        omega = embedding[..., 3].unsqueeze(-1)  # angular frequency fraction deviation from 2.25 MHz
        s = a.shape  # (_, num_fits)

        t = torch.arange(spec_len, dtype=torch.float32).repeat(s[0],s[1],1).to(device)

        # Calculate Morlet profile
        morlet = a * torch.exp(-0.5 * ((t - mu) / sigma)**2) * torch.cos(2 * np.pi * omega * (t-mu))
        
        return morlet.to(torch.float32)


class morlet_1D_fitters_complex():
    def __init__(self, limits=[1,1,975], device='cpu'):
        self.limits = limits
        self.omega = 4.5e-03# 2.25 MHz # TODO: make this an init parameter
    
    def scale_parameters(self, embedding):
        a = self.limits[0] * embedding[..., 0] # amplitude
        mu = self.limits[1] * embedding[..., 1] # mean
        sigma = self.limits[2] * embedding[..., 2] # standard deviation
        omega = (self.limits[3] * embedding[..., 3] + 1) * self.omega # +-1% of the angular frequency
        
        return torch.stack([a,mu,sigma,omega],axis=2)

    def apply_activations(self, embedding):
        '''This function takes an embedding and scales it to the limits of the parameters
        
        This function implements the Pseudo-Voigt profile as described in: (but not exactly anymore)
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330705/
        
        Args:
            embedding (torch.Tensor): Tensor of shape (batch_size, num_fits, 4) containing:
                - A: Area under curve (index 0)
             - x: Mean position (index 1)
                - w: Full Width at Half Maximum (FWHM) (index 2)
                - nu: Lorentzian character fraction (index 3)
            limits (list): Scale factors for [A, x, w]. Defaults to [1, 1, 975]
        '''
        a = nn.Tanh()(embedding[..., 0])/2 + 0.5 # amplitude [0,1]
        mu = torch.clamp(nn.Tanh()(embedding[..., 1])/2 + 0.5, min=1e-10) # mean [1e-10,1]
        sigma = torch.clamp(nn.Tanh()(embedding[..., 2])/2 + 0.5, min=0.1) # stdv [0.1,1]
        omega = nn.Tanh()(embedding[..., 3]) # angular freq [-1,1]
        
        return torch.stack([a,mu,sigma,omega],axis=2)
    
    def generate_fit(self, embedding, spec_len, **kwargs, ):
        '''Generate 1D Morlet profiles from embedding parameters.
        # H2O: 1.5 MRayl (specific acoustic impedance), 1500 m/s -> TT= 13,333 ns
        impedance, loss coefficient, and travel time of the layer
        mode (str) : 'echo', 'transmission', 'both' - the acoustic signal type to generate

        not compatible with cwt: pi**-0.25 * (exp(1j*w*(x - mu)) - exp(-0.5*(w**2))) * exp(-0.5*(x - mu)**2)
        compatible with cwt: exp(1j*w*x/s) * exp(-0.5*(x/s)**2) * pi**(-0.25) * sqrt(1/s)

        Args:
            embedding (torch.Tensor): Tensor of shape (batch_size, num_fits, 4) containing:
                - a: Amplitude (index 0)
                - mu: Center frequency (index 1)
                - sigma: Standard deviation (index 2)
                - omega: Angular frequency fraction deviation from 2.25 MHz (index 3)
            spec_len (int): Length of the spectrum
        '''
        device = embedding.device
        # Unpack embedding tensor along last dimension (shape: [..., 4])
        a = embedding[..., 0].unsqueeze(-1)  # amplitude
        mu = embedding[..., 1].unsqueeze(-1)  # center frequency
        sigma = embedding[..., 2].unsqueeze(-1)  # standard deviation of gaussian window
        omega = embedding[..., 3].unsqueeze(-1)  # angular frequency fraction deviation from 2.25 MHz
        s = a.shape  # (_, num_fits)

        t = torch.arange(spec_len, dtype=torch.float32).repeat(s[0],s[1],1).to(device)

        # Calculate Morlet profile
        morlet = a * torch.exp(-0.5 * ((t - mu) / sigma)**2) * torch.cos(2 * np.pi * omega * (t-mu))
        
        return morlet.to(torch.float32)


class gaussian_1D_fitters():# TODO: read through math in these (esp art 2) before modifying model
    '''https://pubs.acs.org/doi/pdf/10.1021/acssensors.1c00787?ref=article_openPDF
        https://psl.noaa.gov/people/gilbert.p.compo/Torrence_compo1998.pdf'''
    def __init__(self, limits=[1,1,975], ):
        self.limits = limits
    
    def scale_parameters(self, embedding):
        a = self.limits[0] * embedding[..., 0] # amplitude
        mu = self.limits[1] * embedding[..., 1] # mean
        sigma = self.limits[2] * embedding[..., 2] # standard deviation
        
        return torch.stack([a,mu,sigma],axis=2)
    
    def apply_activations(self, embedding):
        '''This function takes an embedding and scales it to the limits of the parameters
        
        This function implements the Pseudo-Voigt profile as described in:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9330705/
        
        Args:
            embedding (torch.Tensor): Tensor of shape (batch_size, num_fits, 4) containing:
                - A: Area under curve (index 0)
                - x: Mean position (index 1)
                - sigma: Standard deviation (index 2)
            limits (list): Scale factors for [a, mu, sigma]. Defaults to [1, 1, 975]
        '''
        a = nn.ReLU()(embedding[..., 0]) # amplitude 
        mu = torch.clamp(nn.Tanh()(embedding[..., 1])/2 + 0.5, min=1e-3) # mean
        sigma = torch.clamp(nn.Tanh()(embedding[..., 2])/2 + 0.5, min=1e-3) # standard deviation
        return torch.stack([a,mu,sigma],axis=2)

    def generate_fit(self, embedding, spec_len, **kwargs, ):
        """Calculate the Gaussian component of the Pseudo-Voigt profile
        
        Args:
            A (torch.Tensor): Area under curve
            x (torch.Tensor): Mean positions
            w (torch.Tensor): Full Width at Half Maximum (FWHM)
        """
        t = torch.arange(spec_len, dtype=torch.float32).repeat(s[0],s[1],1).to(embedding.device)

        gaussian = embedding[..., 0] * torch.exp( -4 * torch.log(torch.tensor(2)) \
                                     * ((t-embedding[..., 1])/embedding[..., 2])**2 )
        return gaussian.to(torch.float32)
    

class Fitter_AE:
    """Autoencoder-based fitter for spectroscopic data.

    This class implements an autoencoder architecture for fitting spectroscopic data,
    particularly designed for Pseudo-Voigt profiles.

    Args:
        function (callable): Function to generate profiles from embeddings
        dset (Dataset): Dataset containing spectroscopic data
        num_params (int): Number of parameters in the embedding
        num_fits (int): Number of profiles to fit simultaneously
        limits (list): Scale factors for the profile parameters
        learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-3
        device (str, optional): Device to run computations on. Defaults to 'cuda:0'
        encoder (class, optional): Encoder architecture class. Defaults to Multiscale1DFitter
        encoder_params (dict, optional): Parameters for the encoder architecture

    Attributes:
        dset: The input dataset
        checkpoints_label: Label for the checkpoints
        num_fits: Number of profiles to fit
        limits: Scale factors for parameters
        device: Computation device
        learning_rate: Optimizer learning rate
        encoder: The encoder model
        optimizer: Adam optimizer
        best_train_loss: Best training loss achieved
        folder: Directory for saving checkpoints
    """
    def __init__(self,
                 function, 
                 dset,
                 num_params,
                 num_fits,
                 input_channels,
                 checkpoints_label='',
                 learning_rate=3e-5,
                 device='cuda:0',
                 encoder = Multiscale1DFitter,
                 encoder_params = { "model_block_dict": { # factory wrapper for blocks
                    "hidden_x1": block_factory(Conv_Block)(output_channels_list=[8,6,4], 
                                                           kernel_size_list=[7,7,5], 
                                                           pool_list=[64], 
                                                           max_pool=False),
                    "hidden_xfc": block_factory(FC_Block)(output_size_list=[64,32,20]),
                    "hidden_x2": block_factory(Conv_Block)(output_channels_list=[4,4,4,4,4,4], 
                                                           kernel_size_list=[5,5,5,5,5,5], 
                                                           pool_list=[16,8,4], 
                                                           max_pool=True),
                    "hidden_embedding": block_factory(FC_Block)(output_size_list=[16,8,4])
                },
                "skip_connections": {"hidden_xfc": "hidden_embedding"} },
            ):
        self.dset = dset
        self.num_fits = num_fits
        self.num_params = num_params
        self.device = device
        self.learning_rate = learning_rate
        self.encoder_params = encoder_params
        self.encoder = encoder( function = function,
                                x_data = dset,
                                input_channels = input_channels,
                                num_fits = num_fits,
                                num_params = num_params,
                                device=device,
                                **encoder_params
                                ).to(self.device).type(torch.float32)
        self.optimizer = optim.Adam( self.encoder.parameters(), lr=self.learning_rate )
        self.lr_scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=100, eta_min=self.learning_rate/100)
        self.configure_dataloader()
        
        self.start_epoch = 0
        self.best_train_loss = float('inf')
        self.checkpoint = None
        self.scheduler = None
        self._checkpoint_folder = os.path.split(dset.dataset_path)[0] + '/' + checkpoints_label + f'/checkpoints/{dset.dset_name}' # TODO: change to dset.dataset_path
        self.embedding_h5_name = os.path.split(dset.dataset_path)[0] + '/' + checkpoints_label + f'/{dset.dset_name}/embeddings.h5'
        
    @property
    def dataloader(self): return self._dataloader
    def configure_dataloader(self, **kwargs):
        '''Set the dataloader for the fitter using random sampling
        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            shuffle (bool, optional): Set to True to have the data reshuffled at every epoch. Defaults to True.
        '''
        batch_size = kwargs.get('batch_size', 32)
        shuffle = kwargs.get('shuffle', True)
        
        # builds the dataloader with random sampling
        self._dataloader = DataLoader(self.dset, batch_size=batch_size, shuffle=shuffle)
        
    @property
    def checkpoint_folder(self): return self._checkpoint_folder
    @property 
    def checkpoint_file(self): return self._checkpoint_file
    @property
    def check(self): return self._check
    
    @property
    def checkpoint(self): return self._checkpoint
    @checkpoint.setter
    def checkpoint(self, value):
        self._checkpoint = value
        try:
            checkpoint_folder,checkpoint_file = os.path.split(self._checkpoint)
            self._checkpoint_file = checkpoint_file
            self._check = checkpoint_file.split('.pkl')[0]
            self._checkpoint_folder = checkpoint_folder
            self.embedding_h5_name = '/'.join(self._checkpoint_folder.split('/')[:-2]) + '/embeddings.h5'
        except:
            self._check = None
            self._checkpoint_folder = None
            self._checkpoint_file = None
            self.embedding_h5_name = None
            
    def train(self, seed=42, epochs=100, 
              save_every=1, batch_size=100, return_losses=False, 
              log_wandb=False, primary_loss_function=F.mse_loss, 
              lr_scheduling=True, **kwargs):
        """Train the model.
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 42
            epochs (int): Number of training epochs. Defaults to 100
            save_every (int, optional): Save checkpoint every n epochs. Defaults to 1
            batch_size (int, optional): Batch size for training. Defaults to 100
            return_losses (bool, optional): Whether to return losses. Defaults to False
            log_wandb (bool, optional): Whether to log to wandb. Defaults to False
            primary_loss_function (callable, optional): Primary loss function. Defaults to F.mse_loss
            lr_scheduling (bool, optional): Whether to use learning rate scheduling. Defaults to True
        """
        make_folder(self.checkpoint_folder)
        print(os.path.abspath(self.checkpoint_folder))

        # set seed
        torch.manual_seed(seed)
        
        # Configure dataloader with random sampling
        self.configure_dataloader(shuffle=True, batch_size=batch_size)
        
        if lr_scheduling: self.lr_scheduler.max_steps = epochs

        # training loop
        for epoch in range(self.start_epoch, epochs):

            loss_dict = self.loss_function( self.dataloader,
                                           primary_loss_function=primary_loss_function, **kwargs)
            
            # divide by batches inplace
            loss_dict.update( (k,v/len(self.dataloader)) for k,v in loss_dict.items())
            
            print(
                f'Epoch: {epoch:03d}/{epochs:03d} | Train Loss: {loss_dict["train_loss"]:.4f}')
            print('.............................')
            if log_wandb: wandb.log(loss_dict)
          #  schedular.step()
          # TODO: add regularization losses
          # TODO: add embedding saver
            if epoch % save_every == 0: self.save_checkpoint(epoch, loss_dict=loss_dict,)
            if lr_scheduling: self.lr_scheduler.step()
        if return_losses: return loss_dict
        
    def save_checkpoint(self,epoch,loss_dict,**kwargs): 
        """Save the checkpoint"""
        today = datetime.today()
        save_date=today.strftime('(%Y-%m-%d, %H:%M:%S)')
        lr_ = format(self.optimizer.param_groups[0]['lr'], '.5f')
        self.checkpoint = self.checkpoint_folder + f'/{save_date}_epoch:{epoch:04d}_lr:{lr_}_trainloss:{loss_dict["train_loss"]:.4f}.pkl'
        
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss_dict': loss_dict,
            'loss_params': kwargs,
        }
        torch.save(checkpoint, self.checkpoint)

    def load_weights(self, path_checkpoint): # TODO: make a quickload feature so weights are not imported unless claculations will be done
        """loads the weights from a checkpoint

        Args:
            path_checkpoint (str): path where checkpoints are saved 
        """
        self.checkpoint = path_checkpoint
        checkpoint = torch.load(path_checkpoint, weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch']
        
        try: self.loss_dict = checkpoint['loss_dict']
        except: self.loss_dict = None
        
        try: self.loss_params = checkpoint['loss_params']
        except: self.loss_params = None
    
    # Loss stuff
    def _initialize_loss_components(self, train_iterator, 
                                    coef1=0, coef2=0, coef3=0, coef4=0, 
                                    primary_loss=F.mse_loss):
        """Initialize loss components and their coefficients
        Args:
            train_iterator: DataLoader for training data
            coef1: Coefficient for weighted LN loss
            coef2: Coefficient for contrastive loss
            coef3: Coefficient for divergence loss
            coef4: Coefficient for sparse max loss
            primary_loss: Loss function for primary loss
        """
        components = {
            'primary': (primary_loss),
            'weighted_ln': (LN_loss(coef=coef1,).to(self.device) if coef1 > 0 else None),
            'contrastive': (ContrastiveLoss(coef2).to(self.device) if coef2 > 0 else None),
            'divergence': (DivergenceLoss(train_iterator.batch_size, coef3).to(self.device) if coef3 > 0 else None),
            'sparse_max': (Sparse_Max_Loss(min_threshold=self.learning_rate, channels=self.num_fits, coef=coef4).to(self.device) if coef4 > 0 else None)
        }
        return components


    def _compute_losses(self, embedding, x, predicted_x, loss_components, coef5):
        """Compute all loss components"""
        loss_dict = {
            'reg_loss_1': 0, 'primary_loss': 0, 'mae_loss': 0, 'train_loss': 0,
            'sparse_max_loss': 0, 'l2_batchwise_loss': 0,
        }
        
        # Compute individual losses
        losses = {
            'reg_loss_1': loss_components['weighted_ln'](embedding[:,:,0]) if loss_components['weighted_ln'] else 0,
            'contras_loss': loss_components['contrastive'](embedding[:,:,0]) if loss_components['contrastive'] else 0,
            'maxi_loss': loss_components['divergence'](embedding[:,:,0]) if loss_components['divergence'] else 0,
            'sparse_max_loss': loss_components['sparse_max'](embedding[:,:,0]) if loss_components['sparse_max'] else 0,
        }
        
        # L2 batchwise loss
        if coef5 > 0:
            losses['l2_loss'] = coef5 * ((embedding[:,:,1]/embedding[:,:,2]).max(dim=0).values - 
                                        (embedding[:,:,1]/embedding[:,:,2]).min(dim=0).values).mean()
        else:
            losses['l2_loss'] = 0
        
        # MSE loss
        primary_loss = loss_components['primary'](x, predicted_x, reduction='mean')

        # Update loss dictionary
        loss_dict.update({k: v for k, v in losses.items() if v != 0})
        loss_dict['primary_loss'] = primary_loss.item()
        
        # Compute total loss
        total_loss = primary_loss + losses['reg_loss_1'] + losses['contras_loss'] - losses['maxi_loss'] + losses['l2_loss']
        loss_dict['train_loss'] = total_loss.item()
        
        return total_loss, loss_dict

    def loss_function(self, train_iterator, coef1=0, coef2=0, coef3=0, coef4=0, coef5=0, primary_loss_function=F.mse_loss,
                     ln_parm=1, beta=None, fill_embeddings=False, minibatch_logging_rate=None):
        """Calculate the loss for training.

        Combines multiple loss components:
        - MSE loss between input and reconstructed spectra
        - Weighted LN loss for regularization
        - Contrastive loss for embedding space structure
        - Divergence loss for embedding distribution
        - Sparse max loss for sparsity
        - L2 batchwise loss for parameter consistency

        Args:
            train_iterator: DataLoader for training data
            coef1-5 (float): Coefficients for different loss components
            ln_parm (float): Parameter for weighted LN loss
            beta (float, optional): Parameter for variational loss
            fill_embeddings (bool): Whether to store embeddings
            minibatch_logging_rate (int, optional): Logging frequency
            primary_loss_function (callable): Primary loss function

        Returns:
            dict: Dictionary containing different loss components and total loss
        """
        self.encoder.train()
        loss_components = self._initialize_loss_components(train_iterator, coef1, coef2, coef3, coef4, primary_loss=primary_loss_function)
        accumulated_loss_dict = {'reg_loss_1': 0, 'mse_loss': 0, 'train_loss': 0,
                               'sparse_max_loss': 0, 'l2_batchwise_loss': 0}

        for i, batch in enumerate(tqdm(train_iterator, leave=True, total=len(train_iterator))):
            # Handle different batch formats: (idx, x) or just x
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                idx, x = batch
                idx = idx.to(self.device).squeeze() if isinstance(idx, torch.Tensor) else None
            else:
                x = batch
                idx = None
            
            # Ensure x is a tensor and move to device
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.to(self.device, dtype=torch.float32)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if beta is None:
                predicted_x, embedding = self.encoder(x)
            else:
                predicted_x, embedding, sd, mn = self.encoder(x, beta)
            
            # Compute losses
            loss, batch_loss_dict = self._compute_losses(embedding, x, predicted_x.sum(axis=1), loss_components, coef5)

            # Update accumulated losses
            for k in accumulated_loss_dict:
                if k in batch_loss_dict:
                    accumulated_loss_dict[k] += batch_loss_dict[k]
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            
            # Handle embeddings and logging
            if fill_embeddings and idx is not None:
                sorted_idx, indices = torch.sort(idx)
                self.embedding[sorted_idx.detach().numpy()] = embedding[indices].cpu().detach().numpy()
                
            if minibatch_logging_rate and i % minibatch_logging_rate == 0:
                wandb.log({k: v/(i+1) for k, v in accumulated_loss_dict.items()})

        return accumulated_loss_dict
    
    def performance_metrics(self, dset_name): # TODO: mse, r^2 
        pass
    