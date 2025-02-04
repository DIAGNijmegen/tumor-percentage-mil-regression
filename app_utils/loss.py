import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang

class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
    
    def forward(self, attention_scores, is_tumorous):
        return 
        '''N = attention_scores.size(0)
        log_N = torch.log(torch.tensor(N, dtype=attention_scores.dtype))

        if is_tumorous.sum() == 0:
            entropy_loss = -(1/log_N) * (attention_scores*torch.log(attention_scores)).sum()
            return entropy_loss
        else:
            tumorous_indices = torch.where(is_tumorous == 1)[0]
            non_tumorous_indices = torch.where(is_tumorous == 0)[0]

            m = tumorous_indices.size(0)
            s = non_tumorous_indices.size(0)
            m = torch.tensor(m, dtype=attention_scores.dtype)

            # Term 1: Sum of attention scores for non-tumorous patches
            term1 = attention_scores[non_tumorous_indices].sum()

            # Term 2: Sum of weighted attention scores for tumorous patches
            term2 = (1/torch.log(m)) * (attention_scores[tumorous_indices] * torch.log(attention_scores[tumorous_indices])).sum()

            # Term 3: Sum of attention scores for tumorous patches
            term3 = attention_scores[tumorous_indices].sum()

            # Calculate the loss
            attention_loss = term1 + term2 - term3

            return attention_loss'''


class KLDLoss(nn.Module):
    """KL-divergence loss between attention weight and uniform distribution"""
    
    def __init__(self):
        super(KLDLoss, self).__init__()
        
    def forward(self, attn_val, cluster):
        """
        Example:
          Input - attention value = torch.tensor([0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.05, 
                                0.1, 0.05, 0.1, 0.05, 0.1, 0.05])
                  cluster = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
          Output - 0.0043
        """
        kld_loss = 0
        is_cuda = attn_val.device
        cluster = np.array(cluster)
        for cls in np.unique(cluster):
            index = np.where(cluster==cls)[0]
            # HARD CODE - if number of images less than 4 in a cluster then skip
            if len(index)<=4:
                continue            
            kld_loss += F.kl_div(F.log_softmax(attn_val[index], dim=0)[None],\
                            torch.ones(len(index), 1)[None].to(is_cuda)/len(index),\
                            reduction = 'batchmean')
            
        return kld_loss


def get_lds_kernel_window(kernel, ks, sigma):
    
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def get_bin_idx(x, bins):
    #TODO: find optimal binning strategy
        '''
        x is a continuous variable (normalised) between 0-1
        to get the bins, x is rounded to its nearest decimal,
        and then multiplied by ten. Totalling in 11 bins which
        will be weighed accordingly to its frequency within
        '''
        label = None
        for i, bin in enumerate(bins):
            if x <= bin:
                label = i
                break


        return label


def weighting_continuous_values(labels) -> torch.FloatTensor:
    

    edges = np.histogram_bin_edges(labels, bins='auto')
    bin_index_per_label = np.array([get_bin_idx(label, edges) for label in labels])
    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1

    #i.e., np.histogram(bin_index_per_label)
    unique, counts = np.unique(bin_index_per_label, return_counts=True)
    num_samples_of_bins = dict(zip(unique, counts))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = torch.FloatTensor([np.float32(1 / x) for x in eff_num_per_label])

    return weights


class WeightedMSELoss(nn.Module):
    def __init__(self, weights, **kwargs): #weights=None,
        super().__init__()
        self.weights = weights
        print("The weights are: ", self.weights)

    def forward(self, inputs, targets): #added weights as targets[1]
        # weights = self.weights
        #breakpoint()
        weights = self.weights + 1
        weights = torch.from_numpy(np.array(weights)).float().to('cuda')
        loss = (inputs - targets) ** 2
        if weights is not None:
            loss = loss*weights #remove squeeze()
        loss = torch.mean(loss)
        return loss