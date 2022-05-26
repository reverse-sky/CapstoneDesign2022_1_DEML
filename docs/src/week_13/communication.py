import torch
import numpy as np
from torch import nn
import scipy

class communication():
    def __init__(self, comm_parameters, device = 'cpu'):
        
        self.mod_type = comm_parameters['mod_type']
        self.mod_order = comm_parameters['mod_order']
        self.device = device
        
        self.bits_per_symb = int(np.log2(self.mod_order))
        self.get_constellation()
        self.get_bitmap()
        
        
    def get_constellation(self):

        if self.mod_type == 'bpsk':
            self.constellation = torch.tensor([[-1],[1]], dtype = torch.float).to(self.device)
            return self.constellation
        elif self.mod_type == 'qam':
            ptfile = torch.load('constellations/qam.pt')
            string = 'qam_'+str(self.mod_order)
            self.constellation = torch.tensor(ptfile[string], dtype = torch.float).to(self.device)
            return self.constellation

    def get_bitmap(self):
        if self.mod_type == 'bpsk':
            self.bitmap = torch.tensor([[0],[1]], dtype = torch.float).to(self.device)
            return self.bitmap

        elif self.mod_type == 'qam':    
            ptfile = torch.load('constellations/qam.pt')
            string = 'bitmap_'+str(self.mod_order)+'QAM'
            self.bitmap = torch.tensor(ptfile[string], dtype = torch.float).to(self.device)
            return self.bitmap


    # Dimension setting is ( numsamp, seqence_length, 2 [real and complex dim] )
    def modulate(self, cw, complex_format = False):
        bits_per_sym = int(np.log2(self.mod_order))
        num_samp = cw.shape[0]
        cw = cw.reshape(-1, bits_per_sym)
        ind = bin2dec(cw).type(torch.long)
        mod_sig = self.constellation[ind]
        
        if complex_format:
            return torch.view_as_complex(mod_sig.reshape(num_samp, -1, 2))
        else:
            return mod_sig.reshape(num_samp, -1, 2)
    
    def AWGN(self, x, snr):
        if self.mod_type == 'bpsk':
            return x+torch.randn_like(x)/torch.sqrt(snr).to(self.device)
        
        elif self.mod_type == 'qam': 
            return x+torch.randn_like(x)/torch.sqrt(2.0*snr).to(self.device)
        
    def mod_bpsk(self, C):
        X = -(2.0*C-1.0)
        return X
    
    
    def demodulate(self, y, snr):
        num_samp = y.shape[0]
        cc_length = y.shape[1]
        
        y_reshape = y.reshape(-1,2)
        pdf = cpdf_conditional_premargin(y_reshape, self.constellation, 1/snr)
        
        flat_num_samp = pdf.shape[0]
        
        fb_1 = torch.empty(flat_num_samp, self.bits_per_symb).to(self.device)
        fb_0 = torch.empty(flat_num_samp, self.bits_per_symb).to(self.device)

        for bit_indx in np.arange(self.bits_per_symb):
            fb_1[:, bit_indx] = torch.sum(pdf[:,self.bitmap[:,bit_indx]==1.0], dim = 1)
            fb_0[:, bit_indx] = torch.sum(pdf[:,self.bitmap[:,bit_indx]==0.0], dim = 1)
    
        LLR = torch.clamp(torch.log(fb_0) - torch.log(fb_1), -30.0, 30.0)
        #pb_1 = 1/(1+torch.exp(-LLR))
        #return torch.clamp(pb_1,0.0,1.0)
        
        return LLR.reshape(num_samp, -1)
    
    
        
def gen_random_bits(num_samp, seq_length, device='cpu'):
    return torch.randint(0,2, [num_samp, seq_length],dtype = torch.float, device = device)


def dec2bin(x, bits):
    # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    x = x.type(torch.int)
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    #return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    return x.bitwise_and(mask).ne(0).float()


def bin2dec(b):
    bits = b.shape[1]
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1).reshape(-1,1)


def cpdf_conditional_premargin(y, x, var):
    real = (y[:,0].reshape(-1,1) - x[:,0])
    imag = (y[:,1].reshape(-1,1) - x[:,1])
    pdf = 1/(np.pi*var) * torch.exp(-(real**2+imag**2)/var)
    return pdf

# pdf for p_{Y|X} (sample base)
def cpdf_conditional(y, x, m,  var):
    pdf = 1/(np.pi*var) * torch.exp(-torch.norm(y-x, dim = 1)**2/var)
    #nf = []
    #for ind in range(m):
    #    nf.append(torch.sum(pdf[index==ind]).item())
    #for ind in range(m):
    #    pdf[index==ind] = pdf[index==ind]/nf[ind]
    return pdf

# complex pdf for p_Y
def cpdf_marginal(y, cb, var):
    cond_pdf = cpdf_conditional_premargin(y, cb, var)
    #nf = torch.sum(cond_pdf, dim=0).numpy()
    #cond_pdf = cond_pdf/nf
    py = torch.sum(cond_pdf, dim=1)/cb.shape[0]
    return cond_pdf, py

