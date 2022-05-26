import torch
import numpy as np
from torch import nn
import scipy


class nrLDPC():
    def __init__(self, code_parameters, device = 'cpu'):
        

            
        self.Zc = code_parameters['Zc']
        self.BG = code_parameters['BG']
        self.max_iter = code_parameters['max_iter']
        self.channel_code_rate = code_parameters['channel_code_rate']
        self.device = device
        self.decode_format = code_parameters['Decode_format']
              
        if self.BG == 1:
            ef = 22
        elif self.BG == 2:
            ef = 10
        else:
            assert False, "Base graph does not exist. BG can be either 1 or 2"
        
        # Can easily generalize
        if self.channel_code_rate == '1/3':
            self.cc_n_punctured = 66*self.Zc
        elif self.channel_code_rate == '1/2':
            self.cc_n_punctured = 44*self.Zc
        elif self.channel_code_rate == '2/3':
            self.cc_n_punctured = 33*self.Zc
        elif self.channel_code_rate == '8/9':
            self.cc_n_punctured = 24.75*self.Zc
        else:
            assert False, "Supported rates 1/3, 1/2, 2/3, 8/9"
            
        self.cc_k = 22*self.Zc
        self.cc_n = int(self.cc_n_punctured+2*self.Zc) 
        
        
        self.get_H(self.Zc, self.BG)
        
        
        if self.channel_code_rate != '1/3':
            self.H = resize_sparse(self.H, self.cc_n-self.cc_k, self.cc_n, self.device)
            
        
        #self.H = self.H[0:self.cc_k, 0:self.cc_n]    
        #self.H.sparse_resize_and_clear_((self.cc_k, self.cc_n))
        
              
        self.n_Hrows = int(ef * self.Zc)
        self.n_Hcols = self.H.shape[1]
        self.eps = 1e-4
        
        
    def get_H(self, zc, bg):
        file = './nrldpc_H/H_BG'+str(bg)+'_Zc'+str(zc)+'.pt'
        H_ind = torch.from_numpy(torch.load(file)['H_ind'].astype('int16'))
        A_ind = torch.from_numpy(torch.load(file)['A_ind'].astype('int16'))
        Binv_ind = torch.from_numpy(torch.load(file)['Binv_ind'].astype('int16'))
        C_ind = torch.from_numpy(torch.load(file)['C_ind'].astype('int16'))
        D_ind = torch.from_numpy(torch.load(file)['D_ind'].astype('int16'))
        
        H_shape = torch.from_numpy(torch.load(file)['H_shape'].astype('int16'))
        A_shape = torch.from_numpy(torch.load(file)['A_shape'].astype('int16'))
        Binv_shape = torch.from_numpy(torch.load(file)['Binv_shape'].astype('int16'))
        C_shape = torch.from_numpy(torch.load(file)['C_shape'].astype('int16'))
        D_shape = torch.from_numpy(torch.load(file)['D_shape'].astype('int16'))
               
        H_shape = tuple(H_shape[0])
        A_shape = tuple(A_shape[0])
        Binv_shape = tuple(Binv_shape[0])
        C_shape = tuple(C_shape[0])
        D_shape = tuple(D_shape[0])

        self.H = matlabsparse_2_coo(H_ind, H_shape).to(self.device)
        self.A = matlabsparse_2_coo(A_ind, A_shape).to(self.device)
        self.Binv = matlabsparse_2_coo(Binv_ind, Binv_shape).to(self.device)
        self.C = matlabsparse_2_coo(C_ind, C_shape).to(self.device)
        self.D = matlabsparse_2_coo(D_ind, D_shape).to(self.device)
        
        return self.H
      
    
    def nrLDPC_encode(self, data_batch, puncture = True):
        
        data_batch = data_batch.T
        data_batch = data_batch.type(torch.float)
        p_1 = mod(torch.mm(torch.sparse.mm(self.Binv, self.A), data_batch))
        p_2 = mod(torch.mm(self.C, data_batch)+torch.mm(self.D, p_1))
        
        cw = torch.vstack([data_batch, p_1, p_2]).T
        
        if puncture == False:
            return cw[:, 0:self.cc_n]
        else:
            return cw[:, int(2*self.Zc):self.cc_n]
            
    
    
    def LDPC_decode_v0(self, llr, punctured = True):
        
        if punctured == True:
            llr = torch.hstack([torch.zeros(llr.shape[0], int(2*self.Zc)).to(self.device), llr])
        
        # LLR inputs in the form of ln(p0/p1)
        
        Hind = self.H.indices()
                
        # Initialization
        f0 = torch.sigmoid(llr)
        f1 = 1-f0
        
        sq0 = f0[:,Hind[1]]
        sq1 = f1[:,Hind[1]]
        
        sff0 = sq0
        sff1 = sq1
        
        tanhllr = torch.tanh(0.5*llr)
        
        k=0
        success = 0
        
        while ((success == 0) and (k<self.max_iter)):
            k += 1
            # Horizontal Step
            sdq = sq0 - sq1 
            
            sdq[torch.abs(sdq)<=1e-12] = 1e-12
            
            Pdq_v = multiply_reduce_dim1(self.H, sdq, self.device)
            Pdq = Pdq_v[:, Hind[0]]

            sr0 = 0.5*(1+Pdq/sdq)
            sr0[torch.abs(sr0)<=1e-12] = 1e-12
            sr1 = 0.5*(1-Pdq/sdq)
            sr1[torch.abs(sr1)<=1e-12] = 1e-12

            # Vertical Step
            Pr0_v = multiply_reduce_dim0(self.H, sr0, self.device)
            sPr0 = Pr0_v[:, Hind[1]]
            Q0 = Pr0_v*f0
            #Q0 = sum_reduce_dim0(self.H, sPr0*sff0, self.device)
            sq0 = sPr0*sff0/sr0
            
            Pr1_v = multiply_reduce_dim0(self.H, sr1, self.device)
            sPr1 = Pr1_v[:, Hind[1]]
            Q1 = Pr1_v*f1
            #Q1 = sum_reduce_dim0(self.H, sPr1*sff1, self.device)
            sq1 = sPr1*sff1/sr1
            
            sqq = sq0+sq1
            sq0 = sq0/sqq
            sq1 = sq1/sqq
            
            # tentative decoding
            QQ = Q0+Q1
            prob = Q1/QQ
            Q0 = Q0/QQ
            Q1 = Q1/QQ

            tent = (Q1-Q0) #soft?
            x_hat = (torch.sign(tent)+1)/2 #hard bits estimated
            
            check = torch.sum(mod(torch.mm(self.H, x_hat.T)))
            if check == 0:
                success = 1
        
        if self.decode_format == 'bits':
            return x_hat
        elif self.decode_format == 'llr':
            return -tent
        else:
            assert False, "Code parameter `decode_format` must be either `bits` or `llr`"
    
    def LDPC_decode(self, llr, punctured = True):
        
        # Potenially faster than "LDPC_decode_v0" 
        # LLR inputs in the form of ln(p0/p1)
        # Generic SPA works for any LDPC code
        # H matrix should be given as pytorch.sparse (COO) tensor, must be coalesced
        # https://pytorch.org/docs/stable/sparse.html
        
        if punctured == True:
            llr = torch.hstack([(1e-8)*torch.ones(llr.shape[0], int(2*self.Zc)).to(self.device), llr])
            
            
        Hind = self.H.indices()
                
        # Initialization
        
        # In this future, consider using llr_ch.sparse_mask(H)
        llr_ch = llr[:,Hind[1]]
        v_to_c = llr_ch
        
        k=0
        success = 0
        
        while ((success == 0) and (k<self.max_iter)):
            k += 1
            
            # Horizontal Step
            tanhllr = torch.tanh(0.5*v_to_c)
            tanhllr[torch.abs(tanhllr)<=1e-12] = 1e-12
            
            prod_tllr_v = multiply_reduce_dim1(self.H, tanhllr, self.device)
            prod_tllr = prod_tllr_v[:, Hind[0]]
            c_to_v = 2.0*atanh(prod_tllr/tanhllr, self.eps)


            # Vertical Step
            sum_var_v = sum_reduce_dim0(self.H, c_to_v, self.device)
            sum_var = sum_var_v[:, Hind[1]]
            dec_llr = sum_var_v + llr 
            v_to_c = sum_var + llr_ch - c_to_v
            
            # Final steps
            x_hat = (torch.sign(-dec_llr)+1)/2 #hard bits estimated
            
            check = torch.sum(mod(torch.mm(self.H, x_hat.T)))
            if check == 0:
                success = 1
        
        if self.decode_format == 'bits':
            return x_hat
        elif self.decode_format == 'llr':
            return dec_llr
        else:
            assert False, "Code parameter `decode_format` must be either `bits` or `llr`"
       
    
    def parity_check_LLR(self, llr):
        
        Hind = self.H.indices()
       
        llr_ch = llr[:,Hind[1]]
        
        tanhllr = torch.tanh(0.5*llr_ch)
        prod_tllr = multiply_reduce_dim1(self.H, tanhllr, self.device)
        #prod_tllr = prod_tllr_v[:, Hind[0]]

        return 2.0*atanh(prod_tllr, self.eps)
        
        
       
    
def atanh(x, eps):   
    clamp = 1-eps
    return torch.atanh(torch.clamp(x, min=-clamp, max=clamp))

def matlabsparse_2_coo(matlab_index_matrix, shape):
    matlab_index_matrix = matlab_index_matrix - 1
    return torch.sparse_coo_tensor(matlab_index_matrix, torch.ones(matlab_index_matrix.shape[1]), shape).coalesce()
    
    
    
def decimal_to_binary_tensor(value, width=4):
    string = format(value, '0{}b'.format(width))
    binary = [0 if c == '0' else 1 for c in string]
    return torch.tensor(binary, dtype=torch.float64)

def mod(x):
    return torch.remainder(x,2)

# Computes torch.prod(Hs*llr, dim = 1).T for sparse Hs 
# and vector llr (broadcast multiply) for all llr in batch and saves as
# numsamp x num_parity_checks
def multiply_reduce_dim1(Hs, llr_row, device):
    ind = Hs.indices()
    prod_mat = torch.ones(llr_row.shape[0], Hs.shape[0]).to(device)
    indmat = ind[0].repeat(llr_row.shape[0], 1)
    prod_mat.scatter_(1, indmat, llr_row, reduce = 'multiply')
    return prod_mat

def multiply_reduce_dim0(Hs, llr_col, device):
    ind = Hs.indices()
    prod_mat = torch.ones(llr_col.shape[0], Hs.shape[1]).to(device)
    indmat = ind[1].repeat(llr_col.shape[0], 1)
    prod_mat.scatter_(1, indmat, llr_col, reduce = 'multiply')
    return prod_mat

def sum_reduce_dim0(Hs, llr_col, device):
    ind = Hs.indices()
    prod_mat = torch.zeros(llr_col.shape[0], Hs.shape[1]).to(device)
    indmat = ind[1].repeat(llr_col.shape[0], 1)
    prod_mat.scatter_(1, indmat, llr_col, reduce = 'add')
    return prod_mat
  
def resize_sparse(H, nmk, n, device):
    ind = H.indices()
    
    indmask = (ind[0]<nmk) * (ind[1]<n)
    indout = ind[:, indmask]
    val = torch.ones(indout.shape[1]).to(device)
    
    return torch.sparse_coo_tensor(indout, val, (nmk, n)).coalesce()
