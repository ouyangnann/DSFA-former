import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from lib.utils import calculate_random_walk_matrix

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import dgl
import dgl.function as fn

import math


from math import sqrt

from einops import rearrange
import scipy.sparse as sp  

#from model.ldm.module.text_embed.potion import get_jina





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def calculate_supports(adj_mx, threshold=0.0, pos_enc_dim=16, pos=False):
    """根据 adj_mx 计算图卷积所需的支持矩阵, 并添加Laplacian 位置编码。"""
    supports = []
    adj_mx[adj_mx < threshold] = 0
  

    
    adj_sp = sp.coo_matrix(adj_mx)
    g = dgl.from_scipy(adj_sp)

    # 提取边特征 - 节点间的距离
    edge_weights = adj_sp.data
    edge_weights = torch.tensor(adj_sp.data, dtype=torch.float)  # 提取边权重
    g.edata['e'] = edge_weights

    return g


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.input_dim = int(model_kwargs.get('input_dim'))
        
        self.output_dim = int(model_kwargs.get('output_dim'))
        self.num_node = int(model_kwargs.get('num_nodes'))
        self.enc_t_dim = int(model_kwargs.get('enc_t_dim'))

     
        self.num_heads = int(model_kwargs.get('num_heads'))
        self.num_encoder_layers = int(model_kwargs.get('num_encoder_layers'))
       
        self.dropout = float(model_kwargs.get('dropout', 0.1))
        self.l1_decay = float(model_kwargs.get('l1_decay', 1e-5))
        self.seq_len = int(model_kwargs.get('seq_len'))  # for the encoder
        self.horizon = int(model_kwargs.get('horizon'))  # for the decoder

        # Add additional parameters required by GTNModel
       
        self.g_threshold = int(model_kwargs.get('g_threshold',0))
    



class dsfa_former(nn.Module, Seq2SeqAttrs):
    def __init__(self, logger, graph, cuda, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.device = cuda
        self._logger = logger
        self.use_norm = True
       
        self.g = self.process_graph(graph).to(self.device)
        
        self.enc_embedding = DataEmbedding_inverted(
            c_in= self.seq_len, d_model=self.enc_t_dim, dropout=self.dropout)
        

        self.iG_encoder = Encoder(
            [
                GraphAwareEncoderLayer(
                    d_model=self.enc_t_dim,
                    n_heads=self.num_heads,
                    num_nodes=self.num_node,
                    dropout=self.dropout,
                    activation='gelu'
                ) for _ in range(self.num_encoder_layers)
            ],
            norm_layer=nn.LayerNorm(self.enc_t_dim)
        )
        
     

                
        self.projector = nn.Sequential(
            nn.LayerNorm(self.enc_t_dim*2),
            nn.Linear(self.enc_t_dim*2, self.seq_len * 2),
            nn.GELU(),
            nn.Linear(self.seq_len * 2, self.horizon)
        )


        
    def process_graph(self, adj_mx, noise_std=0.0):
        adj_sp = sp.coo_matrix(adj_mx * (adj_mx >= self.g_threshold))
        support = calculate_random_walk_matrix(adj_sp).T
        
        support.setdiag(0)
        support.eliminate_zeros()

        g = dgl.from_scipy(support)
        edge_features = torch.from_numpy(support.data).float().view(-1, 1)
        g.edata['e'] = edge_features
        g.edata['e'] = torch.tensor(support.data, dtype=torch.float)  
        if noise_std > 0:
            g.edata['e'] = g.edata['e'] + torch.randn_like(g.edata['e']) * noise_std

        zero_in_deg = torch.nonzero(g.in_degrees() == 0, as_tuple=True)[0]
        if len(zero_in_deg) > 0:
            # 批量添加自环边并设置边权为1.0
            g.add_edges(zero_in_deg, zero_in_deg, data={'e': torch.ones(len(zero_in_deg))})
        
        return g
  
  
    def forward(self, x):
  
        bs = x.shape[0]

        # 倒置嵌入 [B, L, N] -> [B, N, d_model]
        enc_out = self.enc_embedding(x, None)  # [B, N, d_model]
        res = enc_out
        batch_g = dgl.batch([self.g for _ in range(bs)])
        batch_g.ndata['h'] = enc_out.reshape(bs*self.num_node,-1)
        
 
        enc_out, _ = self.iG_encoder(enc_out, batch_g)
       
        enc_out = torch.cat([res,enc_out],dim=-1)
        dec_out = self.projector(enc_out).permute(0, 2, 1)  # [B, H, N]
        
      
        return dec_out[:,-self.horizon:,:]  # [B, H, N]


        
class GraphAwareEncoderLayer(nn.Module):
  
    def __init__(self, d_model, n_heads, num_nodes, dropout=0.1, activation='gelu', attn=''):
        super().__init__()
        # 时间注意力
        self.raw_rate = 0.6
        
        if attn == 'Prob':
            self.temporal_attn = AttentionLayer(
                ProbAttention(False,attention_dropout=dropout),
                d_model, n_heads
            )
            self.coss_attn = AttentionLayer(
                ProbAttention(False,attention_dropout=dropout),
                d_model, n_heads
            )
            
        elif attn == 'fast':
            from model.pytorch.performer import FastAttention
            self.temporal_attn = AttentionLayer(
                FastAttention(d_model//n_heads),
                d_model, n_heads
            )
            
            self.coss_attn = AttentionLayer(
                FastAttention(d_model//n_heads),
                d_model, n_heads
            )
        
        elif attn == 'lin':
            
            def get_EF(seq_len, proj_len, method="learnable", head_dim=None, bias=False):
              
                assert method in ["learnable", "convolution", "no_params"]

                if method == "convolution":
                    # 在序列维度上做卷积降采样
                    conv = nn.Conv1d(
                        in_channels=head_dim, 
                        out_channels=head_dim, 
                        kernel_size=seq_len // proj_len,
                        stride=seq_len // proj_len
                    )
                    return conv

                if method == "no_params":
                    mat = torch.zeros(seq_len, proj_len)
                    torch.nn.init.normal_(mat, mean=0.0, std=1.0 / proj_len)
                    return nn.Parameter(mat, requires_grad=False)  # 注册但不更新

                # learnable 参数矩阵: [seq_len, proj_len]
                proj = nn.Parameter(torch.empty(seq_len, proj_len))
                torch.nn.init.xavier_normal_(proj)
                return proj

            
            E_proj = get_EF(int(num_nodes*self.raw_rate), int(num_nodes*self.raw_rate)//2, 'learnable')
            F_proj = get_EF(int(num_nodes*self.raw_rate), int(num_nodes*self.raw_rate)//2, 'learnable')
        
            
            self.temporal_attn = AttentionLayer(
                LinearAttentionHead(d_model, dropout, E_proj, F_proj, causal_mask=None),
                d_model, n_heads
            )
            self.coss_attn = AttentionLayer(
                LinearAttentionHead(d_model, dropout, E_proj, F_proj, causal_mask=None),
                d_model, n_heads
            )
            
            
        
        else:
            self.temporal_attn = AttentionLayer(
                FullAttention(False, attention_dropout=dropout),
                d_model, n_heads
            )
            self.coss_attn = AttentionLayer(
                FullAttention(False, attention_dropout=dropout),
                d_model, n_heads
            )
        
        self.compress_q = nn.Linear(num_nodes, int(num_nodes*self.raw_rate))
        self.compress_kv = nn.Linear(num_nodes, int(num_nodes*self.raw_rate))
        

        
        # 图注意力
        self.gat =  nn.ModuleList([
            dglnn.GATConv(
            d_model, d_model//n_heads,  
            num_heads=n_heads,
            feat_drop=dropout,
            attn_drop=dropout),
            nn.GELU(),
            nn.LayerNorm(d_model),
        ]
        )
    
        # 前馈网络
        self.conv1 = nn.Conv1d(d_model, d_model*4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model*4, d_model, kernel_size=3, padding=1)
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model),
            nn.LayerNorm(d_model)
        ])
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu
        self.mu = nn.Parameter(torch.zeros(1))  
        self.sigma = nn.Parameter(torch.ones(1))
  
 
        self.last_weights_2 = None  
        self.last_W_q_2 = None      
        

        
    def gaussian_kernel(self, x):
        """可学习参数的高斯核函数"""
        return torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
    
  
    
    def forward(self, x, batch_graph, attn_mask=None):
        # 时间注意力
   
        B, N, D = x.shape

        xq = F.gelu(self.compress_q(x.permute(0,2,1))).permute(0,2,1)
        xkv = F.gelu(self.compress_kv(x.permute(0,2,1))).permute(0,2,1)
        
        new_x, attn = self.temporal_attn(xq, xkv, xkv)
      
        weights = F.softmax(new_x, dim=1) # B,H,D

        new_x= (weights * new_x).sum(dim=1) # B,D
        new_x = new_x.unsqueeze(1).expand(-1, N, -1) 
        
       
    
        new_x = x + self.dropout(new_x)
        new_x = self.norms[0](new_x)
    
       
        batch_graph.ndata['h'] = new_x.view(-1, D)
    
        batch_graph.edata['w'] = batch_graph.edata['e'].reshape(-1,1)
        batch_graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        current_feat = self.gaussian_kernel(batch_graph.ndata['h'])
       
        graph_x_ = new_x + current_feat.view(B, N, D)
        
        # --- 全局图注意力 ---
        graph_x = rearrange(graph_x_, 'b n d -> (b n) d')
        graph_out = self.gat[0](batch_graph, graph_x).reshape(B*N,-1)
        graph_out = self.gat[1](graph_out)
        graph_out = self.gat[2](graph_out)
        graph_out = rearrange(graph_out, '(b n) d -> b n d', b=B)
        
        graph_out_q = F.gelu(self.compress_q(graph_out.permute(0,2,1))).permute(0,2,1)

        new_kv =  F.gelu(self.compress_kv(new_x.permute(0,2,1))).permute(0,2,1)
        self.last_W_q_2 = self.compress_q.weight.detach().clone()
  
        
        graph_cos,_ = self.coss_attn(graph_out_q, new_kv, new_kv)
        
        weights = F.softmax(graph_cos, dim=1) # B,H,D
        
        self.last_weights_2 = weights.detach().clone()
        
        graph_cos= (weights * graph_cos).sum(dim=1) # B,D
        graph_cos = graph_cos.unsqueeze(1).expand(-1, N, -1) 
      
        y=graph_out = self.norms[1](graph_x_ + graph_out + graph_cos)

        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        
        return self.norms[2](y + graph_out), attn






class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        # x: [Batch Variate d_model]
        return self.dropout(x)
    

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
     
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        



class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x



class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, batch_graph, attn_mask=None):  # 确保接收attn_mask
        attns = []
        res=x
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(
                x,  batch_graph
                 # 根据实际情况传递图结构
            )
            
            attns.append(attn)
      
        return x, attns

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
    


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        
        out = out.reshape(B, L, -1)

        return self.out_projection(out), attn
    

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn
    
 
class LinearAttentionHead(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """
    def __init__(self, dim, dropout, E_proj, F_proj, causal_mask, full_attention=False):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.causal_mask = causal_mask
        self.is_proj_tensor = isinstance(E_proj, torch.Tensor)

    def forward(self, Q, K, V, **kwargs):
        """
        Linformer-style sequence projection on K and V (N -> k).
        Q, K, V: [B, N, H, d] or [B, H, N, d]
        E, F: sequence projections applied on the N-dimension:
            - nn.Parameter / torch.Tensor with shape [N, k]
            - nn.Linear with in_features == N and out_features == k
            - nn.Conv1d over sequence dimension (see below)
        Returns: [B, N, H, d_out]  (与多数上游接口兼容；若你需要 [B,H,N,d_out]，把最后一行 permute 去掉)
        """
        input_mask = kwargs.get("input_mask", None)         # [B, N] boolean -> masks K,V on N
        embeddings_mask = kwargs.get("embeddings_mask", None)  # [B, N] boolean -> masks Q on N
        visualize = kwargs.get("visualize", False)

        # -------------------------
        # 1) Normalize layout to [B, H, N, d]
        # -------------------------
        def to_bhnd(x):
            if x.dim() != 4:
                raise ValueError(f"Expected 4D tensor for Q/K/V, got {x.shape}")
            B, A, C, D = x.shape
            # heuristic: H << N usually; if C < A, treat x as [B,N,H,d]
            if C < A:   # [B, N, H, d] -> [B, H, N, d]
                return x.permute(0, 2, 1, 3).contiguous()
            return x.contiguous()  # already [B, H, N, d]

        Q = to_bhnd(Q)
        K = to_bhnd(K)
        V = to_bhnd(V)
        B, H, N, d = Q.shape
        if K.shape != (B, H, N, d) or V.shape != (B, H, N, d):
            raise ValueError(f"Q/K/V shapes mismatch after normalization: "
                            f"Q{Q.shape}, K{K.shape}, V{V.shape}")

        # -------------------------
        # 2) Apply masks on node axis N
        # -------------------------
        if input_mask is not None:
            mask = input_mask.to(dtype=torch.bool, device=K.device).view(B, 1, N, 1)
            K = K.masked_fill(~mask, 0.0)
            V = V.masked_fill(~mask, 0.0)
        if embeddings_mask is not None:
            mask = embeddings_mask.to(dtype=torch.bool, device=Q.device).view(B, 1, N, 1)
            Q = Q.masked_fill(~mask, 0.0)

        # -------------------------
        # 3) Flatten heads for efficient bmm
        # -------------------------
        Qb = Q.view(B * H, N, d)   # [B*H, N, d]
        Kb = K.view(B * H, N, d)   # [B*H, N, d]
        Vb = V.view(B * H, N, d)   # [B*H, N, d]

        # -------------------------
        # 4) Helper: project along sequence dimension N -> k
        #    x: [B*H, N, d]  ->  x_proj: [B*H, k, d]
        # -------------------------
        def project_seq(x, proj):
            # Tensor / Parameter: [N, k]
            if isinstance(proj, torch.Tensor) and proj.dim() == 2:
                if proj.shape[0] != N:
                    raise ValueError(f"Projection shape {proj.shape} incompatible with N={N}")
                # x: [B*H, N, d], proj: [N, k]  ->  [B*H, k, d]
                return torch.einsum('bnd,nk->bkd', x, proj.to(x.dtype).to(x.device))

            # nn.Linear on N-dim: weight [k, N], bias [k]
            if isinstance(proj, torch.nn.Linear):
                if proj.in_features != N:
                    raise ValueError(f"Linear proj in_features={proj.in_features} must equal N={N}")
                # 把 N 维当作线性层的输入特征：对每个 d 通道独立应用
                # x: [B*H, N, d] -> [B*H, d, N] -> linear -> [B*H, d, k] -> [B*H, k, d]
                x_t = x.transpose(1, 2)  # [B*H, d, N]
                y = torch.matmul(x_t, proj.weight.t().to(x.dtype).to(x.device))  # [B*H, d, k]
                if proj.bias is not None:
                    y = y + proj.bias.to(x.dtype).to(x.device)
                return y.transpose(1, 2).contiguous()  # [B*H, k, d]

            # nn.Conv1d over sequence: in/out channels = d_head or d
            if isinstance(proj, torch.nn.Conv1d):
                # 期望对序列做降采样：输入 [B*H, C=d, L=N]，输出 [B*H, C=d, L=k]
                x_c = x.transpose(1, 2)  # [B*H, d, N]
                y = proj.to(x.device)(x_c)  # [B*H, d, k]
                return y.transpose(1, 2).contiguous()  # [B*H, k, d]

            raise TypeError(f"Unsupported projection type: {type(proj)}")

        # -------------------------
        # 5) Apply Linformer-style projections if needed
        # -------------------------
        if not self.full_attention:
            # K_proj: [B*H, k, d]
            Kp = project_seq(Kb, self.E)
            kp = Kp.size(1)  # k
            # V_proj: [B*H, k, d_v]  (与 Kp 的 k 对齐)
            Vp = project_seq(Vb, self.F)
            if Vp.size(1) != kp:
                raise ValueError(f"Projected V length {Vp.size(1)} != projected K length {kp}")
            # 注意力 logits: [B*H, N, d] @ [B*H, d, k] -> [B*H, N, k]
            scores = torch.bmm(Qb, Kp.transpose(1, 2))
            d_k = Kp.size(-1)
        else:
            # Vanilla: scores: [B*H, N, d] @ [B*H, d, N] -> [B*H, N, N]
            scores = torch.bmm(Qb, Kb.transpose(1, 2))
            d_k = d
            kp = N
            Vp = Vb  # [B*H, N, d]

        scores = scores / (float(d_k) ** 0.5)  # scale

        # -------------------------
        # 6) Causal mask (仅在 full 或 k==N 时严格支持)
        # -------------------------
        if getattr(self, "causal_mask", None) is not None:
            cm = self.causal_mask.to(device=scores.device, dtype=torch.bool)
            # 支持 [1,N,N] 或 [B,N,N]
            if cm.dim() == 3 and cm.size(-2) == N and cm.size(-1) == kp:
                # 若 kp==N，可直接广播；若 kp!=N，你需要自行构造被投影后的 mask
                if cm.size(0) == 1:
                    cm = cm.expand(B, N, kp)
                cm = cm.unsqueeze(1).expand(B, H, N, kp).contiguous().view(B * H, N, kp)
                scores = scores.masked_fill(~cm, float("-inf"))
            elif kp == N and cm.dim() == 3 and cm.size(-1) == N and cm.size(-2) == N:
                if cm.size(0) == 1:
                    cm = cm.expand(B, N, N)
                cm = cm.unsqueeze(1).expand(B, H, N, N).contiguous().view(B * H, N, N)
                scores = scores.masked_fill(~cm, float("-inf"))
            else:
                # 若使用 Linformer 投影（kp != N），严格的自回归掩码需要与投影一致的掩码设计
                # 这里给出友好提示并跳过，以保持数值稳定
                if kp != N:
                    if self.training:
                        print("[Warn] causal_mask provided but kp != N under Linformer projection; "
                            "skipping mask or please supply a projected causal mask of shape [1/B, N, k].")

        # -------------------------
        # 7) Softmax + dropout
        # -------------------------
        P_bar = torch.softmax(scores, dim=-1)   # [B*H, N, kp]
        if visualize:
            # 保存为 [B, H, N, kp] 便于可视化
            self.P_bar = P_bar.view(B, H, N, kp)
        P_bar = self.dropout(P_bar)

        # -------------------------
        # 8) 输出: [B*H, N, kp] @ [B*H, kp, d_v] -> [B*H, N, d_v]
        # -------------------------
        out = torch.bmm(P_bar, Vp)                 # [B*H, N, d_v]
        out = out.view(B, H, N, out.size(-1))      # [B, H, N, d_v]

        # 返回 [B, N, H, d_v]（若你后续期望 [B,H,N,d_v]，改为 `return out`）
        return out.permute(0, 2, 1, 3).contiguous(), None
