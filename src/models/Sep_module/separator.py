import math
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet2.torch_utils.get_layer_from_string import get_layer
from espnet2.enh.separator.abs_separator import AbsSeparator

from asteroid_filterbanks import make_enc_dec
from torch.nn import init
from torch.nn.parameter import Parameter
import src.utils as utils


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class LayerNormPermuted(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNormPermuted, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F]
        """
        x = x.permute(0, 2, 3, 1) # [B, T, F, C]
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2) # [B, C, T, F]
        return x


class FilmLayer(nn.Module):
    def __init__(self, D, C, nF, groups = 1):
        super().__init__()
        self.D = D
        self.C = C
        self.nF = nF
        self.weight = nn.Conv1d(self.D, self.C * nF, 1, groups = groups)
        self.bias = nn.Conv1d(self.D, self.C * nF, 1, groups = groups)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        """
        x: (B, D, F, T)
        embedding: (B, D, F)
        """
        B, D, _F, T = x.shape
        w = self.weight(embedding).reshape(B, self.C, _F, 1) # (B, C, F, 1)
        b = self.weight(embedding).reshape(B, self.C, _F, 1) # (B, C, F, 1)

        return x * w + b

class TFGridNet(AbsSeparator):
    """Offline TFGridNet

    Reference:
    [1] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Integrating Full- and Sub-Band Modeling for Speech Separation",
    in arXiv preprint arXiv:2211.12433, 2022.
    [2] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe,
    "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural
    Speaker Separation", in arXiv preprint arXiv:2209.03952, 2022.

    NOTES:
    As outlined in the Reference, this model works best when trained with variance
    normalized mixture input and target, e.g., with mixture of shape [batch, samples,
    microphones], you normalize it by dividing with torch.std(mixture, (1, 2)). You
    must do the same for the target signals. It is encouraged to do so when not using
    scale-invariant loss functions such as SI-SDR.

    Args:
        input_dim: placeholder, not used
        n_srcs: number of output sources/speakers.
        n_fft: stft window size.
        stride: stft stride.
        window: stft window type choose between 'hamming', 'hanning' or None.
        n_imics: number of microphones channels (only fixed-array geometry supported).
        n_layers: number of TFGridNet blocks.
        lstm_hidden_units: number of hidden units in LSTM.
        attn_n_head: number of heads in self-attention
        attn_approx_qk_dim: approximate dimention of frame-level key and value tensors
        emb_dim: embedding dimension
        emb_ks: kernel size for unfolding and deconv1D
        emb_hs: hop size for unfolding and deconv1D
        activation: activation function to use in the whole TFGridNet model,
            you can use any torch supported activation e.g. 'relu' or 'elu'.
        eps: small epsilon for normalization layers.
        use_builtin_complex: whether to use builtin complex type or not.
    """

    def __init__(
        self,
        input_dim,
        n_srcs=2,
        n_fft=128,
        look_back = 0,
        stride=64,
        window="hann",
        n_imics=1,
        n_layers=6,
        lstm_hidden_units=192,
        lstm_down=4,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=1,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        ref_channel=-1,
        use_attn=True,
        chunk_causal=True,
        local_atten=False,
        use_first_ln=False,
        conv_lstm = True,
        fb_type='stft',
        spk_emb_dim=256,
        lstm_fold_chunk = -1,
        groups = 1,
        chunk_lstm = False,
        seq2seq_mdl = None,
        seq2seq_mdl_params = None,
        use_flash_attention = False,
        pos_enc = False,
        pool = "mean",
        spk_id = True
    ):
        super().__init__()
        self.n_srcs = n_srcs
        self.n_layers = n_layers
        self.n_imics = n_imics
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.ref_channel = ref_channel
        self.emb_dim = emb_dim
        self.eps = eps
        self.chunk_size = stride
        self.chunk_lstm = chunk_lstm
        self.istft_pad = n_fft - stride
        self.lookback = look_back
        self.lookahead = self.istft_pad - look_back

        # ISTFT overlap-add will affect this many chunks in the future
        self.istft_lookback = 1 + (self.istft_pad - 1) // self.istft_pad
        
        self.enc, self.dec = make_enc_dec(fb_type,
                                        n_filters = n_fft,
                                        kernel_size = n_fft,
                                        stride=stride,
                                        window_type=window)

        t_ksize = 3
        self.t_ksize = t_ksize
        ks, padding = (t_ksize, 3), (1, 1)

        module_list = [nn.Conv2d(2 * n_imics, emb_dim, ks, padding=padding)]
        
        if use_first_ln:
            module_list.append(LayerNormPermuted(emb_dim))
        
        self.conv = nn.Sequential(
            *module_list
        )

        self.blocks = nn.ModuleList([])

        self.spk_id = spk_id
        if self.spk_id:
            self.embeds = nn.ModuleList([])

        for _i in range(n_layers):
            self.blocks.append(
                GridNetBlock(
                    emb_dim,
                    emb_ks,
                    emb_hs,
                    n_freqs,
                    lstm_hidden_units,
                    lstm_down,
                    n_head=attn_n_head,
                    approx_qk_dim=attn_approx_qk_dim,
                    activation=activation,
                    eps=eps,
                    use_attn=use_attn,
                    chunk_causal=chunk_causal,
                    local_atten=local_atten,
                    conv_lstm = conv_lstm,
                    lstm_fold_chunk = lstm_fold_chunk,
                    pos_enc = pos_enc,
                    seq2seq_mdl=seq2seq_mdl,
                    seq2seq_mdl_params=seq2seq_mdl_params,
                    use_flash_attention = use_flash_attention,
                    chunk_lstm = chunk_lstm,
                    pool = pool
                )
            )

            if _i > 0 and self.spk_id:
                self.embeds.append(FilmLayer(spk_emb_dim, emb_dim, n_freqs, groups))


        self.deconv = nn.ConvTranspose2d(emb_dim, n_srcs * 2, ks, padding=( self.t_ksize - 1, 1))
    
    def init_buffers(self, batch_size, device):
        return None

    def forward(
        self,
        input: torch.Tensor,
        embed: torch.Tensor,
        input_state
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor): batched multi-channel audio tensor with
                    M audio channels and N samples [B, N, M]
            ilens (torch.Tensor): input lengths [B]
            additional (Dict or None): other data, currently unused in this model.

        Returns: # MODIFIED: WILL RETURN [B, M, N] !!!
            enhanced (List[Union(torch.Tensor)]):
                    [(B, T), ...] list of len n_srcs
                    of mono audio tensors with T samples.
            ilens (torch.Tensor): (B,)
            additional (Dict or None): other data, currently unused in this model,
                    we return it also in output.
        """

        # print("input", input.shape)
        # n_samples = input.shape[-1]

        if input_state is None:
            input_state = self.init_buffers(input.shape[0], input.device)

        input_stft = self.enc(input) # [B, M, nfft + 2, T]
        if self.n_imics == 1:
            batch = input_stft.unsqueeze(1)

        imag = batch[..., self.n_freqs:, :]
        real = batch[..., :self.n_freqs, :]
        batch = torch.cat((real, imag), dim=1)  # [B, 2*M, F, T]
        batch = batch.transpose(2, 3) # [B, M, T, F]
        n_batch, _, n_frames, n_freqs = batch.shape # B, 2M, T, F
        # print("batch = ", batch.shape)

        batch = self.conv(batch)  # [B, -1, T, F]

        # print(batch.shape)

        # TODO: CHANGE INPUT OUTPUT TO B,T,Q,C TO AVOID PERMUTES
        for ii in range(self.n_layers):
            batch = batch.transpose(2, 3)
            if self.spk_id and ii > 0:
                batch = self.embeds[ii - 1](batch, embed)
            batch = batch.transpose(2, 3)
            batch, _ = self.blocks[ii](batch, None)  # [B, -1, T, F]

        batch = nn.functional.pad(batch, (0, 0, 1, 1))
        
        batch = self.deconv(batch)  # [B, n_srcs*2, T, F]batch ] 
        batch = batch.view([n_batch, self.n_srcs, 2, n_frames, n_freqs]) # [B, n_srcs, 2, n_frames, n_freqs]
        
        batch = batch.transpose(3, 4) # (B, n_srcs, 2, n_fft//2 + 1, T)
        
        # Concat real and imaginary parts
        batch = torch.cat([batch[:, :, 0], batch[:, :, 1]], dim=2) # (B, n_srcs, nfft + 2, T)

        #print(batch.shape)
        batch = self.dec(batch) # [B, n_srcs, n_srcs, -1]
        batch = batch[..., self.lookback:-self.lookahead]

        # batch = batch * mix_std_  # reverse the RMS normalization

        return batch, input_state

    @property
    def num_spk(self):
        return self.n_srcs

    @staticmethod
    def pad2(input_tensor, target_len):
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, target_len - input_tensor.shape[-1])
        )
        return input_tensor

class Attention_STFT(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)
    def __init__(
            self,
            emb_dim,
            n_freqs,
            approx_qk_dim=512, 
            n_head=4,
            activation="prelu",
            eps=1e-5,
            pos_enc = False,
            skip_conn = True,
            use_flash_attention = False,
            dim_feedforward = -1
        ):
        super().__init__()
        self.pos_enc = pos_enc
        if self.pos_enc:
            self.position_code = utils.PositionalEncoding(emb_dim*n_freqs, max_len=5000)

        self.skip_conn = skip_conn
        self.n_freqs = n_freqs
        self.E = math.ceil(
                    approx_qk_dim * 1.0 / n_freqs
                )  # approx_qk_dim is only approximate
        self.n_head = n_head
        self.V_dim = emb_dim // n_head
        self.emb_dim = emb_dim
        assert emb_dim % n_head == 0
        E = self.E

        self.use_flash_attention = use_flash_attention

        self.add_module(
            "attn_conv_Q",
            nn.Sequential(
                nn.Linear(emb_dim, E * n_head), # [B, T, Q, HE]
                get_layer(activation)(),
                # [B, T, Q, H, E] -> [B, H, T, Q, E] ->  [B * H, T, Q * E]
                Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)\
                                    .permute(0, 3, 1, 2, 4)\
                                    .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)), # (BH, T, Q * E)
                LayerNormalization4DCF((n_freqs, E), eps=eps),
            ),
        )
        self.add_module(
            "attn_conv_K",
            nn.Sequential(
                nn.Linear(emb_dim, E * n_head),
                get_layer(activation)(),
                Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, E)\
                                    .permute(0, 3, 1, 2, 4)\
                                    .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * E)),
                LayerNormalization4DCF((n_freqs, E), eps=eps),
            ),
        )
        self.add_module(
            "attn_conv_V",
            nn.Sequential(
                nn.Linear(emb_dim, (emb_dim // n_head) * n_head),
                get_layer(activation)(),
                Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2], n_head, (emb_dim // n_head))\
                                    .permute(0, 3, 1, 2, 4)\
                                    .reshape(x.shape[0] * n_head, x.shape[1], x.shape[2] * (emb_dim // n_head))),
                LayerNormalization4DCF((n_freqs, emb_dim // n_head), eps=eps),
            ),
        )
        self.dim_feedforward = dim_feedforward
        if dim_feedforward == -1:
            self.add_module(
                "attn_concat_proj",
                nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    get_layer(activation)(),
                    Lambda(lambda x: x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])),
                    LayerNormalization4DCF((n_freqs, emb_dim), eps=eps)
                ),
            )
        else:
            self.linear1 = nn.Linear(emb_dim, dim_feedforward)
            self.dropout = nn.Dropout(p =0.1)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(dim_feedforward, emb_dim)
            self.dropout2 = nn.Dropout(p =0.1)
            self.norm = LayerNormalization4DCF((n_freqs, emb_dim), eps=eps)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
        
    
    def forward(self, batch):
        ### input/output B T F C
        # attention
        inputs = batch
        B0, T0, Q0, C0 = batch.shape

        if self.pos_enc:
            pos_code = self.position_code(batch) # 1, T, embed_dim
            # print("pos_code", pos_code.shape)
            _, T, QC = pos_code.shape
            pos_code = pos_code.reshape(1, T, Q0, C0)
            batch = batch + pos_code
        

        Q = self["attn_conv_Q"](batch) # [B', T, Q * C]
        K = self["attn_conv_K"](batch) # [B', T, Q * C]
        V = self["attn_conv_V"](batch) # [B', T, Q * C]
        # print("Q", Q.shape, V.shape)
        if self.use_flash_attention:
            from flash_attn import flash_attn_func
            Q = Q.reshape(B0, self.n_head, T0, Q0*self.E).transpose(1, 2) # [B, T, H, QC]
            K = K.reshape(B0, self.n_head, T0, Q0*self.E).transpose(1, 2) # [B, T, H, QC]
            V = V.reshape(B0, self.n_head, T0, Q0*self.V_dim).transpose(1, 2) # [B, T, H, QC]

            out = flash_attn_func(Q, K, V) # [B, T, H, QC]
            out = out.reshape(B0, T0, self.n_head, Q0, self.V_dim).transpose(2, 3) # [B, T, Q, H, C]
            batch = out.reshape(B0, T0, Q0, self.named_buffers * self.V_dim) # [B, T, Q, HC]
        else:
            # print("K=", K.shape, "Q=", Q.shape, "V=", V.shape)
            emb_dim = Q.shape[-1]

            attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
            attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
            #print("KQV2", K.shape, Q.shape, V.shape)
            # print("attn_mat", attn_mat.shape)

            V = torch.matmul(attn_mat, V)  # [B', T, Q*C]
            V = V.reshape(-1, T0, V.shape[-1]) # [BH, T, Q * C]
            V = V.transpose(1, 2) # [B', Q * C, T]
            
            batch = V.reshape(B0, self.n_head, self.n_freqs, self.V_dim, T0) # [B, H, Q, C, T]
            batch = batch.transpose(2, 3) # [B, H, C, Q, T]
            batch = batch.reshape(B0, self.n_head * self.V_dim, self.n_freqs, T0) # [B, HC, Q, T]
            batch = batch.permute(0, 3, 2, 1) # [B, T, Q, C]
        
        if self.dim_feedforward == -1:
            batch = self["attn_concat_proj"](batch) # [B, T, Q * C]
        else:
            batch = batch + self._ff_block(batch)  # [B, T, Q, C]
            batch = batch.reshape(batch.shape[0], batch.shape[1], batch.shape[2] * batch.shape[3])
            batch = self.norm(batch)
        batch = batch.reshape(batch.shape[0], batch.shape[1], Q0, C0)  # [B, T, Q, C])

        # Add batch if attention is performed
        if self.skip_conn:
            return batch + inputs
        else:
            return batch

class GridNetBlock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        lstm_down,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
        use_attn=True,
        chunk_causal = True,
        conv_lstm = True,
        lstm_fold_chunk = -1,
        pos_enc = False,
        chunk_lstm = True,
        local_atten = False,
        seq2seq_mdl=None,
        seq2seq_mdl_params=None,
        use_flash_attention = False,
        pool = "mean"
    ):
        super().__init__()
        bidirectional = True # Non-Causal
        if chunk_lstm:
            self.global_atten = use_attn
            self.local_atten = local_atten
        else:
            self.global_atten = use_attn
            self.local_atten = False
        
        self.chunk_lstm = chunk_lstm
        

        self.pool = pool
        
        self.lstm_fold_chunk = lstm_fold_chunk
        self.E = math.ceil(
                    approx_qk_dim * 1.0 / n_freqs
                )  # approx_qk_dim is only approximate
        
        self.V_dim = emb_dim // n_head
        self.chunk_causal = chunk_causal
        self.H = hidden_channels
        in_channels = emb_dim * emb_ks
        self.in_channels = in_channels
        self.n_freqs = n_freqs
        self.use_flash_attention = use_flash_attention

        ## intra RNN can be optimized by conv or linear because the frequence length are not very large
        self.intra_norm = LayerNormalization4D_old(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels*2, emb_dim, emb_ks, stride=emb_hs
        )
        self.inter_norm = LayerNormalization4D_old(emb_dim, eps=eps)
        
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs


        if self.local_atten:
            self.local_atten = Attention_STFT(
                emb_dim=emb_dim,
                n_freqs=n_freqs,
                approx_qk_dim=approx_qk_dim, 
                n_head=n_head,
                activation=activation,
                eps=eps,
                pos_enc = pos_enc
                # dim_feedforward = 64
            )
        else:
            
            self.inter_rnn = nn.LSTM( 
                in_channels, hidden_channels, 1, batch_first=True, bidirectional=bidirectional
            )
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels*(bidirectional + 1), emb_dim, emb_ks, stride=emb_hs
            )

        if self.global_atten:
            self.pool_atten = Attention_STFT(
                emb_dim=emb_dim,
                n_freqs=n_freqs,
                approx_qk_dim=approx_qk_dim, 
                n_head=n_head,
                activation=activation,
                eps=eps,
                pos_enc = pos_enc
            )

        self.seq2seq_mdl = seq2seq_mdl
        
        if self.seq2seq_mdl is not None:
            input_dim = seq2seq_mdl_params['input_dim']
            self.channel_compression = nn.Linear(emb_dim * n_freqs, input_dim)
            self.seq2seq_mdl = utils.import_attr(self.seq2seq_mdl)(**seq2seq_mdl_params)
            self.channel_decompression = nn.Linear(input_dim, emb_dim * n_freqs)

        if self.use_flash_attention:
            self.flash_attention = Attention_STFT(
                                        emb_dim=emb_dim,
                                        n_freqs=n_freqs,
                                        approx_qk_dim=approx_qk_dim, 
                                        n_head=n_head,
                                        activation=activation,
                                        eps=eps,
                                        pos_enc = pos_enc,
                                        use_flash_attention = True
                                    )



    
    def init_buffers(self, batch_size, device):
        return None

    def _unfold_timedomain(self, x):
        ### x - [BQ, C, T]
        BQ, C, T= x.shape
        # print(x.shape)
        # x = x.unfold(2, self.lstm_fold_chunk, self.lstm_fold_chunk) ### BQ, C, NUM_CHUNK, 500
        x = torch.split(x, self.lstm_fold_chunk, dim=-1) # [Num_chunk, BQ, C, 100]
        x = torch.cat(x, dim=0).reshape(-1, BQ, C, self.lstm_fold_chunk) # [Num_chunk, BQ, C, 100]
        x = x.permute(1, 0, 3, 2)
        # print("SHAPE", x.shape)
        # x = x.permute(0, 2, 3, 1)  ### BQ, NUM_CHUNK, 500, C
        return x

    def forward(self, x, init_state = None):
        """GridNetBlock Forward.

        Args:
            x: [B, C, T, Q]
            out: [B, C, T, Q]
        """
        MAX_LEN = 5000
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # intra RNN
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, Q]
        intra_rnn = (
            intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)
        )  # [BT, C, Q]
        
        
        # intra_rnn = F.unfold(
        #     intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
        # )  # [BT, C*emb_ks, -1]
        intra_rnn = torch.split(intra_rnn, self.emb_ks, dim=-1) # [Q/I, BT, C, I]
        intra_rnn = torch.stack(intra_rnn, dim=0)
        intra_rnn = intra_rnn.permute(1, 2, 3, 0).flatten(1, 2) # [BT, CI, Q/I]

        
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, nC*emb_ks]
        self.intra_rnn.flatten_parameters()
        '''
            frequency-domain LSTM
        '''
        # batch_lists = []
        # # print("intra_rnn = ", intra_rnn.shape)
        # ITER_NUM = int(np.ceil(intra_rnn.shape[0]/MAX_LEN))
        # for ii in range(ITER_NUM):
        #     tmp, _ = self.intra_rnn(intra_rnn[ii*MAX_LEN:(ii+1)*MAX_LEN])  # [BT, -1, H]
        #     batch_lists.append(tmp)
        # intra_rnn = torch.cat(batch_lists, dim = 0)
        intra_rnn, _ = self.intra_rnn(intra_rnn)  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, Q]
        intra_rnn = intra_rnn + input_  # [B, C, T, Q]
        intra_rnn = intra_rnn[:, :, :, :old_Q] # [B, C, T, Q]
        Q = old_Q
        # print(self.chunk_lstm)
        if self.chunk_lstm == False:
            # inter RNN
            input_ = intra_rnn
            inter_rnn = self.inter_norm(input_)  # [B, C, T, F]
            inter_rnn = (
                inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
            )  # [BF, C, T]
            inter_rnn = F.unfold(
                inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
            )  # [BF, C*emb_ks, -1]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]
            self.inter_rnn.flatten_parameters()
            inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
            inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
            inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
            inter_rnn = inter_rnn.view([B, Q, C, T])
            inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()  # [B, C, T, Q]
            inter_rnn = inter_rnn + input_  # [B, C, T, Q]

            # Output is inter_rnn by default
            inter_rnn = inter_rnn[..., :old_T, :old_Q] # [B, C, T, Q]

            
            if self.global_atten:
                ## sparse attnetion 
                # print("original TFGRIDNET")
                inter_rnn = inter_rnn.permute(0, 2, 3, 1) # B T Q C
                inter_rnn = self.pool_atten(inter_rnn) #B T Q C
                inter_rnn = inter_rnn.permute(0, 3, 1, 2)  #B C T Q
      
            return inter_rnn, init_state
        #### other if condition
        elif self.seq2seq_mdl:
            input_ = intra_rnn
            
            # Merge C with Q
            tensor = intra_rnn.transpose(1, 2) # [B, T, C, Q]
            tensor = tensor.reshape(B, T, -1) # [B, T, C * Q]

            tensor = self.channel_compression(tensor) # [B, T, D]
            
            # Pass to seq2seq model 
            tensor = self.seq2seq_mdl(tensor)

            # Decompress D to CQ
            tensor = self.channel_decompression(tensor) # [B, T, C * Q]
            
            # Decouple C and Q
            tensor = tensor.reshape(B, T, -1, Q) # [B, T, C, Q]
            tensor = tensor.transpose(1, 2) # [B, C, T, Q]

            # Residual
            tensor = input_ + tensor
            
            return tensor, init_state
        elif self.use_flash_attention:
            
            input_ = intra_rnn # [B, C, T, Q]
            
            tensor = intra_rnn.permute(0, 2, 3, 1) # [B, T, Q, C]
            atten_output = self.flash_attention(tensor)
            atten_output = atten_output.permute(0, 3, 1, 2) # [B, C, T, Q]
            
            tensor = atten_output + input_
            
            return tensor, init_state
        else:
            # fold the time domain to chunk
            inter_rnn = self.inter_norm(intra_rnn)  # [B, C, T, F]
            inter_rnn = (
                inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)
            )  # [BF, C, T]
            inter_rnn = self._unfold_timedomain(inter_rnn) ### BQ, NUM_CHUNK, CHUNK_SIZE, C

            
            BQ, NUM_CHUNK, CHUNKSIZE, C  = inter_rnn.shape
            # print("inter_rnn", inter_rnn.shape)

            if self.local_atten:
                ####### local attention
                inter_rnn = inter_rnn.reshape(B, Q, NUM_CHUNK, CHUNKSIZE, C) 
                inter_rnn = inter_rnn.permute(0, 2, 3, 1, 4)
                inter_rnn = inter_rnn.reshape(B*NUM_CHUNK, CHUNKSIZE, Q, C)
                # print("local attention", inter_rnn.shape)
                inter_rnn = self.local_atten(inter_rnn) #B T Q C
                inter_rnn = inter_rnn.reshape(B, NUM_CHUNK, CHUNKSIZE, Q, C)
                inter_rnn = inter_rnn.permute(0, 3, 1, 2, 4) # B, Q, NUM_CHUNK, CHUNKSIZE, C
            else:
                ####### local_lstm
                
                inter_rnn = inter_rnn.reshape(BQ*NUM_CHUNK, CHUNKSIZE, C) ### BQ* NUM_CHUNK, CHUNK_SIZE, C
                inter_rnn = inter_rnn.transpose(2, 1) # [B, C, T]
                input_ = inter_rnn
                
                
                # inter_rnn = F.unfold(
                #     inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1)
                # )  # [BQ*NUM_CHUNK, C*emb_ks, -1]
                # inter_rnn_even = inter_rnn[..., 0::2]
                # inter_rnn_odd = inter_rnn[..., 1::2]
                # inter_rnn = torch.cat([inter_rnn_even.unsqueeze(2), inter_rnn_odd.unsqueeze(2)], dim=2)

                inter_rnn = torch.split(inter_rnn, self.emb_ks, dim=-1) # [T/2, BQ, C, 2]
                inter_rnn = torch.stack(inter_rnn, dim=0)
                # print("SAHPE", inter_rnn.shape)
                inter_rnn = inter_rnn.permute(1, 2, 3, 0)
                
                BF, C, EO, _T = inter_rnn.shape
                inter_rnn = inter_rnn.reshape(BF, C* EO, _T)
                
                inter_rnn = inter_rnn.transpose(1, 2)  # [BF, -1, C*emb_ks]

                
                self.inter_rnn.flatten_parameters()
                # print("inter_rnn", inter_rnn.shape)
                inter_rnn, _ = self.inter_rnn(inter_rnn)  # [BF, -1, H]
                inter_rnn = inter_rnn.transpose(1, 2)  # [BF, H, -1]
                inter_rnn = self.inter_linear(inter_rnn)  # [BF, C, T]
                inter_rnn = inter_rnn + input_  # [BQ* NUM_CHUNK, C, T]
                #######      

                inter_rnn = inter_rnn.reshape(B, Q, NUM_CHUNK, C, CHUNKSIZE)        
                inter_rnn = inter_rnn.permute(0, 1, 2, 4, 3)  # B, Q, NUM_CHUNK, CHUNKSIZE, C
                
                # inter_rnn = inter_rnn.reshape(B, Q, NUM_CHUNK, CHUNKSIZE, C)
            
            if self.global_atten:
                ## sparse attnetion 
                input_ = inter_rnn # B, Q, NUM_CHUNK, CHUNKSIZE, C
                # print(self.pool)
                if self.pool == "mean":
                    inter_rnn = torch.mean(inter_rnn, dim = 3) # B, Q, NUM_CHUNK, C
                elif self.pool == "max":
                    inter_rnn, _ = torch.max(inter_rnn, dim = 3) # B, Q, NUM_CHUNK, C
                    # print(inter_rnn.shape)
                else:
                    raise ValueError("INvalid pool type!")
                inter_rnn = inter_rnn.transpose(1, 2)
                # print("pool_atten = ", inter_rnn.shape)
                inter_rnn = self.pool_atten(inter_rnn) #B T Q C
                inter_rnn = inter_rnn.transpose(1, 2)  #B Q T C
                inter_rnn = inter_rnn.unsqueeze(3)
                inter_rnn = input_ + inter_rnn # B, Q, NUM_CHUNK, CHUNKSIZE, C

            # Output is inter_rnn by default
            inter_rnn = inter_rnn.reshape(B, Q, T, C)
            inter_rnn = inter_rnn.permute(0, 3, 2, 1) # B C T Q
            inter_rnn = inter_rnn[..., :old_T, :]

            return inter_rnn, init_state#, [t0 - t0_0, t1 - t0, t2 - t2_0, t3 - t2, t5 - t4, t7 - t6]

##
class LayerNormalization4D_old(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat

# Use native layernorm implementation
class LayerNormalization4D(nn.Module):
    def __init__(self, C, eps=1e-5, preserve_outdim=False):
        super().__init__()
        self.norm = nn.LayerNorm(C, eps=eps)
        self.preserve_outdim = preserve_outdim

    def forward(self, x: torch.Tensor):
        """
        input: (*, C)
        """
        x = self.norm(x)
        return x
    
class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        assert len(input_dimension) == 2
        Q, C = input_dimension
        super().__init__()
        self.norm = nn.LayerNorm((Q * C), eps=eps)

    def forward(self, x: torch.Tensor):
        """
        input: (B, T, Q * C)
        """
        x = self.norm(x)

        return x

def mod_pad(x, chunk_size, pad):
    # Mod pad the rminput to perform integer number of
    # inferences
    mod = 0
    if (x.shape[-1] % chunk_size) != 0:
        mod = chunk_size - (x.shape[-1] % chunk_size)

    x = F.pad(x, (0, mod))
    x = F.pad(x, pad)

    return x, mod

class Sep_Net(nn.Module):
    def __init__(self, stft_chunk_size = 160, stft_pad_size = 120, stft_back_pad = 0,
                 num_ch=2, D=64, B=6, I=1, J=1, L=0, H=128,
                 use_attn=False, lookahead=True, local_atten=False,
                 E = 4, chunk_causal=False, num_src = 2, use_first_ln=False, 
                 conv_lstm = True, fb_type='stft', spk_emb_dim = 256,
                 lstm_fold_chunk = -1, groups = 1, pos_enc = False, local_atten_len = 0,
                 seq2seq_mdl = None, seq2seq_mdl_params = None, use_flash_attention=False, chunk_lstm = True, pool = "mean", spk_id = True):
        
        super(Sep_Net, self).__init__()
        # print("chunk_lstm: ", chunk_lstm)
        self.stft_chunk_size = stft_chunk_size
        self.stft_pad_size = stft_pad_size
        self.num_ch = num_ch
        self.lookahead = lookahead
        self.lookback = stft_back_pad
        self.num_src = num_src
        self.embed_dim = D
        self.E = E
        # Input conv to convert input audio to a latent representation
        self.nfft = stft_back_pad + stft_chunk_size + stft_pad_size
        nfreqs = self.nfft//2 + 1

        # TF-GridNet
        self.tfgridnet = TFGridNet(None,
                                   n_srcs=num_src,
                                   n_fft=self.nfft,
                                   look_back = stft_back_pad,
                                   stride=stft_chunk_size,
                                   emb_dim=D,
                                   emb_ks=I,
                                   emb_hs=J,
                                   n_layers=B,
                                   n_imics=num_ch,
                                   attn_n_head=L,
                                   attn_approx_qk_dim=E*nfreqs,
                                   use_attn = use_attn,
                                   lstm_hidden_units=H,
                                   local_atten=local_atten,
                                   chunk_causal = chunk_causal,
                                   use_first_ln=use_first_ln,
                                   conv_lstm = conv_lstm,
                                   fb_type=fb_type,
                                   spk_emb_dim = spk_emb_dim,
                                   lstm_fold_chunk = lstm_fold_chunk,
                                   groups = groups,
                                   pos_enc = pos_enc,
                                   seq2seq_mdl=seq2seq_mdl,
                                   seq2seq_mdl_params=seq2seq_mdl_params,
                                   use_flash_attention=use_flash_attention,
                                   chunk_lstm=chunk_lstm,
                                   pool = pool,
                                   spk_id = spk_id)
        

    def init_buffers(self, batch_size, device):
        return self.tfgridnet.init_buffers(batch_size, device)


    def forward(self, x, embed, input_state = None, pad=True):
        if input_state is None:
            input_state = self.init_buffers(x.shape[0], x.device)
            
        mod = 0
        embed = embed.unsqueeze(2) #### B, 256, 1
        if pad:
            pad_size = (self.lookback, self.stft_pad_size) if self.lookahead else (0, 0)
            x, mod = mod_pad(x, chunk_size=self.stft_chunk_size, pad=pad_size)
        x, next_state = self.tfgridnet(x, embed, input_state)
        if mod != 0:
            x = x[:, :, :-mod]

        return {'output': x,  'next_state': next_state}

