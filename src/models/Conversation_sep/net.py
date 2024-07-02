import torch
import torch.nn as nn

import torch.nn.functional as F
import src.utils as utils
from src.utils import count_parameters


class FilmLayer(nn.Module):
    def __init__(self, D, C, nF):
        super().__init__()
        self.D = D
        self.C = C
        self.nF = nF
        self.weight = nn.Conv1d(self.D, self.C, 1)
        self.bias = nn.Conv1d(self.D, self.C, 1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        """
        x: (B, D, T) # X F
        embedding: (B, D)
        """
        B, D, T = x.shape
        w = self.weight(embedding).reshape(B, self.C, 1) # (B, C, F, 1)
        b = self.weight(embedding).reshape(B, self.C, 1) # (B, C, F, 1)

        return x * w + b


'''
MERGE USE FilmLayer
1. MEL_SPECT + CNN + LSTM + Linear + relu + norm + average
2. TFG-net
'''


class Net(nn.Module):
    def __init__(self, summary_model, summary_params, sep_model, sep_params):
        super(Net, self).__init__()
        if summary_model == "None":
            self.summarize = 0
        else:
            self.summarize = 1
            self.Summarize_net = utils.import_attr(summary_model)(**summary_params)
            sep_params["spk_emb_dim"] = self.Summarize_net.output_dim
        self.sep_net = utils.import_attr(sep_model)(**sep_params)
        count_parameters(self)
        
    def forward(self, inputs):
        x = inputs['mixture']
        embed = inputs['embed']
        
        ### summarizr the example to generate the embed
        if self.summarize:
            example = inputs['example']
            Ls = inputs['seq_len']
            embed = self.Summarize_net(example, embed, Ls)

        ### 
        y = self.sep_net(x, embed)
        return y

class NetWithoutEmebdding(Net):
    def __init__(self, *args, **kwargs):
        super(NetWithoutEmebdding, self).__init__(*args, **kwargs)
        
    def forward(self, inputs):
        x = inputs['mixture']
        
        y = self.sep_net(x)
        return y

if __name__ == "__main__":
    model_params = {
        "stft_chunk_size": 192,
        "stft_pad_size": 96,
        "stft_back_pad": 0,
        "num_ch": 6,
        "D": 16,
        "L": 4,
        "I": 1,
        "J": 1,
        "B": 4,
        "H": 64,
        "E": 2,
        "local_atten_len": 50,
        "use_attn": False,
        "lookahead": True,
        "chunk_causal": True,
        "use_first_ln": True,
        "merge_method": "early_cat",
        "directional": True
    }
    device = torch.device('cpu') ##('cuda')
    model = Net(**model_params).to(device)

    num_chunk = 50
    test_num = 10
    chunk_size = model_params["stft_chunk_size"]
    look_front = model_params["stft_pad_size"]
    look_back = model_params["stft_back_pad"] #model_params["lookback"]
    x = torch.rand(4, 6, look_back + chunk_size*num_chunk + look_front)
    x = x.to(device)
    x2 = x[..., :look_back + chunk_size*test_num + look_front]
    inputs = {"mixture": x}
    inputs2 = {"mixture": x2}
    y = model(inputs, pad=False)['output']
    y2 = model(inputs2, pad=False)['output']

    print(x.shape, y.shape, y2.shape)
    _id  = 3
    check_valid = torch.allclose(y2[:, 0, :chunk_size*test_num], y[:, 0, :chunk_size*test_num], atol=1e-2 )
    print((y2[_id, 0, :chunk_size*test_num] - y[_id, 0, :chunk_size*test_num]).abs().max())
    print(check_valid)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot( y[_id, 0, :chunk_size*test_num].detach().numpy())
    # plt.plot( y2[_id, 0, :chunk_size*test_num].detach().numpy(), linestyle = '--', color = 'r')
    # plt.show()
