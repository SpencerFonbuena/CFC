import torch
class Hyperparameters:
    pred_size = 24
    lookback = 168
    seg_len = 6
    batch_size = 16
    split = .95
    if torch.cuda.is_available():
        num_workers = 12
    else:
        num_workers = 1
    data_dim = 5
    learning_rate = .0001
    win_size = 2
    factor = 10
    d_model = 512
    d_ff = 4 * d_model
    n_heads = 8
    e_layers = 3
    dropout = 0
    baseline = False

    '''
    win_size = 2,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=5, 
                dropout=0.0, baseline = False, device=torch.device('cuda:0')
    '''