import math

def human_format(number):
    """Convert number to a human-readable format with suffixes"""
    units = ['', 'K', 'M', 'B', 'T']
    k = 1000.0
    magnitude = int(math.floor(math.log(number, k)))
    value = number / (k ** magnitude)
    return f"{value:.2f}{units[magnitude]}"

def calculate_mamba2_params(model_config):
    """Calculate and return the number of parameters based on the Mamba2 model configuration."""

    # model hyperparameters
    d_model = model_config['d_model']
    d_state = model_config.get('d_state', 128)
    d_conv = model_config.get('d_conv', 4)
    expand = model_config.get('expand', 2)
    headdim = model_config.get('headdim', 64)
    d_ssm = model_config.get('d_ssm', None)
    ngroups = model_config.get('ngroups', 1)
    D_has_hdim = model_config.get('D_has_hdim', False)
    rmsnorm = model_config.get('rmsnorm', True)
    bias = model_config.get('bias', False)
    conv_bias = model_config.get('conv_bias', True)
    vocab_size = model_config['vocab_size']
    num_layers = model_config['num_layers']

    d_inner = expand * d_model
    if d_ssm is None:
        d_ssm = d_inner
    nheads = d_ssm // headdim

    # Embedding layer
    embedding_params = vocab_size * d_model

    # Mamba2 block
    # in_proj (Linear layer)
    d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
    in_proj_params = d_model * d_in_proj + (d_in_proj if bias else 0)

    # Conv1d (Conv1d layer), Selective scan
    conv_dim = d_ssm + 2 * ngroups * d_state
    conv1d_params = d_conv * conv_dim * (conv_dim // conv_dim) + (conv_dim if conv_bias else 0)

    # dt_bias (Parameter)
    dt_bias_params = nheads

    # A_log (Parameter)
    A_log_params = nheads

    # D (Parameter)
    D_params = d_ssm if D_has_hdim else nheads

    # RMS norm
    norm_params = d_ssm if rmsnorm else 0

    # out_proj (Linear layer)
    out_proj_params = d_inner * d_model + (d_model if bias else 0)

    # Total parameters for a single layer
    layer_params = (
        in_proj_params + conv1d_params + dt_bias_params +
        A_log_params + D_params + 1.5 * norm_params + out_proj_params
    )

    # final RMS normalization
    norm_f_params = d_model if rmsnorm else 0

    # Total
    total_params = embedding_params + num_layers * layer_params + norm_f_params

    return total_params


def find_optimal_num_layers(model_config, baseline, fixed_d_model):
    
    # model hyperparameters
    d_model = model_config['d_model']
    expand = model_config.get('expand', 2)
    headdim = model_config.get('headdim', 64)
    
    best_diff = float('inf')
    optimal_num_layers = None

    num_layers_range = range(1, 1000)

    for num_layers in num_layers_range:
        model_config['d_model'] = fixed_d_model
        model_config['num_layers'] = num_layers

        num_params = calculate_mamba2_params(model_config)
        diff = abs(num_params - baseline)

        # constraint = d_model * expand / headdim % 8 == 0
        if diff < best_diff:
            best_diff = diff
            optimal_num_layers = num_layers

    return best_diff, optimal_num_layers


model_config = {
    'd_model': 512,
    'num_layers': 224,
    'vocab_size': 65024,
    'd_state': 128,
    'd_conv': 4,
    'expand': 2,
    'headdim': 32,
    'd_ssm': None,
    'ngroups': 1,
    'D_has_hdim': False,
    'rmsnorm': True,
    'bias': False,
    'conv_bias': True,
}

baseline = 802795776
fixed_d_model = 768
best_diff, optimal_num_layers = find_optimal_num_layers(model_config, baseline, fixed_d_model)

print(f"Target n_params: {baseline}")
print(f"optimal d_model: {fixed_d_model}")
print(f"Closest num_layers: {optimal_num_layers}")
print(f"Best diff in %: {best_diff / baseline:.2f} %")

model_config["d_model"] = fixed_d_model
model_config["num_layers"] = optimal_num_layers

num_params = calculate_mamba2_params(model_config)
print(f"Model has {human_format(num_params)} parameters")