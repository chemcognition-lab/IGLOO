from src.igloo.models import AttentionConvPoolingModel, ESMAttention1d, LinearRegressionModel, MeanPoolingModel


def create_model(config, device):
    """Create a model from configuration dictionary."""
    input_dim = config['input_dim']
    L = config['L']
    pooling_method = config['pooling_method']

    if pooling_method == 'mean':
        model = MeanPoolingModel(
            L, input_dim,
            hidden_dim1=config['hidden_dim1'],
            hidden_dim2=config['hidden_dim2'],
            hidden_dim3=config['hidden_dim3'],
            hidden_dim4=config['hidden_dim4'],
            dropout_rate=config['dropout_rate'],
            final_activation=config['final_activation'],
            target_min=None, target_max=None
        )
    elif pooling_method == 'attention':
        model = AttentionConvPoolingModel(
            L, input_dim,
            conv_dim=256, kernel_size=5,
            hidden_dim1=config['hidden_dim1'],
            hidden_dim2=config['hidden_dim2'],
            hidden_dim3=config['hidden_dim3'],
            hidden_dim4=config['hidden_dim4'],
            dropout_rate=config['dropout_rate'],
            final_activation=config['final_activation'],
            target_min=None, target_max=None
        )
    elif pooling_method == 'linear':
        model = LinearRegressionModel(
            L, input_dim,
            final_activation=config['final_activation'],
            target_min=None, target_max=None
        )
    elif pooling_method == 'esm_attention':
        model = ESMAttention1d(max_length=L, d_embedding=input_dim)
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")

    return model.to(device)
