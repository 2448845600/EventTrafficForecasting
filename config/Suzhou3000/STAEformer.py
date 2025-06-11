exp_conf = dict(
    model_name="STAEformer",
    dataset_name="Suzhou3000ETF",
    task="traffic_forecasting",

    input_dim=3,
    output_dim=1,
    input_embedding_dim=24,
    tod_embedding_dim=24,
    dow_embedding_dim=24,
    spatial_embedding_dim=0,
    adaptive_embedding_dim=80,
    feed_forward_dim=256,
    num_heads=4,
    num_layers=3,
    dropout=0.1,
    use_mixed_proj=True,

    lr=0.001,
    batch_size=16,
    max_epoch=30,
)