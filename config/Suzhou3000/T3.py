exp_conf = dict(
    model_name="T3STAEformer",
    dataset_name="Suzhou3000ETF",
    task="traffic_forecasting",

    input_dim=3,
    output_dim=1,
    input_embedding_dim=24,
    tod_embedding_dim=24,
    dow_embedding_dim=24,
    spatial_embedding_dim=0,
    adaptive_embedding_dim=80,
    feed_forward_dim=512,
    num_heads=4,
    num_layers=3,
    use_mixed_proj=True,

    doy2embedding_path="/home/pcl/80T/projects/ETF_TKDE/mm_easytsf/dataset/Shenzhen3000ETF/Shenzhen_ChatGPT-4o-1_voyage-3.pickle",
    event_embedding_dim=1024,
    event_hidden_dim=120,
    pretrained_model_path="/home/pcl/80T/projects/ETF_TKDE/mm_easytsf/Benchmark/STAEformer_Suzhou3000ETF/8925e041ba/seed_3/checkpoints/epoch=42-step=84452.ckpt",
    dropout=0.2,

    lr=1e-4,
)