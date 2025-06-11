exp_conf = dict(
    model_name="STIDEF",
    dataset_name="Suzhou3000ETF",
    task="traffic_forecasting",

    block_num=3,
    ts_emb_dim=32,
    node_emb_dim=32,
    tod_emb_dim=32,
    dow_emb_dim=32,
    eod_emb_dim=32,

    lr=0.01  # search from [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
)