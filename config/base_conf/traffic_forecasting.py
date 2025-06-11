task_conf = dict(
    hist_len=12,
    pred_len=12,
    dm="dm_multivariate",
    runner="traffic_forecasting_runner",

    batch_size=64,
    max_epochs=100,
    lr=0.002,
    optimizer="Adam",
    optimizer_weight_decay=0.00001,
    lr_scheduler='ReduceLROnPlateau',
    lrs_factor=0.5,
    lrs_patience=3,
    gradient_clip_val=5,
    val_metric="val/loss",
    null_value=0.0,
    test_metric="test/mae",
    es_patience=5,

    num_workers=2,
)
