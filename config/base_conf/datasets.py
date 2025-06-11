

Suzhou3000ETF_conf = dict(
    dataset_name='Suzhou3000ETF',
    var_num=19,
    freq=10,
    eod_size=3,
    data_split=[31449, 10483, 10484],  # (31+28+31+30+31+30)*24*6, (31+31+30)*24*6, (31+30+31-1)*24*6
    doy2embedding_path="Suzhou3000ETF/event/SuZhou_GLM4_voyage-large-2-instruct.pickle",
    max_event_per_day=3,  # 用于 rescaled time_marker
)
