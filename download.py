from pyecotaxa.transfer import Transfer, ProgressListener

# (.*)\s+\[(\d+)\].*  => $2, # $1

PROJECT_IDS = [
    1085,  # LOKI_PS106.2
    5198,  # LOKI_PS106.2-78-3
    4229,  # LOKI_PS106.2_49+62+65+67+71+73
    4566,  # LOKI_PS106.2_74-3
    4621,  # LOKI_PS106.2_75-2
    4821,  # LOKI_PS106.2_76-2
    4827,  # LOKI_PS106.2_88-4
    4834,  # LOKI_PS106.2_91-3
    4871,  # LOKI_PS106.2_93-2
    939,  # LOKI_PS107
    1246,  # LOKI_PS114/4
    2925,  # LOKI_PS122_MOSAiC
    1607,  # LOKI_PS93.2
    368,  # LOKI_PS93.2/58
    362,  # LOKI_PS93.2/80
    376,  # LOKI_PS93.2/82
    1469,  # LOKI_PS94
    1622,  # LOKI_PS99.2
    5280,  # LOKI_SO21
    5008,  # LOKI_PS122_MOSAiC_without-zoomie
]

t = Transfer()
progress_listener = ProgressListener()
t.register_observer(progress_listener.update)
t.pull(
    PROJECT_IDS,
    target_directory="input",
    n_parallel=8,
    check_integrity=False,
    cleanup_task_data=False,
)
