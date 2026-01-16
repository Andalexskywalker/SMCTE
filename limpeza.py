from utils.data_preprocessing import main  # se preferires, podes nem usar Config

main(args=dict(
    input=r".\csv\train_data.csv",
    output_train=r".\tensors\train_seq.npz",
    output_val=r".\tensors\val_seq.npz",
    output_test=r".\tensors\test_seq.npz",
    task="classification",
    positive_threshold=-6.0,
    stratify_events=True,
    event_target_strategy="any_high_risk",
    clip_limits_json=r".\reports\eda\clip_limits.json",
    sequence_timestep=10,
    sequence_fill=-999.0,
    sequence_format="npz",
    sequence_order="desc"   # <— em vez de "chron"
))
