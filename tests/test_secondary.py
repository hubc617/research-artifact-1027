from privileged_emg.secondary import EEG_CHANNELS, OUTER_RING_WIDE, parser


def test_channel_removal_preset_leaves_28_channels():
    assert len(EEG_CHANNELS) == 60
    assert len(OUTER_RING_WIDE) == 32
    assert len(EEG_CHANNELS) - len(OUTER_RING_WIDE) == 28


def test_secondary_parser_requires_real_arguments():
    args = parser().parse_args(["fig2", "--inputs", "a.csv", "--output", "figure.png"])
    assert args.kind == "fig2"
