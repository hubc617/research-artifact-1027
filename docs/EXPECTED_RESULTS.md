# Audited result contract

The authoritative clean-main grid contains 3,600 Stage-3 tasks: four variants,
three seeds, five folds, 30 subjects, and two paradigms. The audited primary
balanced-accuracy means (%) are:

| Variant | Overt | Silent | Stage-2 validation |
|---|---:|---:|---:|
| Full | 58.225573 | 45.322136 | 48.61 |
| NoDistill | 49.145955 | 38.670686 | 41.23 |
| Unrestricted Shuffled-EMG | 49.291248 | 38.590122 | 41.23 |
| Within-class Shuffled-EMG | 57.090016 | 44.719407 | 48.11 |

The independently completed InstanceOnlyAlpha0 grid contains 900 tasks and
gives `58.319822338156/45.433989590656` Overt/Silent BA and Stage-2 pooled
validation BA `48.736331036914`. Its Full differences are `-0.094249084249`
and `-0.111853516854` pp, with paired confidence intervals crossing zero.

The completed EEG-only LabelSimplex control contains 15 Stage-2 cells and 900
Stage-3 tasks. Stage-2 pooled validation BA is `38.396029720596%`; Stage-3 BA
is `47.00317084150417/36.937455199121864%` Overt/Silent. Full minus
LabelSimplex is `+11.222402412402413 pp` (95% CI
`[10.296485084360084, 12.185968061259734]`) Overt and
`+8.384680874680875 pp` (95% CI
`[7.1356404690988, 9.592694732736403]`) Silent. Both two-sided sign-flip p
values are below `0.001`, with Full higher for 30/30 subjects per paradigm.
The fixed label geometry did not reproduce the EMG-teacher benefit, but this
does not rule out every possible label-only regularizer.

The clean Stage-2 pooled-validation values are NoDistill `41.23`, unrestricted
shuffle `41.23`, within-class shuffle `48.11`, Full `48.61`, InstanceOnlyAlpha0
`48.74`, and LabelSimplex `38.40`; Full minus NoDistill is `+7.38 pp`.

The new subject-isolated grid passed all 15 Stage-1, 30 Stage-2, and 1,800
Stage-3 task gates. Three-seed zero-shot Full/NoDistill BA is `23.63/23.52`
Overt and `20.54/20.51` Silent. Adapted BA is `43.33/35.98` Overt and
`33.09/27.62` Silent. Adapted Full minus NoDistill is `+7.35 pp` (95% CI
`[6.29, 8.49]`, 30/30 positive) Overt and `+5.47 pp` (95% CI
`[4.37, 6.63]`, 29/30 positive) Silent; both sign-flip p values are below
`0.001`.

The submitted EEGNet values are retained unchanged: means `53.26/40.12`,
Full differences `+4.97/+5.20 pp`, CIs `[3.19, 6.62]/[3.40, 6.98]`, and
two-sided sign-flip `p < 0.001`. Their exact author-final subject-level source
and calculation script remain `EXACT_SOURCE_ARCHIVAL_PENDING`. The older
complete `54.53/41.17` run is not substituted, renamed, or used to recompute
the submitted statistics.
