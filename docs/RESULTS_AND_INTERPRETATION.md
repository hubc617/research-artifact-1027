# Submitted results and interpretation

Values come from the [submitted results record](SUBMITTED_RESULTS_20260919.json).

## Primary results

| Method / 方法 | Stage 2 validation BA (%) | Stage 3 Overt BA (%) | Stage 3 Silent BA (%) |
| --- | --- | --- | --- |
| NoDistill | 41.23 | 49.15 | 38.67 |
| LabelSimplex | 38.40 | 47.00 | 36.94 |
| Unrestricted shuffle | 41.23 | 49.29 | 38.59 |
| Within-class shuffle | 48.11 | 57.09 | 44.72 |
| Instance-only (α=0) | 48.74 | 58.32 | 45.43 |
| Full (α=0.3) | 48.61 | 58.23 | 45.32 |
| Full − NoDistill | +7.38 | +9.08 | +6.65 |

Stage-2 values are pooled checkpoint-selection validation BA. For each seed/fold cell, BA is computed from concatenated validation predictions, the best of 150 validation epochs is selected, and 15 cell values are equally averaged. Stage-3 testing averages folds and seeds within each participant before comparing 30 subject-level values.

| Comparison / 比较 | Overt: Δ pp [95% CI] | Silent: Δ pp [95% CI] | Positive subjects: Overt; Silent |
| --- | --- | --- | --- |
| Full − NoDistill | +9.08 [8.18, 9.98]; p<0.001 | +6.65 [5.65, 7.59]; p<0.001 | 30/30; 30/30 |
| Unrestricted − NoDistill | +0.15; p=0.491 | −0.08; p=0.721 | — |
| Full − Within-class | +1.14 [0.72, 1.54]; p<0.001 | +0.60 [0.18, 1.06]; p=0.012 | 25/30; 19/30 |
| Full − Instance-only | −0.09 [−0.55, 0.37]; p=0.694 | −0.11 [−0.62, 0.39]; p=0.672 | 14/30; 15/30 |
| Full − LabelSimplex | +11.22 [10.30, 12.19]; p<0.001 | +8.38 [7.14, 9.59]; p<0.001 | 30/30; 30/30 |
| Full − EEGNet-8,2 | +4.97 [3.19, 6.62]; p<0.001 | +5.20 [3.40, 6.98]; p<0.001 | — |

The manuscript reports EEGNet means of 53.26% Overt and 40.12% Silent.
Full versus EEGNet compares the complete systems, including Full's pooled
upstream pretraining. Full versus NoDistill compares distillation conditions
under the same upstream training protocol. See the
[EEGNet reproduction note](REPRODUCIBILITY.md#eegnet-statistics).

Within-class shuffling retains 87.5%/90.9% of Full's improvement. The
comparison with Full estimates the contribution of exact EEG–EMG pairing
while preserving subject, paradigm, and lexical grouping. The paired
intervals for Full versus Instance-only span zero in both paradigms.

LabelSimplex gives lower balanced accuracy than NoDistill. The unrounded
LabelSimplex−NoDistill differences are −2.142784/−1.733230 pp, with approximate
95% CIs [−2.8395,−1.4648]/[−2.3502,−1.1374] for the tested fixed target.

## Subject-isolated transfer and adaptation

| Evaluation / 评估 | Full Overt / Silent (%) | NoDistill Overt / Silent (%) | Full−NoDistill Overt / Silent (pp) | 95% CI: Overt; Silent | p: Overt; Silent |
| --- | --- | --- | --- | --- | --- |
| Subject-isolated zero-shot | 23.63 / 20.54 | 23.52 / 20.51 | +0.11 / +0.03 | [−0.53, 0.74]; [−0.57, 0.63] | 0.735; 0.914 |
| EEG-only target adaptation, 3 seeds | 43.33 / 33.09 | 35.98 / 27.62 | +7.35 / +5.47 | [6.29, 8.49]; [4.37, 6.63] | <0.001; <0.001 |

Each of the five outer folds holds out six participants from upstream
training and validation. Zero-shot evaluation uses the pretrained classifier
directly. Adaptation uses labeled target EEG to train the encoder and a new
classifier. Both evaluations use three seeds. After adaptation, Full
improves on NoDistill for 30/30 Overt and 29/30 Silent participants.

## Channels, attribution, and modality context

| Stage 3 condition / 条件 | Full Overt / Silent (%) | NoDistill Overt / Silent (%) | Full−NoDistill Overt / Silent (pp) |
| --- | --- | --- | --- |
| All 60, seed 42 | 58.12 / 45.47 | 49.10 / 38.43 | +9.03 / +7.04 |
| Remaining 28, seed 42 | 34.42 / 26.12 | 27.52 / 21.52 | +6.90 / +4.60 |
| Attenuation interaction | — | — | −2.13 / −2.44 |

The seed-42 comparison covers the same 30 participants and five folds per
paradigm. It compares all-channel Stage 3 with Stage 3 trained, validated,
and tested with a fixed zero mask on 32 of the 60 channels. Both conditions
use all-channel Stage 2 checkpoints. See the
[channel mask](ARCHITECTURE_AND_PROTOCOL.md#channel-mask) for the channel lists.

For participant s, the attenuation interaction is:

$$I_s=(BA_{Full,28,s}-BA_{No,28,s})-(BA_{Full,60,s}-BA_{No,60,s}).$$

Mean interactions are −2.13/−2.44 pp, with p=0.0109/0.0012. Remaining-channel
gain intervals are [5.58,8.33]/[3.56,5.64], with 29/30 and 28/30 positive
participants. The negative interactions quantify the reduction in the
distillation benefit under channel masking.

[SmoothGrad-Abs](ARCHITECTURE_AND_PROTOCOL.md#smoothgrad-and-fig-2) measures
the share of each seed-42 model's normalized true-class logit attribution
assigned to each EEG channel. Outer-region Full−NoDistill differences are
0.0136142353/0.0132848302, or +1.36/+1.33 percentage points of attribution
mass, positive in 30/30 participants in each paradigm. The share also
increases at Cz and CPz. Fig. 2 maps the channel-level differences.

EMG-only uses a 256–256–10 head on the Stage 1 EMG CLS token. Fusion
concatenates the 256-D EEG and EMG CLS tokens for a 512–256–10 head. EMG-only
obtains 81.71%/73.11% and EEG+EMG obtains 84.84%/76.27%. Both use EMG at
inference. The observed fusion increments are +3.13/+3.16 pp.

## Statistical definitions and rounding

For each paradigm, the analysis averages the applicable folds and seeds
within each of the 30 participants and forms paired differences. The paired
percentile bootstrap resamples participants with replacement 50,000 times
and takes the 2.5th and 97.5th percentiles of the mean differences.

The two-sided sign-flip test assigns random signs to the participant
differences in 200,000 draws and evaluates the absolute mean-difference tail.
The manuscript reports unadjusted p values. See the
[public statistics script](REPRODUCIBILITY.md#paired-participant-statistics)
for its input and output conventions.

Compute differences and retention ratios from unrounded estimates, then round for display. For example, the unrounded Overt Unrestricted−NoDistill difference is approximately 0.145293 pp and displays as +0.15, although 49.29−49.15 is 0.14 using the rounded table. The Stage-2 Full−NoDistill difference is 7.38087079 pp and displays as +7.38.
