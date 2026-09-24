# Submitted results and interpretation

Extracted from the final submitted-manuscript dossier; exact-source archival requirements remain explicit.

### 6. Primary results

| Method / 方法 | Stage 2 validation BA (%) | Stage 3 Overt BA (%) | Stage 3 Silent BA (%) |
| --- | --- | --- | --- |
| NoDistill | 41.23 | 49.15 | 38.67 |
| LabelSimplex | 38.40 | 47.00 | 36.94 |
| Unrestricted shuffle | 41.23 | 49.29 | 38.59 |
| Within-class shuffle | 48.11 | 57.09 | 44.72 |
| Instance-only (α=0) | 48.74 | 58.32 | 45.43 |
| Full (α=0.3) | 48.61 | 58.23 | 45.32 |
| Full − NoDistill | +7.38 | +9.08 | +6.65 |


Stage-2 values are pooled checkpoint-selection validation BA. For each seed/fold cell, BA is computed from concatenated validation predictions, the best of 150 validation epochs is selected, and 15 cell values are equally averaged. They are not independent test estimates. Stage-3 testing averages folds and seeds within each participant before comparing 30 subject-level values.

| Comparison / 比较 | Overt: Δ pp [95% CI] | Silent: Δ pp [95% CI] | Positive subjects: Overt; Silent |
| --- | --- | --- | --- |
| Full − NoDistill | +9.08 [8.18, 9.98]; p<0.001 | +6.65 [5.65, 7.59]; p<0.001 | 30/30; 30/30 |
| Unrestricted − NoDistill | +0.15; p=0.491 | −0.08; p=0.721 | — |
| Full − Within-class | +1.14 [0.72, 1.54]; p<0.001 | +0.60 [0.18, 1.06]; p=0.012 | 25/30; 19/30 |
| Full − Instance-only | −0.09 [−0.55, 0.37]; p=0.694 | −0.11 [−0.62, 0.39]; p=0.672 | 14/30; 15/30 |
| Full − LabelSimplex | +11.22 [10.30, 12.19]; p<0.001 | +8.38 [7.14, 9.59]; p<0.001 | 30/30; 30/30 |
| Full − EEGNet-8,2 | +4.97 [3.19, 6.62]; p<0.001 | +5.20 [3.40, 6.98]; p<0.001 | not specified / 未列出 |


Positive-subject counts belong to specific comparisons and paradigms. The selected EEGNet values are 53.26%/40.12%, with the submitted intervals retained. Do not reuse the older 54.53%/41.17% run or its confidence intervals. Full versus EEGNet is a system comparison because EEGNet lacks the same pooled upstream pretraining; Full versus NoDistill is the principal controlled distillation comparison.

Within-class shuffling retains 87.5%/90.9% of Full's improvement. Exact pairing provides a smaller incremental benefit, with weaker evidence for Silent than for the primary comparison. Full versus Instance-only has confidence intervals spanning zero: no incremental prototype benefit was detected, but equivalence was not established. Keep Full as the prespecified complete model while transparently reporting Instance-only; do not relabel a test-selected numerical winner as the original primary model.

LabelSimplex is below NoDistill. The supplied report gives unrounded LabelSimplex−NoDistill differences of −2.142784/−1.733230 pp, with approximate 95% CIs [−2.8395,−1.4648]/[−2.3502,−1.1374]. This supports a limitation of the tested fixed target geometry, not a rejection of all label-based representation learning. Optimization conflict, geometry choice, and hyperparameter suitability were not separately identified as causes.

### 7. Subject-isolated transfer and adaptation

| Evaluation / 评估 | Full Overt / Silent (%) | NoDistill Overt / Silent (%) | Full−NoDistill Overt / Silent (pp) | 95% CI: Overt; Silent | p: Overt; Silent |
| --- | --- | --- | --- | --- | --- |
| Subject-isolated zero-shot | 23.63 / 20.54 | 23.52 / 20.51 | +0.11 / +0.03 | [−0.53, 0.74]; [−0.57, 0.63] | 0.735; 0.914 |
| EEG-only target adaptation, 3 seeds | 43.33 / 33.09 | 35.98 / 27.62 | +7.35 / +5.47 | [6.29, 8.49]; [4.37, 6.63] | <0.001; <0.001 |


After adaptation, improvement is positive for 30/30 Overt and 29/30 Silent participants. These three-seed results support better initialization for supervised EEG-only target adaptation after target-isolated upstream learning. They do not demonstrate improved direct zero-shot decoding or calibration-free transfer.

The subject-dependent 58.23%/45.32% and subject-isolated adapted 43.33%/33.09% values arise under different information conditions. They should not be merged or interpreted as the same experiment losing accuracy. Earlier zero-shot summaries such as 23.89%/24.05% are superseded by this final three-seed record.

### 8. Channels, attribution, and modality context

| Stage 3 condition / 条件 | Full Overt / Silent (%) | NoDistill Overt / Silent (%) | Full−NoDistill Overt / Silent (pp) |
| --- | --- | --- | --- |
| All 60, seed 42 | 58.12 / 45.47 | 49.10 / 38.43 | +9.03 / +7.04 |
| Remaining 28, seed 42 | 34.42 / 26.12 | 27.52 / 21.52 | +6.90 / +4.60 |
| Attenuation interaction | — | — | −2.13 / −2.44 |


The channel experiment retrains Stage 3 with the same fixed zero mask during training, validation, and testing. It preserves the 60-channel tensor and uses unchanged all-channel Stage-2 checkpoints. It is not inference-only masking and not a fully 28-channel upstream pipeline. All-channel references are matched seed-42 results, not three-seed averages.

Retained 28 channels:

`F3, F1, FZ, F2, F4, FC3, FC1, FCZ, FC2, FC4, C3, C1, CZ, C2, C4, CP3, CP1, CPZ, CP2, CP4, P3, P1, PZ, P2, P4, PO3, POZ, PO4`

Zeroed 32 channels:

`FP1, FPZ, FP2, AF3, AF4, F7, F5, F6, F8, FT7, FC5, FC6, FT8, T7, C5, C6, T8, TP7, CP5, CP6, TP8, P7, P5, P6, P8, PO7, PO5, PO6, PO8, O1, OZ, O2`

Map names to the actual input montage and verify ordered-mask identity; do not assume the written list is the tensor order.

For participant s, the attenuation interaction is:

$$I_s=(BA_{Full,28,s}-BA_{No,28,s})-(BA_{Full,60,s}-BA_{No,60,s}).$$

Mean interactions are −2.13/−2.44 pp, with p=0.0109/0.0012. Remaining-channel gain intervals are [5.58,8.33]/[3.56,5.64], with 29/30 and 28/30 positive participants. Negative interactions indicate reduced distillation benefit under channel removal, not physiological source localization.

SmoothGrad-Abs analyzes the true-class logit of seed-42 Full/NoDistill models using 16 noisy samples per trial and noise SD=0.1×trial EEG SD. Each channel-by-time map is L1-normalized before participant aggregation. Outer-region Full−NoDistill changes are 0.0136142353/0.0132848302, or +1.36/+1.33 percentage points of attribution mass, positive in 30/30 participants. Central-midline attribution also increases at Cz and CPz. Fig. 2 uses the final grayscale-safe v7 plotting design. These are relative attribution redistributions, not absolute gradient increases or proof of cortical origin.

EMG-only uses a 256–256–10 head on Stage-1 EMG CLS; fusion concatenates 256-D EEG and EMG CLS representations for a 512–256–10 head. EMG-only obtains 81.71%/73.11%; EEG+EMG obtains 84.84%/76.27%. Both require EMG at inference. The +3.13/+3.16-pp fusion increments provide modality context; without a corresponding verified paired test, describe higher observed performance rather than proven complementary mechanisms.

### 9. Statistical definitions and rounding

For each paradigm, form 30 paired participant differences dₛ after averaging the appropriate folds and seeds. Report mean(dₛ). The paired percentile bootstrap samples participants with replacement 50,000 times, retaining the pairing, and uses the 2.5th/97.5th percentiles of bootstrap mean differences. Repeated seed/fold cells are not additional independent participants.

The 200,000-draw two-sided sign-flip test randomly assigns ± signs to these paired differences and evaluates the absolute mean-difference tail. Archive the random seed and exact counting convention, including whether the plus-one correction is used. A CI and a p value describe related but different inferential quantities; neither can be recovered exactly from the reported mean alone.

The manuscript reports unadjusted p values. If multiplicity is questioned, define and disclose the hypothesis family before presenting a sensitivity analysis; do not select a correction family merely to preserve significance. A nonsignificant comparison does not establish equivalence.

Compute differences and retention ratios from unrounded estimates, then round for display. For example, the unrounded Overt Unrestricted−NoDistill difference is approximately 0.145293 pp and displays as +0.15, although 49.29−49.15 is 0.14 using the rounded table. The Stage-2 Full−NoDistill difference is 7.38087079 pp and displays as +7.38.
