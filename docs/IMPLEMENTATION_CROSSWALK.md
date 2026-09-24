# Submission-to-source crosswalk

The submission PDF is not distributed with this artifact. The companion
submitted-results JSON supplies the submitted numeric baseline. This table
separately links every claim that can be verified from retained source or
completed-run records. Missing archival inputs are not replaced with older
drafts.

| Submission item | Frozen implementation or evidence | Repository entry | Integrated | Verification |
|---|---|---|---|---|
| HuBERT Base, ninth block, mean pool, 768-D | submitted method record; cache contract `(N,50,768)` and Stage-1 temporal mean | `dataset.py`, `model_stage1.py`, `configs/stage1.yaml` | partial | shape/dtype/alignment tests; extractor revision and resampling provenance pending |
| EEG epoch `-100` to `900` ms | submitted-results record only | current-method docs | partial | frozen retained generation source proves only a 256-sample cache; event anchor and endpoints remain archival pending |
| Four-layer, eight-head, 256-D encoder; 1024-D FFN | shared MAE source | `model_MAE.py` | yes | architecture unit test |
| EEG/EMG patch sizes 16/25 | Stage-2 model source | `model_stage2.py` | yes | constructor inspection/test |
| Full / NoDistill / two shuffles | clean task chain | `chain.py`, four Stage-2 configs | yes | 60/3,600 deterministic manifests |
| Instance-only alpha=0 | clean control source and final success audit | `stage2_instance_only_alpha0.yaml`, `chain.py` | yes | config and loss-target tests |
| LabelSimplex | frozen control source and final success audit | `label_simplex.py`, LabelSimplex configs | yes | prototype hash, no-EMG, initialization and boundary tests |
| Unified six-condition matrix | independent completed run families | `generate_protocol_manifests.py` | yes | exact 90/5,400 grid tests; shared Stage 1 counted once |
| Config-E selection | validation-only eight-candidate pilot | `stage3_validation_search.yaml`, `stage3.yaml` | yes | candidate/frozen-winner test; no test metric in selection config |
| Three-seed subject isolation | clean subject-isolated final success audit | `subject_isolated.yaml`, `generate_subject_isolated_manifests.py`, `run_manifest_task.py` | yes | 15/30/1,800 grids, six-target exclusion, separate final-test role |
| EEGNet-8,2 | frozen baseline source | EEGNet model, manifest, task and aggregator scripts | yes | architecture and deterministic 900-task manifest tests |
| 28-channel removal | frozen channel task and audit | `secondary.py`, `generate_secondary_manifests.py` | yes | exact 60/32/28 montage and 600-task grid tests |
| SmoothGrad-Abs | frozen task source | `secondary.py`, secondary manifest generator | yes | true-class/16/noise/order assertions and 600-task grid |
| Fig. 2 v7 | locked v7 design plus fresh clean regeneration source | `make_fig2_v7.py` | parameterized | fixed canvas/scale/contour/font tests; no result array or PDF shipped |
| EMG-only and EEG+EMG fusion | existing secondary interface | `run_modality_baselines.py`, `secondary.py` | yes | CLI help/import tests; no unsupported task count or significance claim |
| Submitted EEGNet values | author-supplied submission baseline | README and expected-results document | values retained | exact author-final subject source archival pending; older run excluded |

Historical filenames containing `audio`, Wav2Vec2-compatible processor names,
or old checkpoint keys are not silently rewritten. They are compatibility or
historical interfaces unless a current-method document explicitly says HuBERT.
