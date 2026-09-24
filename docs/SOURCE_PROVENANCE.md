# Source provenance

This refresh was reconstructed from source files named by successful
clean-20260811 audits. Server paths, manifests containing restricted paths,
checkpoints, task results, predictions, and arrays were not copied.

Key upstream SHA-256 evidence:

| Scope | Upstream source | SHA-256 |
|---|---|---|
| Shared MAE | `model_MAE.py` | `185a46f0e0f65e47d077f8ec21b2a6c171d6c66b13a72b20857d86783a54ee3c` |
| Stage 1 | `model_stage1.py` | `b33b8f234b49e76c8b2179a43ebd477868ea8d629638b311cf042bfb915b0f25` |
| Stage 2 | `model_stage2.py` | `821fdf7c4e736c932c1e45018229ecc004d402f43d6da4cdcae6b48406b18725` |
| Stage 3 | `model_stage3.py` | `888b9ceef3c1c2ca1a14e8bb8fdf92a0eb96ec8643d0f893b29be74a01d65d29` |
| Clean chain | `clean_sd5_chain.py` | `b9bdf1606b4579c6bd81b64c536b14953de7113ebdff68e07df0cdd47d2a81e5` |
| Clean Stage-2 task | `stage2_task.py` | `cc58137460520efdf6f5e48b58e70804365803057147d9c5a60479f7694a31e9` |
| Clean final task | `final_task.py` | `311ce66c82e2e7d9ef94ebcab3e337ae161b74b9ded0c20e34959f54fb50eec0` |
| Clean final audit | `audit_final.py` | `ac8536b3f654904c49c60f13676294f6b9d79212bf7599a95f0c13d8629f3075` |
| InstanceOnlyAlpha0 Stage 2 | `stage2_task.py` | `51a6a6f11cb4b7284e5c50911c55f2ed475cbd73c954ecfc04cfccfcfd7ec003` |
| InstanceOnlyAlpha0 audit | `audit_stage3.py` | `487446119f81b8a559ab6e3e09c6f23d629ab1f99453bbb19094ff9fde594037` |
| LabelSimplex prototype generator | `generate_prototypes.py` | `7ec6cecfe5cbd535373529fb9d8852d283fdc512411d56e7f3b3916e82adfb71` |
| LabelSimplex EEG-only loader | `eeg_only_data.py` | `bca026d4712d07c0f56a12ad7230bb1725c169b9e2a79fffe8c04a6986ab671d` |
| LabelSimplex EMG guard | `emg_access_guard.py` | `7b917f180271df29b078e8ff6c532f59ebdc858750d18553d3575e72e628390b` |
| LabelSimplex Stage 2 model | `label_simplex_model.py` | `e78859771ab265996d0199f4b6da89b796940d34990f88d0fbb6b7f6db11c103` |
| LabelSimplex Stage 2 task | `stage2_task.py` | `d7ba96ea24f6b8d6826c394fdea0c6ea4032dc5a6e245ad79fff88b7eae7ff96` |
| LabelSimplex Stage 3 wrapper | `stage3_task.py` | `941cbbda25407abf197003c2d09ce943e8d6057f846816f540e56c049534cd02` |
| EEGNet-8,2 model | `eegnet_model.py` | `a52dd2818c91c78999b228fa6b7c82cff11a3ba3ffa1f6c3c764418ba6530530` |
| EEGNet task | `train_eegnet_task.py` | `c597e62f9eac3b185e6d2b38d591351c7e0806cd38705149d1ea523208472cea` |
| Channel removal | `channel_task.py` | `9dc49a4f59c985e62c4ab28adc804db176840d252d97bf62a1f58d6517c6600c` |
| SmoothGrad-Abs | `smoothgrad_task.py` | `c4bb581861418dc32b1749c99dc9ab76beed2c68c0ebc7af919542c328c35309` |
| Fig. 2 v7 | `make_fig2_grayscale_safe_v7.py` | `45f5853c4d1108ba30b39106d63d85fc2ed2e20c72b43c06891b12255e834b27` |

The original v7 plotting source hard-coded historical inputs and annotations.
The repository entry is a parameterized source-only implementation: it keeps
the fixed canvas, Type-42 font policy, signed contour styles, montage ordering,
and fixed color scale, but requires fresh channel and annotation summaries.
It intentionally does not reproduce the operational file hash.

The repository code is intentionally cleaned and parameterized, so its hashes
need not equal the operational scripts. Scientific invariants are locked in
the YAML configs, tests, and documented execution order.

## Result sources used by README

| README values | Audited aggregate or marker | SHA-256 |
|---|---|---|
| Four-variant Stage-3 means | `four_variant_summary.csv` | `9c58dfa64925138befcc422b5c635a8bd49f79cea5e9d38c998c8096805bc7c2` |
| Four-variant paired statistics | `paired_subject_statistics.csv` | `e43c8e763f82a122bb84144fa96dd29226de09362114460ef03d7d134837af8b` |
| Four-variant final success | `CLEAN_20260811_FOUR_VARIANT_SUCCESS.json` | `d26a3cb1dc91d6b66979850c279af1a0b9be07ec9e2c49a3e8fa5bc3837c40c1` |
| Full/NoDistill/Unrestricted Stage-2 Aux. Val. | `clean_sd5_3seed_stage2_overall_summary_final_v1.csv` | `f7ef72f5ea942c91159ccf8bab3dc956613c86a187ae827c535898705e0cbb95` |
| Audited 60-cell Stage-2 matrix, including Within-class | `stage2_matrix_60.csv` | `9333ad29b080af5ea63927e18182a3eb1a8941ac936089ad3b14755f68df6137` |
| InstanceOnlyAlpha0 Stage-2 | `stage2_validation_summary.csv` | `05c07dbb695e4097585023dd7f1fd735d74ec742e8ea1e4587af8ca4a923fffe` |
| InstanceOnlyAlpha0 Stage-3 | `instance_only_alpha0_summary.csv` | `86b97b7246226b46b679ae8b5da37ecdbee7fa3ecbd3a2ccd4aea1d4394cd62a` |
| InstanceOnlyAlpha0 paired statistics | `paired_subject_statistics.csv` | `de8417a1b4a6cbd03ac506d0911d45c4fa746cca90f01b54f27ce1cc8349c43b` |
| InstanceOnlyAlpha0 final success | `FINAL_SUCCESS.json` | `2905e755f8adf43c62837e63bb441208e1db6cf0f3990fe700526cd07a222094` |
| LabelSimplex Stage-2 summary | `stage2_validation_summary.csv` | `5a041189f8b89c4aa27e9f898363ac62b0498ef2334bd05dec2f92f071616fbe` |
| LabelSimplex Stage-3 summary | `label_simplex_summary.csv` | `3389e6b2e8bac7a8840c0531b79a3fd54d222149270387ce10bae4ac9e273095` |
| LabelSimplex paired statistics | `paired_subject_statistics.csv` | `c34f230214aca2fb94ee6112b333428cf0fa5c5cac38c7c688da956709474587` |
| LabelSimplex final success | `FINAL_SUCCESS.json` | `c15ce55a7c57080f949f6cd39ddf9ee9be66217a16951fa947bb15cf35233ce3` |
| Subject-isolated three-seed summary | `three_seed_adaptation_summary.csv` | `32fd3a8074153cea63c5169cff3cda98937911a6d437174960916568fd977a2d` |
| Subject-isolated paired statistics | `paired_subject_statistics.csv` | `58563f8f077dcbab52d4575bfaf0394b596440e1cbd30b32bb99141d8c0648ae` |
| Subject-isolated final success | `FINAL_SUCCESS.json` | `b40f0911e68b2f12a8a9ffeaf83f382a48712596d521eb0d729635262593c621` |

The submitted EEGNet values are not assigned a result hash because their exact
author-final subject-level source is unresolved; see the dedicated provenance
audit. The numbers and inferential statistics remain frozen to the submission,
with exact-source archival explicitly pending.
