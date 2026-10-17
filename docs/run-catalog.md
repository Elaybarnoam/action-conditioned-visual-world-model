# Complete frozen run catalog

Every retained run from run-001 through run-125 is represented. Files include
all saved checkpoint states, numerical predictions, plots, scores, diagnostics
and operational metadata. `running` viewer records describe captured live
sessions. Failed records remain failed even when later recovery succeeds.

Use [artifact instructions](history-artifacts.md) to restore a run. The
[checkpoint registry](../results/checkpoint-registry.csv) distinguishes training
states from other PT tensors. Download sizes and original/public file identities
are pinned by the [history manifest](../results/history-manifest.json).

## Research stages

## Data, encoder and memory

### run-001

Fetch original BAIR training/test TFRecord shards.

Status: **completed**. Kind: `dataset_fetch`. Saved files: **6**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-001`

### run-002

Fetch and verify BAIR image/action sequences.

Status: **completed**. Kind: `dataset_verification`. Saved files: **14**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-002`

### run-003

Trace BAIR images through the untrained CNN.

Status: **completed**. Kind: `encoder_inspection`. Saved files: **11**; training-directory checkpoints: **1**; all PT files: **1**.

`python -m src.history fetch --run run-003`

### run-004

Trace real observed history into latent memory.

Status: **completed**. Kind: `memory_inspection`. Saved files: **9**; training-directory checkpoints: **1**; all PT files: **1**.

`python -m src.history fetch --run run-004`

## Initial joint training

### run-005

First joint one-step visual world-model experiment.

Status: **failed**. Kind: `training`. Saved files: **312**; training-directory checkpoints: **12**; all PT files: **12**.

`python -m src.history fetch --run run-005`

### run-006

Checkpoint history-gradient reproduction.

Status: **failed**. Kind: `diagnosis`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-006`

### run-007

First joint one-step visual world-model experiment.

Status: **completed**. Kind: `training`. Saved files: **335**; training-directory checkpoints: **13**; all PT files: **13**.

`python -m src.history fetch --run run-007`

[Published stage results](../results/milestones/run-007.json).

### run-008

Frozen checkpoint comparison on test.

Status: **completed**. Kind: `evaluation`. Saved files: **67**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-008`

[Published stage results](../results/milestones/run-008.json).

### run-009

Checkpoint history-gradient reproduction.

Status: **completed**. Kind: `diagnosis`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-009`

## Temporal and action diagnosis

### run-010

World-model diagnosis: all.

Status: **failed**. Kind: `experiment-suite`. Saved files: **10**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-010`

### run-011

Tiny fixed-window overfit.

Status: **completed**. Kind: `controlled-training`. Saved files: **2174**; training-directory checkpoints: **10**; all PT files: **10**.

`python -m src.history fetch --run run-011`

### run-012

history diagnostic.

Status: **completed**. Kind: `frozen-diagnostic`. Saved files: **28**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-012`

### run-013

one_step_control.

Status: **completed**. Kind: `controlled-training`. Saved files: **332**; training-directory checkpoints: **13**; all PT files: **13**.

`python -m src.history fetch --run run-013`

### run-014

multi_step.

Status: **completed**. Kind: `controlled-training`. Saved files: **332**; training-directory checkpoints: **13**; all PT files: **13**.

`python -m src.history fetch --run run-014`

### run-015

history diagnostic.

Status: **completed**. Kind: `frozen-diagnostic`. Saved files: **28**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-015`

### run-016

history diagnostic.

Status: **completed**. Kind: `frozen-diagnostic`. Saved files: **28**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-016`

### run-017

actions diagnostic.

Status: **completed**. Kind: `frozen-diagnostic`. Saved files: **30**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-017`

### run-018

actions diagnostic.

Status: **completed**. Kind: `frozen-diagnostic`. Saved files: **30**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-018`

### run-019

no_actions.

Status: **failed**. Kind: `controlled-training`. Saved files: **332**; training-directory checkpoints: **13**; all PT files: **13**.

`python -m src.history fetch --run run-019`

### run-020

Finish frozen action ablation and report.

Status: **completed**. Kind: `experiment-suite-recovery`. Saved files: **12**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-020`

[Published stage results](../results/milestones/run-020.json).

### run-021

actions diagnostic.

Status: **completed**. Kind: `frozen-diagnostic`. Saved files: **30**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-021`

## 10,000-recording expansion

### run-022

Verify exactly 10000 training recordings.

Status: **failed**. Kind: `dataset-expansion`. Saved files: **9**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-022`

### run-023

Verify exactly 10000 training recordings.

Status: **failed**. Kind: `dataset-expansion`. Saved files: **9**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-023`

### run-024

Verify exactly 10000 training recordings.

Status: **failed**. Kind: `dataset-expansion`. Saved files: **9**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-024`

### run-025

Verify exactly 10000 training recordings.

Status: **completed**. Kind: `dataset-expansion`. Saved files: **10**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-025`

### run-026

Verify the large-data GPU path on existing records.

Status: **completed**. Kind: `scaling-integration-check`. Saved files: **8**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-026`

### run-027

GPU packed-loader verification.

Status: **completed**. Kind: `controlled-training`. Saved files: **72**; training-directory checkpoints: **4**; all PT files: **4**.

`python -m src.history fetch --run run-027`

### run-028

Expand to 10000 training recordings end to end.

Status: **completed**. Kind: `scaled-training-pipeline`. Saved files: **12**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-028`

### run-029

Verify exactly 10000 training recordings.

Status: **completed**. Kind: `dataset-expansion`. Saved files: **8**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-029`

### run-030

10000-record three-step training.

Status: **completed**. Kind: `controlled-training`. Saved files: **420**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-030`

[Published stage results](../results/milestones/run-030.json).

### run-031

Frozen validation comparison of training data sizes.

Status: **completed**. Kind: `expanded-data-evaluation`. Saved files: **95**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-031`

[Published stage results](../results/milestones/run-031.json).

### run-032

Frozen test comparison of training data sizes.

Status: **completed**. Kind: `expanded-data-evaluation`. Saved files: **95**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-032`

[Published stage results](../results/milestones/run-032.json).

### run-033

Interactive recorded one-step predictions.

Status: **running**. Kind: `viewer`. Saved files: **30**; training-directory checkpoints: **0**; all PT files: **22**.

`python -m src.history fetch --run run-033`

## B: blur and capacity

### run-034

B0 fixed validation blur comparison.

Status: **completed**. Kind: `blur_benchmark`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-034`

### run-035

B1-B5 controlled blur investigation.

Status: **failed**. Kind: `blur-suite`. Saved files: **12**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-035`

### run-036

B1 reconstruction.

Status: **completed**. Kind: `blur-training`. Saved files: **28**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-036`

### run-037

B1 held-out appearance versus prediction.

Status: **completed**. Kind: `reconstruction-evaluation`. Saved files: **27**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-037`

### run-038

B2 one additional epoch.

Status: **completed**. Kind: `blur-training`. Saved files: **28**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-038`

### run-039

B0 fixed validation blur comparison.

Status: **failed**. Kind: `blur_benchmark`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-039`

### run-040

B1-B5 controlled blur investigation.

Status: **completed**. Kind: `blur-suite`. Saved files: **18**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-040`

[Published stage results](../results/milestones/run-040.json).

### run-041

B0 fixed validation blur comparison.

Status: **completed**. Kind: `blur_benchmark`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-041`

### run-042

B3 mse_branch.

Status: **completed**. Kind: `blur-training`. Saved files: **28**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-042`

### run-043

B0 fixed validation blur comparison.

Status: **completed**. Kind: `blur_benchmark`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-043`

### run-044

B3 l1_gradient_branch.

Status: **completed**. Kind: `blur-training`. Saved files: **28**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-044`

### run-045

B0 fixed validation blur comparison.

Status: **completed**. Kind: `blur_benchmark`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-045`

### run-046

B4 capacity_128.

Status: **completed**. Kind: `blur-training`. Saved files: **28**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-046`

### run-047

B0 fixed validation blur comparison.

Status: **completed**. Kind: `blur_benchmark`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-047`

### run-048

B4 capacity_256.

Status: **completed**. Kind: `blur-training`. Saved files: **28**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-048`

### run-049

B0 fixed validation blur comparison.

Status: **completed**. Kind: `blur_benchmark`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-049`

### run-050

B5 fixed test comparison.

Status: **completed**. Kind: `blur-final-evaluation`. Saved files: **108**; training-directory checkpoints: **0**; all PT files: **12**.

`python -m src.history fetch --run run-050`

### run-051

Interactive recorded one-step predictions.

Status: **running**. Kind: `viewer`. Saved files: **21**; training-directory checkpoints: **0**; all PT files: **13**.

`python -m src.history fetch --run run-051`

## C: appearance and motion

### run-052

C0-C5 execution through c1.

Status: **completed**. Kind: `appearance-suite`. Saved files: **13**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-052`

### run-053

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-053`

### run-054

C1_control.

Status: **completed**. Kind: `blur-training`. Saved files: **29**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-054`

### run-055

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-055`

### run-056

C1_appearance.

Status: **completed**. Kind: `blur-training`. Saved files: **29**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-056`

### run-057

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-057`

### run-058

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-058`

### run-059

C0-C5 execution through c5.

Status: **completed**. Kind: `appearance-suite`. Saved files: **27**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-059`

[Published stage results](../results/milestones/run-059.json).

### run-060

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-060`

### run-061

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-061`

### run-062

C2_direct.

Status: **completed**. Kind: `blur-training`. Saved files: **29**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-062`

### run-063

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-063`

### run-064

C2_motion.

Status: **completed**. Kind: `blur-training`. Saved files: **29**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-064`

### run-065

Verify actual C1/C2 initial checkpoints.

Status: **completed**. Kind: `appearance-pair-audit`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-065`

### run-066

Fixed C2 motion example at update 3000; exploratory.

Status: **failed**. Kind: `appearance-visual-inspection`. Saved files: **8**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-066`

### run-067

Fixed C2 motion example at update 3000; exploratory.

Status: **completed**. Kind: `appearance-visual-inspection`. Saved files: **17**; training-directory checkpoints: **0**; all PT files: **1**.

`python -m src.history fetch --run run-067`

### run-068

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **93**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-068`

### run-069

C3 motion controls.

Status: **completed**. Kind: `appearance-action-diagnostic`. Saved files: **9**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-069`

### run-070

C4_vector.

Status: **completed**. Kind: `blur-training`. Saved files: **29**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-070`

### run-071

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **93**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-071`

### run-072

C4_spatial.

Status: **completed**. Kind: `blur-training`. Saved files: **29**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-072`

### run-073

Verify fresh C4 vector/spatial shared renderer and data order.

Status: **completed**. Kind: `appearance-pair-audit`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-073`

### run-074

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **93**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-074`

### run-075

C5_confirmation_control.

Status: **completed**. Kind: `blur-training`. Saved files: **29**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-075`

### run-076

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-076`

### run-077

C5_confirmation_candidate.

Status: **completed**. Kind: `blur-training`. Saved files: **29**; training-directory checkpoints: **16**; all PT files: **16**.

`python -m src.history fetch --run run-077`

### run-078

Verify C5 seed-7 initialization and shuffle replication.

Status: **completed**. Kind: `appearance-pair-audit`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-078`

### run-079

Fixed validation detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-079`

### run-080

Fixed test detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-080`

### run-081

Fixed test detail and movement.

Status: **completed**. Kind: `appearance-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-081`

### run-082

Post-selection descriptive validation of selected C1 model.

Status: **completed**. Kind: `appearance-action-diagnostic`. Saved files: **9**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-082`

### run-083

Interactive recorded one-step predictions.

Status: **running**. Kind: `viewer`. Saved files: **9**; training-directory checkpoints: **0**; all PT files: **1**.

`python -m src.history fetch --run run-083`

## D: detail and transport

### run-084

Replay fixed selected-model one-step softness.

Status: **completed**. Kind: `blur-diagnostic`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-084`

### run-085

Replay fixed selected-model one-step softness.

Status: **completed**. Kind: `blur-diagnostic`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-085`

### run-086

Compare changed/static detail in six archived fixed visual examples.

Status: **completed**. Kind: `blur-diagnostic`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-086`

### run-087

Phase D through tiny.

Status: **completed**. Kind: `detail-suite`. Saved files: **12**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-087`

### run-088

Frozen D validation evaluation.

Status: **completed**. Kind: `detail-evaluation`. Saved files: **57**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-088`

### run-089

D1_single.

Status: **completed**. Kind: `detail-training`. Saved files: **25**; training-directory checkpoints: **9**; all PT files: **9**.

`python -m src.history fetch --run run-089`

### run-090

D1_multi.

Status: **completed**. Kind: `detail-training`. Saved files: **59**; training-directory checkpoints: **28**; all PT files: **28**.

`python -m src.history fetch --run run-090`

### run-091

D1_multi_extension.

Status: **completed**. Kind: `detail-training`. Saved files: **49**; training-directory checkpoints: **18**; all PT files: **18**.

`python -m src.history fetch --run run-091`

### run-092

Phase D through flow.

Status: **completed**. Kind: `detail-suite`. Saved files: **14**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-092`

### run-093

Mask-free known translations; held-out sources/offsets.

Status: **completed**. Kind: `synthetic-flow-training`. Saved files: **18**; training-directory checkpoints: **9**; all PT files: **10**.

`python -m src.history fetch --run run-093`

### run-094

Phase D through all.

Status: **completed**. Kind: `detail-suite`. Saved files: **30**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-094`

[Published stage results](../results/milestones/run-094.json).

### run-095

D3_fixed.

Status: **completed**. Kind: `detail-training`. Saved files: **18**; training-directory checkpoints: **6**; all PT files: **6**.

`python -m src.history fetch --run run-095`

### run-096

Frozen D validation evaluation.

Status: **completed**. Kind: `detail-evaluation`. Saved files: **105**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-096`

### run-097

D3_aligned.

Status: **completed**. Kind: `detail-training`. Saved files: **18**; training-directory checkpoints: **6**; all PT files: **6**.

`python -m src.history fetch --run run-097`

### run-098

Matched archived C2 versus D3 fixed-source render components.

Status: **completed**. Kind: `transport-health-diagnostic`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-098`

### run-099

Verify actual fixed/aligned initial model, optimizer and shuffle.

Status: **completed**. Kind: `detail-pair-audit`. Saved files: **7**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-099`

### run-100

Frozen D validation evaluation.

Status: **completed**. Kind: `detail-evaluation`. Saved files: **105**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-100`

### run-101

D4_mse.

Status: **completed**. Kind: `detail-training`. Saved files: **18**; training-directory checkpoints: **6**; all PT files: **6**.

`python -m src.history fetch --run run-101`

### run-102

Frozen D validation evaluation.

Status: **completed**. Kind: `detail-evaluation`. Saved files: **63**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-102`

### run-103

D4_balanced.

Status: **completed**. Kind: `detail-training`. Saved files: **18**; training-directory checkpoints: **6**; all PT files: **6**.

`python -m src.history fetch --run run-103`

### run-104

Frozen D validation evaluation.

Status: **completed**. Kind: `detail-evaluation`. Saved files: **63**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-104`

### run-105

D4_balanced_boundary.

Status: **completed**. Kind: `detail-training`. Saved files: **18**; training-directory checkpoints: **6**; all PT files: **6**.

`python -m src.history fetch --run run-105`

### run-106

Frozen D validation evaluation.

Status: **completed**. Kind: `detail-evaluation`. Saved files: **63**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-106`

### run-107

D5_fixed.

Status: **completed**. Kind: `detail-training`. Saved files: **21**; training-directory checkpoints: **9**; all PT files: **9**.

`python -m src.history fetch --run run-107`

### run-108

Frozen D validation evaluation.

Status: **completed**. Kind: `detail-evaluation`. Saved files: **63**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-108`

### run-109

D5_curriculum.

Status: **completed**. Kind: `detail-training`. Saved files: **21**; training-directory checkpoints: **9**; all PT files: **9**.

`python -m src.history fetch --run run-109`

### run-110

Frozen D validation evaluation.

Status: **completed**. Kind: `detail-evaluation`. Saved files: **63**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-110`

### run-111

Frozen D test evaluation.

Status: **completed**. Kind: `detail-evaluation`. Saved files: **63**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-111`

### run-112

Frozen final selected model controls.

Status: **completed**. Kind: `detail-action-diagnostic`. Saved files: **9**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-112`

## E: residual generation

### run-113

E one-step conditional generation pilot.

Status: **completed**. Kind: `conditional-diffusion-suite`. Saved files: **11**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-113`

### run-114

E reference quick.

Status: **completed**. Kind: `conditional-head-evaluation`. Saved files: **15**; training-directory checkpoints: **0**; all PT files: **2**.

`python -m src.history fetch --run run-114`

### run-115

E diffusion one-step pilot.

Status: **completed**. Kind: `conditional-head-training`. Saved files: **14**; training-directory checkpoints: **5**; all PT files: **5**.

`python -m src.history fetch --run run-115`

### run-116

E one-step conditional generation pilot.

Status: **completed**. Kind: `conditional-diffusion-suite`. Saved files: **27**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-116`

[Published stage results](../results/milestones/run-116.json).

### run-117

E training probe samples.

Status: **completed**. Kind: `conditional-head-evaluation`. Saved files: **20**; training-directory checkpoints: **0**; all PT files: **4**.

`python -m src.history fetch --run run-117`

### run-118

E deterministic one-step pilot.

Status: **completed**. Kind: `conditional-head-training`. Saved files: **20**; training-directory checkpoints: **9**; all PT files: **9**.

`python -m src.history fetch --run run-118`

### run-119

E deterministic full validation.

Status: **completed**. Kind: `conditional-head-evaluation`. Saved files: **24**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-119`

### run-120

E deterministic zero-action control.

Status: **completed**. Kind: `conditional-head-evaluation`. Saved files: **16**; training-directory checkpoints: **0**; all PT files: **2**.

`python -m src.history fetch --run run-120`

### run-121

E diffusion one-step pilot.

Status: **completed**. Kind: `conditional-head-training`. Saved files: **20**; training-directory checkpoints: **9**; all PT files: **9**.

`python -m src.history fetch --run run-121`

### run-122

E diffusion full validation.

Status: **completed**. Kind: `conditional-head-evaluation`. Saved files: **24**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-122`

### run-123

E diffusion zero-action control.

Status: **completed**. Kind: `conditional-head-evaluation`. Saved files: **16**; training-directory checkpoints: **0**; all PT files: **2**.

`python -m src.history fetch --run run-123`

### run-124

E reference full validation.

Status: **completed**. Kind: `conditional-head-evaluation`. Saved files: **23**; training-directory checkpoints: **0**; all PT files: **6**.

`python -m src.history fetch --run run-124`

### run-125

Browse E fixed validation comparisons.

Status: **running**. Kind: `saved-results-viewer`. Saved files: **11**; training-directory checkpoints: **0**; all PT files: **0**.

`python -m src.history fetch --run run-125`
