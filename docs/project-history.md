# End-to-end research milestones

## Project purpose

Learn an action-conditioned visual world model from recorded camera/action
sequences. Given observed history and a proposed action sequence, predict the
corresponding future observations. The work focuses on representation,
temporal dynamics, training and evaluation, with an image viewer for inspection.
There is no trained policy or planning/control harness in this practice.

The project begins with random initialization and proceeds through a series
of questions. Diffusion is the last rendering experiment in that sequence.
All stage-specific mathematics, tables and execution details are preserved in
the [development record](development-record.md). Every run and saved checkpoint
is located through the [catalog](run-catalog.md) and
[checkpoint registry](../results/checkpoint-registry.csv).

## 1. Data and observable history — run-001–004

**Question:** What information is available, and how does it reach a learned state?

Fetch original BAIR training/test TFRecord members by verified HTTP ranges;
validate tar headers, TFRecord CRC32C, image/action arrays and file receipts.
Split complete recordings, initially 205 train / 51 validation / 256 official
test, preserving all frames of a recording in one split. Eight main-camera
images and seven preceding actions summarize observed history; the next
four-component action specifies the prediction transition. Robot positions
are retained for diagnostics and excluded from model inputs.

Implement and inspect the CNN encoder and recurrent memory separately.
Random component snapshots, tensor traces and source/data identities are
saved before joint learning. The latent vector is a learned representation,
not the physical robot state and not an explicitly labeled list of objects.

**Next decision:** join encoder, memory, transition and decoder and test whether
the complete model can learn future images.

## 2. First joint training and saturation diagnosis — run-005–009

The CNN maps images to features, the GRU integrates features and past actions,
the dynamics MLP predicts a next latent, and a transpose-convolution decoder
renders RGB. The one-step image loss backpropagates through all four modules.
Save random, periodic, best and latest checkpoints and compare against the
actual target and copy-last-frame baseline.

The initial run-005 terminates without an established process-failure cause.
A separate probe reproduces zero history gradients: raw CNN magnitudes reach
529 and 99.87% of state coordinates saturate near absolute value one.
Per-frame normalization reduces that fraction to 0.78% on the probe and
restores gradients. This diagnosis explains the gradient problem, not the
separate termination.

Corrected run-007 restarts from random weights and completes ten epochs,
selecting epoch nine on validation. Test one-step MSE drops from 0.095703 at
initialization to 0.030591, while copying achieves 0.019174. Recursive h20 test
MSE remains 0.085965 versus copy 0.055230. Learning succeeds; the resulting
predictor is still blurry and weak over open-loop rollouts.

**Next decision:** separate fitting capacity, history refresh, temporal
supervision and action dependence before attributing failure to one cause.

## 3. Controlled temporal/action diagnosis — run-010–021

- **Tiny fitting:** run-011 reaches MSE 0.001376 on 32 fixed windows, versus
  copy 0.017638; 29/32 windows beat copying. This demonstrates fitting capacity,
  not held-out generalization.
- **Additional observations:** run-012 compares matching validation targets;
  h20 MSE is 0.085870 recursively and 0.029994 with refreshed real history.
  The information advantage is large, without identifying its sole cause.
- **Matched one-/three-step training:** run-013/014 share initial parameters
  and starting windows, completing 2,570 updates each. Their subsequent h20
  validation MSEs are 0.077939 and 0.056005; copy is 0.058061. Three-step
  supervision uses more target frames and compute per update.
- **Action controls:** run-017/018 substitute past and future action sequences.
  Future-action substitution worsens the three-step model's h20 MSE by
  0.003833. Past-action effects are smaller. Implausible substitutions limit
  causal interpretation.
- **Trained no-action control:** run-019 has one-step validation MSE 0.031387,
  comparable to the conditioned control's 0.031672, but h20 error is worse.
  Run-021 verifies exactly zero action sensitivity in that ablation.

Run-010 and run-019 retain status-writing failures after completed work.
Run-020 verifies completed stage identities and recovers the remaining
diagnostic/reporting work without repeating training. Failed status and
successful recovery remain separate records.

**Next decision:** expand training diversity while preserving holdout identity
and the three-step objective.

## 4. Expansion to 10,000 recordings — run-022–032

Implement resumed downloads with prefix receipts, exact-count archive-order
selection, immutable holdout hashes, disk-backed packed arrays and spawned
loader workers. Preserve unsuccessful acquisition attempts run-022–024.
Run-026/027 verify the GPU/packed-loader path before the full experiment.

Run-030 completes 12,500 updates, batch 16: 200,000 starting windows and
600,000 supervised future-frame exposures. Run-031/032 score validation/test
against the earlier 205-recording reference with byte-identical holdouts.

| Test metric | 205-record reference | 10,000-record model | Copy |
|---|---:|---:|---:|
| One-step MSE | 0.030325 | 0.023891 | 0.019174 |
| Recursive h20 MSE | 0.054927 | 0.046178 | 0.055230 |

Expanded training improves one-step and recursive errors and beats copying
at h5/h10/h20. Blur remains. Both data diversity and the optimization budget
change, so the improvement cannot be assigned to data size alone.

**Next decision:** investigate whether blur is primarily a fitting, rendering,
objective or temporal-representation limitation.

## 5. Phase B: reconstruction, optimization, loss and capacity — run-034–050

Freeze validation starts, horizons, image-change strata and six visual cases
before evaluating candidates. Compare reconstruction against future prediction,
an additional epoch, matched MSE and L1+gradient branches, and vector latent
widths 128/256. Source identities and common scoring protocols are recorded.

Additional optimization improves pixels. L1+gradient changes boundary quality
with an accuracy tradeoff. Increasing vector width gives a smaller improvement
than the remaining visible blur suggests. Select run-042 by the frozen
validation one-step MSE rule, then report its official-test comparison in
run-050. Run-035/039 failures remain part of the archive.

**Next decision:** add direct access to observed appearance and investigate
whether source-image transport can retain detail.

## 6. Phase C: appearance, transport and spatial memory — run-052–082

Warm-start paired controls from the same original checkpoint, use fresh Adam,
record matching initialization/data-order audits and evaluate appearance and
movement together over h1/5/10/20. The appearance model adds observed CNN
feature adapters to decoder scales; target images never enter this context.

Run-056 becomes the selected appearance backbone. C2 compares direct rendering
with a copy/warp/generate renderer and diagnoses suppressed warp use and
out-of-bounds sampling. C3 tests action dependence. C4 compares vector memory
with a tested 8x8 ConvGRU alternative; the spatial candidate fails rollout
promotion guards. C5 repeats the appearance intervention with a second
initialization/data-order setting, sharing base weights rather than providing
an independent from-scratch replication.

The selected checkpoint differs from the latest endpoint where required by
quality guards. Run-059 records selection and official-test comparisons;
run-065/073/078 audit actual paired states. Run-082 reports descriptive action
controls after selection.

**Next decision:** test sharp fitting explicitly, verify flow conventions with
known motion and repair transport before more full-scale training.

## 7. Phase D: detail, flow repair, losses and curriculum — run-084–112

Diagnose softness on the fixed cases and test whether sharp output is possible
on one moving window and 32 windows. Run-089 passes the one-window fit; the
32-window extension does not pass the combined accuracy gate despite sharper
examples. Memorization does not establish generalization.

Train a synthetic translation diagnostic with known flow. Run-093 achieves
held-out EPE 0.1541 pixels and warp MSE 0.000321. Repair bounded/aligned
transport and compare matched components: six-example out-of-bounds sampling
falls from 92.29% to 20.62%, but the real motion problem is not resolved.

Compare MSE, changed/static balanced and boundary losses; then fixed-horizon
and curriculum supervision. The curriculum endpoint improves h20 MSE by
13.16% but worsens h3 changed error by 4.69%, failing the frozen 2% tolerance.
Initial quality-selected states can remain the best states when later updates
fail guards. Run-094 records decisions and final evaluations, including
run-111 official test and run-112 action diagnostics.

**Next decision:** hold the selected backbone fixed and isolate a stronger
image-generation head with a matched deterministic control.

## 8. Phase E: deterministic and diffusion residual heads — run-113–124

Freeze run-056 and train identically initialized conditional U-Nets with
matched private data permutations, batch size and 6,000-update budgets.
Run-115 is the preserved earlier pilot. Run-118 is the deterministic residual
branch; run-121 uses v-parameterized residual diffusion. Both retain the
original decoder and condition on observed appearance, action, predicted
latent and coarse image. Targets construct training supervision only.

Run-119/122/124 score all 1,122 validation windows. Diffusion scores all three
fixed samples per window. Deterministic MSE improves 33.96% and LPIPS 53.54%;
diffusion LPIPS improves 61.96% while MSE worsens 24.71% and changed-region
error worsens 40.98%. Run-120/123 retain paired action controls.

The [final-stage report](experiments.md) provides the full table and descriptive
bootstrap intervals. Phase E uses one training seed and reused validation;
new-head recursive performance and calibrated uncertainty remain untested.

## 9. Inspection and publication

Interactive sessions run-033/051/083 browse model predictions; run-125 browses
saved final comparisons. Their captured status may remain `running` because
they were live services at the freeze. These sessions are exploratory and are
not additional aggregate benchmarks. Their saved files are included in the
complete archive, with their actual captured statuses.

Version 1.0.0 published a focused final-stage package. Version 1.1.0 extends the
publication to the complete project without retraining or changing scientific
outputs. The archive covers every saved file in run-001–125. Publication/export
and later verification sessions are operational records, not new research.

## What the progression establishes

The system learns from real image/action sequences and supports supplied-action
recursive prediction. The experiments expose temporal, optimization and
rendering limitations and measure several useful improvements. Sharp detail,
low perceptual distance, good single-frame fitting and action sensitivity each
answer different questions. No single intervention solves every observed
failure, and no reliable robot-control or general physical-simulation claim
follows from the present evidence.
