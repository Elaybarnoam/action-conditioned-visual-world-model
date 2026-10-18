# DROID: higher-resolution experiment

This new experiment is separate from the frozen BAIR v1.1.0 research release.
It implements a new baseline trained from random weights on DROID. Existing
BAIR models are not used as pretrained weights.

## Question

Can a model trained on higher-resolution robot video preserve observed detail
while predicting the correct action-dependent motion? Higher resolution alone
does not establish improved prediction accuracy.

## Initial data protocol

- Official `droid_100/1.0.0` debugging subset: 100 recordings, approximately 2.2 GB
  of source TFRecords, with variable episode lengths.
- One external camera: `exterior_image_1_left`, RGB **320 wide by 180 high**.
  Preserve native pixels; no enlargement of BAIR images.
- Inputs: eight observed images, seven previous actions, and the next action.
- Target: the recorded next image. Action at index t accompanies the transition
  from observation t to observation t+1; the last action has no next-image target.
- Explicit action dictionary fields: six Cartesian velocity commands plus one
  gripper position command. Do not interpret the default `steps/action` as this
  representation. These commands are not measured image displacement. Confirm
  exact units/frame conventions against the collection code before physical
  interpretation; initial data checks establish serialization and indexing.
- Holdouts: 80 train / 10 validation / 10 test **recordings**, assigned by a seeded
  hash of recording identity. No windows cross recording boundaries. These splits
  do not establish separation by scene, building, task or collection session.
- Fixed checkpoint evaluation: one central eight-frame window and its next-image
  target per validation recording. Final evaluation should additionally cover all
  eligible windows and report uncertainty at the recording level. Test recordings
  remain reserved for final scoring.
- This is a small debugging sample, not a representative full-DROID benchmark.

## Reproduce data preparation

```powershell
uv run --locked python -m src.prepare_droid
```

Configuration: `configs/droid.json`. No TensorFlow installation is required.
The reader uses the existing protobuf dependency and Pillow.

Downloads pin Google Cloud object generations and verify publisher CRC32C, size
and local SHA256. Every TFRecord length/payload CRC is verified; all selected
camera frames, finite action values, lengths, episode boundaries and unique
recording identities are checked. Counts must agree with publisher metadata.
Repeated preparation checks existing bytes and refuses changed prepared arrays
or a changed manifest. Interrupted source downloads restart their current file.

Local outputs (ignored by Git):

```text
data/droid/
  raw/                  # source shards, schema, publisher metadata, receipts
  prepared/<id>/        # memory-mappable images.npy and actions.npy per recording
  manifest.json         # source hashes, splits, array hashes, fixed windows
evaluation/runs/droid/run-NNN/
  config.json
  droid-config.json
  source-version.json
  source-objects.json
  dataset-manifest.json
  data-checks.json       # counts and fixed validation copy baseline
  validation-preview.png
  metrics.jsonl
  status.json
  summary.md
```

The initial preview shows current image, action and actual next image. It has no
model prediction yet. Native pixel arrays require additional disk space beyond
the compressed source download.

### First preparation result

Local `evaluation/runs/droid/run-001` completed successfully:

- 100 verified recordings; 32,212 decoded external-camera frames.
- 24,702 eligible training windows, 4,058 validation windows, 2,652 test windows
  with eight-frame context and one future target.
- Fixed validation copy MSE: **0.0028154**, averaged over the ten preselected
  central windows. This is not an all-window validation score or a model result.
- Nine focused reader/protocol tests pass. Frozen BAIR public-file checks and
  saved-result table reconstruction still pass. Training was not part of that
  data-preparation run.

## Implementation sequence

1. Fetch, verify, split and visually inspect the sample; measure copy baseline.
2. Add a disk-backed window loader, training-only action normalization, and
   high-resolution CNN/temporal-memory baseline with a spatial decoder. Account
   for rectangular dimensions explicitly; do not stretch frames to a square.
3. Verify gradient flow and overfit a tiny set before a full pilot.
4. Train from random weights with scheduled fixed validation, initial/best/latest
   checkpoints, saved predictions and full optimizer/RNG/config provenance.
5. Compare copy, deterministic prediction and action substitutions using MSE,
   changed-region error, boundary error and perceptual metrics. Inspect geometry
   and sharpness separately. Compare resolutions on the **same DROID recordings**
   before attributing differences to resolution rather than dataset change.
6. Consider conditional diffusion only after the dynamics baseline is measured.
   Keep final test data out of model selection.

## Baseline architecture and training protocol

Run the complete gate, training and evaluation sequence:

```powershell
uv run --locked python -m src.train_droid
```

The training configuration is `configs/droid-train.json`. Modules have distinct
responsibilities:

| Module | Responsibility |
|---|---|
| `src/data/droid_windows.py` | Memory-map recordings; align histories/actions/targets; calculate training-only action statistics |
| `src/model/droid.py` | Predict one future RGB image from observed history and commands |
| `src/training/droid.py` | Tiny-fit gate, random initialization, optimizer updates, checkpoint selection and run records |
| `src/evaluation/droid.py` | Copy/action controls, per-window metrics, recording bootstrap and example panels |
| `src/train_droid.py` | Command-line entry point |

### Computation

Each RGB frame is normalized to [0,1], padded at its bottom from 180 to 192
pixels by replication, then processed by four convolution/ELU stages. Channels
are 16/32/64/96, kernels 4, stride 2, padding 1; the final map is 96x12x20.
Fixed 4x4 average pooling produces 96x3x5, then a linear projection and layer
normalization produce 128 frame features. This retains coarse spatial layout.

For seven-dimensional raw commands, normalization uses the population mean and
standard deviation of usable **training** commands only, with standard deviation
floored at 0.001. The final command in each recording is excluded. Statistics are
saved both in the run record and as model buffers; test/validation commands do
not influence them. No physical state measurements are model inputs.

The history GRU has 128 hidden dimensions; each update combines current frame
features and the preceding normalized command. A 135→256→256→128 MLP predicts
the next latent from the final observed memory and the next command, with ELU
hidden activations and final tanh. A linear projection broadcasts this latent
over the last observed feature map. Four bilinear upsampling/convolution stages
combine it with last-observed appearance skips, including the observed RGB image.
The decoder emits a three-channel correction:

$$
\hat o_{t+1}=\operatorname{clip}_{[0,1]}
\left(o_t+\tanh R_\theta(\hat z_{t+1},C_\theta(o_t))\right).
$$

Only the observed past and supplied action enter this expression. The real next
image appears solely in the RGB MSE objective. All encoder, memory, dynamics and
decoder weights update jointly. The final convolution starts with small random
weights (standard deviation 0.001), so the initial model is near copying; its
initial MSE is therefore not comparable to the earlier BAIR random-decoder MSE.
Predictions are cropped back to the native 180x320 dimensions before scoring.

### Gate, selection and scoring

- Tiny gate: choose the four central training windows with largest copy MSE;
  train for 800 updates with Adam at 0.001. Require nonzero finite gradients into
  every model component, MSE below half its initial value and below half copy MSE.
  This is a capacity/implementation check, not evidence of generalization.
- Full pilot: discard tiny-fit weights, reset seed 3 and initialize a new model.
  Three complete epochs, batch 8 (keep partial final batches), Adam at 0.0003,
  gradient norm clipping at 10; float32 and deterministic PyTorch operations.
  Two loader workers transfer compact uint8 images; normalization happens on the
  selected device. Every training window is visited exactly once per epoch.
- Select the smallest MSE on the ten fixed central validation windows, including
  the initial model. Validate/save every 500 updates and at every epoch end.
- Initial, best, latest and scheduled checkpoint files retain model buffers,
  Adam state, Python/NumPy/Torch/CUDA RNG state, shuffle generator, sampler order
  and position where relevant. These are saved continuation states; a command
  for resuming interrupted DROID training is not implemented in this pilot.
- Record exact source ZIP, source hashes, environment, data identity, settings,
  training metrics and checkpoint selection before test scoring.
- Score the selected checkpoint on every validation and test window. Compare
  model, copy and the same trained model with all raw past/next commands zeroed.
  The zero-action substitution measures sensitivity, not causal identification.
- Metrics: RGB MSE; changed-region MSE with observed max-channel change >0.05;
  signed pixel-gradient MAE; uniform-window SSIM; AlexNet LPIPS. Save every
  window's results. Changed-region means exclude windows with no changed pixels.
  AlexNet LPIPS differs from the frozen BAIR phase-E SqueezeNet metric; their
  numerical LPIPS values should not be compared directly.
- Report a paired 2,000-resample, seed-17 recording bootstrap for MSE change vs
  copy, retaining all windows of each sampled recording. Ten holdout recordings
  give limited uncertainty estimates; these do not cover training-seed variation.
- Save all ten fixed validation and test examples, their numerical tensors and
  current/actual-next/predicted-image panels, plus training curves.

Full validation contains the selection windows; its results are not an
independent final test. Final test scores must not guide follow-up adjustments
within this pilot. Only one training seed is used, and consecutive windows are
strongly correlated. Resolution benefits cannot be isolated by comparing this
pilot with BAIR, because scenes, action meanings and training protocol differ.

## First trained pilot: measured results

Training and evaluation completed in local `evaluation/runs/droid/run-004`:
769,939 trainable parameters, 9,264 updates, 74,106 training-window exposures,
three full epochs. Training plus scheduled validation took 1,224.5 seconds
(about 20.4 minutes) on the recorded RTX 5060 Laptop GPU. This excludes tiny
fitting, final evaluation, data preparation and verification.

The four-window tiny-fit check passed in run-003: initial MSE 0.0094672, copy
0.0094665, selected MSE 0.0006756 after 800 updates (92.9% below copy). All model
components had nonzero finite gradients. Run-002 preserves the failed first GPU
attempt: adaptive average pooling lacked deterministic CUDA backward support;
fixed 4x4 average pooling resolved it and passed an actual CUDA-update test.

The full pilot used fresh random weights. Update **6,500** was selected by fixed
validation MSE 0.00214985. The final update 9,264 scored 0.00230648 on those
windows, so later optimization did not improve this selection score.
The selected checkpoint SHA256 is
`95809370fb0f13201eb8a94d2600b961ae88c592cdd57b6e43637e23abdddc4f`.

### All eligible one-step windows

Lower is better for all metrics below except SSIM.

| Split / predictor | MSE | Changed MSE | Boundary MAE | SSIM | AlexNet LPIPS |
|---|---:|---:|---:|---:|---:|
| Validation / copy | 0.00236887 | 0.0370852 | 0.0122539 | 0.949001 | 0.0135463 |
| Validation / model | **0.00207543** | **0.0279322** | 0.0128875 | 0.946048 | 0.0297914 |
| Validation / zero actions | 0.00224964 | 0.0342508 | 0.0123562 | **0.949164** | 0.0163162 |
| Test / copy | 0.00115154 | 0.0262364 | **0.00939286** | **0.964315** | **0.00897147** |
| Test / model | **0.00106378** | **0.0198796** | 0.0102699 | 0.959925 | 0.0238833 |
| Test / zero actions | 0.00110972 | 0.0237368 | 0.00985226 | 0.962754 | 0.0147701 |

Validation contains 4,058 windows / ten recordings; test contains 2,652 windows /
ten recordings. Changed-region scores exclude 11 validation and 15 test windows
with empty masks. Point estimates average windows, so longer recordings receive
more weight. Zero actions means raw past **and** next commands are zeroed in the
same trained model, not a separately trained ablation.

Relative to copy, validation MSE improves **12.4%** and changed-region error
**24.7%**. Test MSE improves **7.6%** and changed-region error **24.2%**. MSE improves
in eight of ten validation recordings and seven of ten test recordings. Recorded
commands give 7.7% lower validation MSE and 4.1% lower test MSE than the joint
zero-action substitution. This does not isolate past versus next-action use.

However, test boundary error is **9.3% worse** and AlexNet LPIPS is **166.2% worse**
than copy; SSIM is also lower. Visual inspection shows preserved static detail,
but softened/ghosted moving robot boundaries in some examples. The pixel-MSE
objective can favor intermediate appearances when motion is uncertain or
incorrect. This experiment does not isolate the cause of those errors.

Paired recording-bootstrap 95% intervals for percentage MSE change versus copy:

- Validation: **[-20.16%, -1.61%]**; includes the checkpoint-selection windows.
- Test: **[-18.09%, +7.04%]**; includes zero, so this small pilot does not establish
  a reliable test-MSE advantage across recordings.

**Conclusion:** the implementation learns and improves average target-pixel and
changed-region error in this sample, while losing to copy on appearance/boundary
metrics. Higher-resolution data alone has not resolved moving-detail prediction.
No additional tuning was performed after test scoring, and no diffusion model
was trained for this DROID pilot. A matched DROID resolution control and stronger
motion/detail experiments remain future work.

### Retained local evidence

Run-004 contains initial/best/latest and every scheduled training checkpoint,
optimizer/RNG/sampler state, `source.zip`, environment and data identities,
training curves, full per-window and per-recording results, bootstrap intervals,
and all ten fixed prediction examples for each holdout split. The files
`validation-predictions.png` and `test-predictions.png` show current, real next
and predicted images with the supplied action. Their `.pt` companions retain
numerical outputs. These working runs remain excluded from Git.

Verification: **191 tests passed**, including a deterministic CUDA update and
evaluation of empty changed-region masks. All 39 frozen BAIR published files and
the original result-table reconstruction still verify. Reloading the selected
DROID checkpoint reproduced saved validation predictions with maximum absolute
error **0.0** and exactly reproduced its fixed-validation MSE. Run-004 also retains
`checkpoint-index.json` (24 files with SHA256/size), `verification.json` and hashes
of the perceptual metric's pretrained assets. No training or selection changed
after test scoring.

## Local results viewer

```powershell
uv run --locked python -m src.droid_viewer --run evaluation/runs/droid/run-004 --port 8769
```

When reproducing, replace `run-004` with the completed training run printed by
`src.train_droid`; run numbers depend on prior local attempts.

Open `http://127.0.0.1:8769/`. The viewer shows all ten saved test examples and
ten saved validation examples, raw actions, one deterministic model prediction
per input, example MSE and complete holdout scores. It loads the evaluation's
saved tensors without model inference or training. Each viewer session creates
a separate DROID run record containing source hashes, captured page and viewing
events. The earlier BAIR viewers remain separate.

## Sources and licensing

[Official DROID documentation](https://droid-dataset.github.io/droid/the-droid-dataset.html)
describes the sample, schema and download formats.
[Dataset project and paper](https://droid-dataset.github.io/) describe collection
and scope. DROID is released under CC BY 4.0; retain source attribution for derived
images and public artifacts. The repository code license does not replace data
license terms. This preparation does not publish DROID recordings.
