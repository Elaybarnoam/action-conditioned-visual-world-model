# Model card: frozen BAIR residual-decoder pilot, v1.0.0

**Maintainer:** Elay Barnoam. **Type:** research software and trained prediction
weights. **Code/model license:** MIT. **Source imagery:** BAIR CC BY 4.0,
with attribution and modifications in THIRD_PARTY_NOTICES.md.

## Models

- Original run-056: CNN/GRU/action-conditioned dynamics and appearance decoder,
  1,997,987 parameters. Originally trained from random through earlier stages;
  reused frozen in the main comparison.
- Deterministic run-118: matching residual U-Net head with random hidden
  initialization and zero output convolution; 1,934,819 trainable parameters.
- Diffusion run-121: matching head with v objective, 100-step cosine schedule,
  50-step DDIM sampling and three reported initial-noise seeds.

Inputs are eight observed images, seven past actions and the next 4-D action.
No robot positions enter prediction. Recorded x/y commands, gripper-close
duration and lift duration are not measured pixel velocity. The original
decoder remains present; diffusion generates a residual correction.

## Data and evaluation

BAIR original-v0, main camera, RGB 64x64, 30-frame recordings; 10,000 train /
51 validation / 256 official-test recordings. Phase E uses validation only.
Selection uses 51 starts; final validation includes them among 1,122 overlapping
windows. Three sampling seeds are not independently trained replications.
Dataset/checkpoint/metric hashes accompany the release.

The deterministic head improves MSE, changed-region error, boundary error,
LPIPS and SSIM at this budget. Diffusion has better LPIPS and sharper inspected
details but worse target accuracy, failing pilot accuracy guards. See
experiments.md for the values and conditional bootstrap intervals.

## Intended interpretation

Uses include learning visual prediction and reproducing controlled decoder
comparisons. There is no validated robot-control, causal-planning, calibration
or general physical-simulation capability. New-head multistep behavior and
training-seed robustness are untested. Generated details can be sharp but wrong.

Inference bundles contain EMA heads, the required backbone and verification
examples. Training bundles contain optimizer/RNG/sampler states. Downloads pin
SHA-256. External metric weights are obtained from their libraries, not bundled
with world models. The standard-library demo is served on loopback.
