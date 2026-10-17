# Limitations and intended interpretation

These detailed limits focus on the final residual-head study. Earlier stages
have their own data sizes, horizons, comparisons and selection rules in the
[milestone report](project-history.md) and [development record](development-record.md).
Their official-test reuse and limited confirmation experiment remain disclosed.

1. **One training seed in the main comparison.** Three noise seeds are repeated
   sampling of one diffusion checkpoint. Bootstrap intervals describe variation
   over the 51 recordings, conditional on the trained weights.
2. **Validation reuse.** Selection windows are included in the larger reported
   validation set. No phase-E independent final-test result is available.
   Official test data was inspected/evaluated in earlier project phases.
3. **Narrow data and resolution.** The 10,000-recording archive-order subset and
   51-recording validation cohort are not representative of every BAIR scene,
   robot, camera or real-world task. All images are 64x64; finer target detail
   cannot be recovered as measured ground truth by enlargement.
4. **Short context and horizon.** Eight real frames initialize each prediction.
   The released new heads are tested for one step only, without a closed-loop
   policy or multistep stability experiment.
5. **Frozen temporal backbone.** Phase E examines new rendering heads on a
   shared fixed state/dynamics model. It does not test jointly retraining a
   diffusion-based dynamics model, a new tokenizer or a different history model.
6. **Matched architecture, different objectives/inference.** The deterministic
   and diffusion heads have the same architecture and conditioning. The pilot
   changes objective and inference procedure together. It is not a pure test
   of randomness alone or an equal-inference-compute comparison.
7. **Perceptual quality differs from prediction accuracy.** Sharper robot
   details can be misplaced or invented. Lower LPIPS does not establish
   physical correctness. Changed pixels include noise/lighting and are not
   segmentation or tracked-object labels.
8. **Uncertainty is uncalibrated.** Sample dispersion can reflect uncertainty,
   underfitting or model artifacts. No probability calibration or conditional
   coverage result is reported. Choosing a sample using the actual future
   would be an oracle evaluation; this project scores all fixed samples.
9. **Limited budgets.** Endpoint learning curves still improve. These budgets
   do not establish convergence or optimal performance of either model family.
10. **Historical provenance predates Git.** Training records contain source
    fingerprints, but no historical Git commits. The publication source import
    is a later snapshot. Portable exports change serialization/path metadata
    while preserving numerical weights; both identities are recorded.
11. **Reproducibility is layered.** Frozen score tables and saved predictions
    are reproducible without training. Numerical inference depends on the
    recorded libraries/device; retraining on another backend may differ.

## Intended use

The release supports research education, architecture inspection, evaluation
of saved outputs and controlled visual-prediction experiments. It is not a
validated robot controller, physical simulator or safety-critical planner.
No transfer to real robot control, robust planning or causal reasoning is claimed.

## Future experiments

Useful next studies include training-seed replication, a newly reserved
holdout, longer-horizon residual training with retained short-horizon accuracy,
joint temporal-backbone adaptation, improved spatial correspondence, and
calibration of stochastic predictions. They require new protocols and new run
records; the current release remains frozen.
