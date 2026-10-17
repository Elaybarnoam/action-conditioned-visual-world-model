# Related work and positioning

- **BAIR visual prediction.** Ebert et al., *Self-Supervised Visual Planning
  with Temporal Skip Connections* (2017), provides the dataset context.
  [Paper](https://arxiv.org/abs/1710.05268).
- **Stochastic future prediction.** Babaeizadeh et al., *Stochastic Variational
  Video Prediction* (2018), models multiple possible futures.
  [Paper](https://arxiv.org/abs/1710.11252).
- **Fitting capacity.** Babaeizadeh et al., *FitVid: Overfitting in Pixel-Level
  Video Prediction* (2021), motivates examining capacity/optimization alongside
  uncertainty. [Paper](https://arxiv.org/abs/2106.13195).
- **Diffusion objectives and sampling.** [DDPM](https://arxiv.org/abs/2006.11239),
  [DDIM](https://arxiv.org/abs/2010.02502) and
  [Progressive Distillation](https://arxiv.org/abs/2202.00512) provide established
  diffusion training, sampling and v-parameterization foundations.
- **Video diffusion.** Voleti et al., *MCVD* (2022), studies conditional video
  diffusion for prediction/generation/interpolation.
  [Paper](https://arxiv.org/abs/2205.09853).
- **World-model rendering.** [GAIA-1](https://arxiv.org/abs/2309.17080) combines
  autoregressive latent-token prediction with a video diffusion decoder;
  [DIAMOND](https://arxiv.org/abs/2405.12399) uses diffusion for action-conditioned
  world modeling. These systems differ in data, training scale, representation
  and tasks from this small residual-head pilot.
- **Perceptual evaluation.** Zhang et al., *The Unreasonable Effectiveness of
  Deep Features as a Perceptual Metric* (2018), introduces the LPIPS framework.
  [Paper](https://arxiv.org/abs/1801.03924).

This repository is an empirical implementation and matched comparison. It
does not replicate published paper scores or establish an algorithmic novelty.
In particular, our diffusion head supplements a frozen conventional decoder;
it is neither a replacement for all latent dynamics nor a scale-equivalent
implementation of GAIA-1/MCVD/DIAMOND.

Machine-readable citations: [references.bib](references.bib). Software citation:
[CITATION.cff](../CITATION.cff).
