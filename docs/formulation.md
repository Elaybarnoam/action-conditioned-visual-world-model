# Mathematical formulation

This document specifies the final phase-E implementation. The original
baseline and multistep learning formulation are preserved in the
[complete development record](development-record.md#problem-formulation), with
their relationship to later variants in the [milestone report](project-history.md).
It separates the physical
environment, observed evidence, learned temporal state and image generation.
Earlier experiment variants are described in [experiments](experiments.md).

## 1. Partially observed process

Let $s_t\in\mathcal S$ denote physical state, $a_t\in\mathbb R^4$ a recorded
action, and $o_t\in[0,1]^{3\times64\times64}$ the main-camera observation.
Conceptually,

$$
s_{t+1}\sim p_{\mathrm{env}}(\cdot\mid s_t,a_t),\qquad
o_t\sim p_{\mathrm{obs}}(\cdot\mid s_t).
$$

These distributions describe the source process; they are not supplied to the
learner. The learned latent state is not identified with physical state.
No robot-position measurement is used as a prediction input.

For a recording window beginning at $j$, the available evidence at $t=j+7$ is

$$
h_t^{(8)}=(o_j,a_j,o_{j+1},\ldots,a_{t-1},o_t).
$$

The supplied $a_t$ is applied to the transition whose target is $o_{t+1}$.
Tensor shapes are images $[B,8,3,64,64]$, past actions $[B,7,4]$, next action
$[B,4]$ and target $[B,3,64,64]$. Each window initializes memory anew; there is
no hidden carry from earlier windows. Recording time zero need not be the
start of the physical interaction.

Each 30-frame recording yields 22 one-step windows: $j=0,\ldots,21$.
Action 29 has no recorded frame 30 and is excluded from prediction pairs.
Pixel preprocessing is uint8 RGB to float32 CHW divided by 255, without resize.

## 2. History representation and latent transition

The encoder produces frame features and a spatial appearance pyramid:

$$
e_i=\operatorname{LN}(C_\phi(o_i)),\qquad
A_t=C_{\phi,\mathrm{pyramid}}(o_t).
$$

For the released backbone, $e_i\in\mathbb R^{256}$ and memory is 128 dimensional.
The first feature initializes a learned state; subsequent observed features
and preceding actions update it:

$$
z_j=\tanh(W_0 e_j+b_0),\qquad
z_i=\operatorname{GRU}_\phi([e_i,a_{i-1}],z_{i-1}),\quad i=j+1,\ldots,t.
$$

Layer normalization is per frame over 256 projected features, with no running
statistics. The action-conditioned dynamics MLP predicts

$$
\hat z_{t+1}=T_\psi(z_t,a_t).
$$

$T_\psi$ is a direct learned transition with final tanh, not a declared
physical integrator or a prescribed additive state update. Latent coordinates
are learned and have no assigned physical units.

The original appearance decoder predicts a coarse image

$$
b_{t+1}=D_\omega(\hat z_{t+1},A_t)\in[0,1]^{3\times64\times64}.
$$

Observed feature adapters connect the last real image pyramid to matching
decoder scales. This gives access to observed appearance without providing a
future image. In phase E, all parameters $\phi,\psi,\omega$ are frozen at
run-056; only a new residual image-generation head is trained.

## 3. Shared residual-head conditioning

Both phase-E heads receive the same target-free context:

$$
c_t=(o_t,\;o_t-o_{t-1},\;A_t,\;\hat z_{t+1},\;a_t,\;b_{t+1}).
$$

Raw images and the coarse image are mapped to $[-1,1]$ inside the head; the
observed difference is multiplied by two. The new head scales the four action
components by $(64,64,1,1)$ before its embedding. This is numerical scaling,
not a robot-to-pixel displacement calibration. The frozen backbone receives
the original raw action values.

No target, target-derived change mask or end-effector position enters $c_t$.
The implementation seam is `model.diffusion.observed_condition`.

## 4. Deterministic residual prediction

The deterministic head uses the same U-Net architecture as the diffusion head,
with a zero noisy-image input and timestep index zero. Let its output be
$R_\theta(c_t)$. Its prediction and objective are

$$
\hat o_{t+1}^{\mathrm{det}}
=\operatorname{clip}_{[0,1]}\left(b_{t+1}+\tfrac12R_\theta(c_t)\right),
$$

$$
\mathcal L_{\mathrm{det}}(\theta)
=\mathbb E_{(h_t,a_t,o_{t+1})\sim\mathcal D_{\mathrm{train}}}
\left[\operatorname{mean}\left((\hat o_{t+1}^{\mathrm{det}}-o_{t+1})^2\right)\right].
$$

Only $\theta$ receives gradients. The final head convolution is initially zero,
so the initial deterministic prediction equals the coarse backbone prediction.
Hidden parameters use random initialization, identically across the pair.

## 5. Conditional residual diffusion

Let $y=o_{t+1}$ and define the clean residual

$$
r_0=2(y-b_{t+1})\in[-2,2]^{3\times64\times64}.
$$

The diffusion index $n$ is distinct from physical time $t$. With $N=100$, set

$$
g(u)=\cos^2\left(\frac{u+0.008}{1.008}\frac\pi2\right),\quad
\beta_n=\min\left(1-\frac{g((n+1)/N)}{g(n/N)},0.999\right),
\quad \bar\alpha_n=\prod_{k=0}^{n}(1-\beta_k).
$$

For $n\sim\operatorname{Uniform}\{0,\ldots,N-1\}$ and independent
$\epsilon\sim\mathcal N(0,I)$, the corruption and v target are

$$
r_n=\sqrt{\bar\alpha_n}r_0+\sqrt{1-\bar\alpha_n}\epsilon,
$$

$$
v_n=\sqrt{\bar\alpha_n}\epsilon-\sqrt{1-\bar\alpha_n}r_0.
$$

The denoiser minimizes

$$
\mathcal L_{\mathrm{diff}}(\theta)
=\mathbb E_{\mathcal D,n,\epsilon}
\left[\operatorname{mean}\left((v_\theta(r_n,n,c_t)-v_n)^2\right)\right].
$$

Here v is a diffusion parameterization, not robot velocity. The real future
is used to construct corrupted training examples and their supervision.
At inference, it is absent: the sampler begins from independent Gaussian noise.

### DDIM sampling

The released sampler uses $\eta=0$ and 50 descending, rounded, uniformly
spaced indices from 99 to 0. At each sampled index,

$$
\hat r_0=\sqrt{\bar\alpha_n}r_n-
\sqrt{1-\bar\alpha_n}\,v_\theta(r_n,n,c_t).
$$

Clip $\hat r_0$ to $[-2,2]$ and recompute

$$
\hat\epsilon_n=
\frac{r_n-\sqrt{\bar\alpha_n}\hat r_0}{\sqrt{1-\bar\alpha_n}}.
$$

For the next smaller sampled index $m$, update

$$
r_m=\sqrt{\bar\alpha_m}\hat r_0+
\sqrt{1-\bar\alpha_m}\hat\epsilon_n.
$$

At the final index, output

$$
\hat o_{t+1}^{(k)}=
\operatorname{clip}_{[0,1]}\left(b_{t+1}+\tfrac12\hat r_0^{(k)}\right).
$$

Each initial noise image yields one prediction. Fixing the initial noise makes
sampling reproducible under the same execution settings. The method models a
conditional distribution; the seed is not a hidden correct action or state.
The current implementation retains the original decoder and generates a
residual correction. It does not replace the latent dynamics with diffusion.

## 6. EMA and matched optimization

Both heads train for 6,000 updates, batch 16, fresh Adam at $2\times10^{-4}$,
float32, and global gradient-norm clipping at 10. The private data permutation
and initial head tensors match. Each branch sees 96,000 one-step windows
spanning all 10,000 training recordings; this is less than one full 220,000-window pass.

EMA is updated after optimizer step $u$:

$$
d_u=\min\left(0.999,\frac{1+u}{10+u}\right),\qquad
\bar\theta_u=d_u\bar\theta_{u-1}+(1-d_u)\theta_u.
$$

Selection and inference use EMA weights. The comparison changes the objective
and inference procedure together. Equal updates and architecture do not imply
equal evaluation or sampling cost.

## 7. Metrics

For an image with $P=64\cdot64$ pixels and three channels,

$$
\operatorname{MSE}(\hat y,y)=\frac1{3P}\sum_{p,c}(\hat y_{p,c}-y_{p,c})^2.
$$

The target-derived changed-pixel mask is

$$
M_p=\mathbf1\left[\max_c|y_{p,c}-o_{t,p,c}|>0.05\right].
$$

Changed-region MSE averages the channel-mean squared error over pixels with
$M_p=1$. Empty-mask windows are excluded from this metric and counted in the
raw records. This mask includes lighting/noise and is not an object label.

Boundary MAE compares signed forward pixel differences:

$$
E_\nabla=\tfrac12\left(
\operatorname{mean}|\Delta_x\hat y-\Delta_x y|+
\operatorname{mean}|\Delta_y\hat y-\Delta_y y|\right).
$$

Differences use valid neighbors without padding, smoothing or thresholds.
SSIM uses uniform 7x7 valid windows, population moments, RGB data range 1,
$C_1=0.01^2$, $C_2=0.03^2$, and equal channel/location averaging. Small negative
numerical variances are clamped to zero. LPIPS uses official SqueezeNet
version-0.1 weights on RGB mapped to $[-1,1]$, without resizing.

Edge-energy ratio compares mean absolute neighbor differences with the target.
It is a sharpness proxy: noisy or misplaced edges can have a favorable ratio.
The reported fraction of channels exactly at 0 or 1 is a saturation proxy,
not a count of pre-clamp out-of-range values.

For diffusion, all $K=3$ fixed samples contribute equally to aggregate errors:

$$
\widehat E=\frac1{WK}\sum_{w=1}^{W}\sum_{k=1}^{K}
\ell(\hat y_w^{(k)},y_w).
$$

No oracle best-of-K selection or averaged-image prediction is used. The
algebraic sample-error decomposition is descriptive, not a calibration claim.

## 8. Selection and inference boundaries

Every 1,000 updates, evaluate 51 fixed validation windows, one per recording.
A promising checkpoint requires at least 5% LPIPS and 1% boundary improvement
relative to the original, while MSE and changed-region MSE stay within +5%.
Rank promising states first, then by the sum of normalized LPIPS and boundary
error. Both selected EMA states are at update 6,000.

The main table evaluates all 1,122 validation windows, including the selection
subset. It is not an independent final test. Phase E trains and evaluates only
one step. Earlier phases recursively advanced predicted latents using supplied
action prefixes, without future images or decoded-image feedback into the CNN.
Multistep validity of the new residual heads is untested.

## 9. Interpretation

For squared error, an unrestricted deterministic predictor estimates a
conditional mean. Averaging uncertain positions can produce blur; limited
representation or optimization can produce similar symptoms. Our experiments
do not isolate irreducible uncertainty as the cause. Diffusion can generate
sharper individual samples while increasing distance to the recorded target.
Sharpness, sample plausibility, target accuracy and calibration are distinct
properties. See [limitations](limitations.md) before extrapolating the results.
