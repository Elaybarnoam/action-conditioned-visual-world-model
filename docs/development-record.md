# Complete development and research record

This is the preserved project narrative, ordered from the initial mathematical
problem and module implementation through the final residual-generation study.
It retains the measured results, unsuccessful experiments, original decisions
and stage-specific limitations. Test counts and "current" statements inside
individual stages describe their historical execution, not the latest release.

The linked [run catalog](run-catalog.md) covers every saved record, including
failures and interactive sessions. Historical commands expect the referenced
runs to be restored locally first; see [artifact instructions](history-artifacts.md).
Source fingerprints identify the code used at training time; the present Git
history begins after those experiments. These are development experiments with
different protocols, rather than one interchangeable leaderboard table.

## Abstract

This project studies self-supervised, action-conditioned visual prediction
using recorded sequences from the BAIR Robot Pushing dataset. We propose a
world model trained from random weights, consisting of a CNN image encoder,
a GRU history representation, a deterministic latent dynamics network, and an
image decoder. The model summarizes past observations and actions in a latent
state, predicts its evolution under a supplied action, and generates the next
camera image. The initial training objective is one-step pixel prediction;
evaluation measures both one-step accuracy and recursive predictions without
future observations. We compare predictions with a copy-last-frame baseline
and inspect movement-sensitive errors and image sequences. The experiment
will test whether one-step image supervision learns a representation useful
for repeated latent prediction, while measuring failures caused by ambiguity,
blur, and mismatch between inferred and predicted latent states. No results
are implied by the formulation alone; the first experiment and its limitations
are reported below. Policy learning and planning are outside scope.


## Problem Formulation

Given observed history $h_t$ and a recorded action $a_t$, we learn a model
$f_\theta$ that predicts the next camera image. The following sections define
the hidden environment, available information, representation, prediction
pipeline, learning objective, and evaluation protocol.

### 1. Environment and observations

We consider a partially observed robotic environment represented by the BAIR
Robot Pushing dataset. A robot arm interacts with objects while cameras record
the scene. We learn from recorded sequences of images and actions.

At each discrete time step $t$, we distinguish three quantities:

- **Physical state** $s_t \in \mathcal{S}$: the configuration and motion of
  the robot and objects, together with other physical properties needed to
  describe their evolution. The complete physical state is not available to
  our model. It is different from the learned latent state defined later.
- **Observation** $o_t \in \mathcal{O}$: an RGB image from one selected camera,
  where $\mathcal{O} = \{0,\ldots,255\}^{64\times64\times3}$ describes the raw
  pixel values. An image does not necessarily reveal the complete physical
  state: objects can be occluded, and a single frame may not reveal velocity.
- **Action** $a_t \in \mathcal{A} \subseteq \mathbb{R}^{4}$: the recorded
  four-component robot action associated with the transition after time $t$.
  The implemented loader preserves the four recorded components and pairs
  action index $t$ with image $t$ and target image $t+1$. See the verified
  dataset inspection below for source evidence and timing diagnostics.

The image dimensions and action-vector size follow the
[BAIR dataset specification](https://www.tensorflow.org/datasets/catalog/bair_robot_pushing_small).
Although robot-position measurements are also provided, our proposed model
uses images and actions as inputs rather than receiving the complete physical
state.

#### Underlying environment process

We model the environment's evolution using a transition distribution:

$$
s_{t+1} \sim p_{\mathrm{env}}(\,\cdot \mid s_t, a_t).
$$

In words: given the current physical state and action, the environment produces
a next physical state. The symbol $\sim$ means "is drawn from," and the dot is
a placeholder for a possible next state. The distribution allows variability
in the transition. We assume $s_t$ contains the physical information needed to
describe this transition given $a_t$.

For a deterministic environment, this simplifies to:

$$
s_{t+1} = F_{\mathrm{env}}(s_t, a_t).
$$

The camera observation is described by an observation distribution:

$$
o_t \sim p_{\mathrm{obs}}(\,\cdot \mid s_t).
$$

In words: the camera produces an image of the physical state, potentially with
observation noise. Even without noise, different physical states can produce
the same image, so the state may remain partially hidden.

These equations describe the environment that generated the recordings. The
transition and observation distributions are not given to our model; the
learned prediction model will be defined in the following sections.

### 2. Available history

At prediction time $t$, the available evidence consists of observations up to
and including $o_t$, and actions applied before that observation. We define
the history as:

$$
h_t = (o_0, a_0, o_1, a_1, \ldots, o_{t-1}, a_{t-1}, o_t),
\qquad h_0 = (o_0).
$$

Each action $a_i$ is associated with the transition from observation $o_i$ to
$o_{i+1}$. The action $a_t$ is supplied separately: it is the action whose
effect we want to predict, rather than an action already included in $h_t$.
Thus, the information available for a one-step prediction is $(h_t, a_t)$.
The actual next observation $o_{t+1}$ is a prediction target, not part of the
available history for this prediction.

History can reveal information missing from a single image. For example,
successive frames can indicate an object's motion, while earlier frames may
show an object that is currently occluded. History can reduce uncertainty
about the physical state without necessarily identifying it completely.

#### Relationship to the latent state

The history $h_t$ is the recorded evidence; the latent state $z_t$ is a learned
representation constructed from that evidence. Conceptually, the prediction
pipeline is:

```text
Available history → current latent state
Current latent state + action → predicted next latent state
Predicted next latent state → predicted next image
```

The image is therefore generated from the predicted latent state. Defining
the history specifies what information may be used to construct that state;
it does not require the decoder to receive the raw history directly.

The full history is a mathematical description of the available information.
Our proposed baseline processes it through the recurrent memory in Section 4.
If implementation uses shorter context windows, the available prefix and
memory initialization must be recorded explicitly. When using recorded sequences, time zero
denotes the start of the available recording, not necessarily the start of
the underlying physical interaction.

### 3. Prediction task

Given the observation-action history $h_t$ up to time $t$ and a supplied action
$a_t$, the world model predicts the next camera observation:

$$
\hat{o}_{t+1} = f_\theta(h_t, a_t).
$$

Here, $\hat{o}_{t+1}$ is the predicted image, while $o_{t+1}$ is the actual next
image in the recording. The function $f_\theta$ denotes the complete world
model: history encoding, latent dynamics, and image decoding. Its trainable
parameters $\theta$ include the parameters of all three components, which we
plan to initialize randomly and train jointly.

The initial formulation uses a deterministic prediction: for a given history
and action, the model produces one predicted image. This does not assume that
the underlying environment is deterministic or that the history identifies
one unique physical state. Explicitly modeling a distribution over possible
future images is left for a later experiment.

#### One-step training

Each training example supplies $(h_t, a_t)$ as input and the recorded next
image $o_{t+1}$ as the target. The actual next image is not provided as an
input to that prediction. History can contain multiple past frames even
though the target is only one future frame.

This is self-supervised learning: the recorded sequence provides the target
without requiring a person to annotate the correct future image. Prediction
error trains the encoder, dynamics model, and decoder through backpropagation.
The training loss will be defined in Section 6.

#### Recursive multi-step evaluation

To evaluate a horizon $H \geq 1$, we provide the history $h_t$ and the recorded
action sequence $(a_t, \ldots, a_{t+H-1})$. Starting from the latent state
constructed from $h_t$, the model repeatedly predicts the next latent state
and decodes images $\hat{o}_{t+1}, \ldots, \hat{o}_{t+H}$. No actual future
observations are supplied during this rollout. Actions are given, not chosen
by the model.

We compare these predicted images with the corresponding recorded images
after the rollout. Evaluation does not update weights. Our initial setup is
therefore **one-step training with one-step and recursive multi-step
evaluation**, rather than training on a multi-step rollout loss.

### 4. Latent representation

We represent the available history $h_t$ using a learned latent vector:

$$
z_t \in \mathcal{Z} = \mathbb{R}^{d}.
$$

The latent state $z_t$ summarizes information useful for predicting future
observations. It is not the true physical state $s_t$. Its coordinates are
learned features, not predefined fields such as position or velocity. The
latent space $\mathcal{Z}$ is the set of possible vectors of this shape; $z_t$
is one particular vector in that space. We will choose the dimension $d$
during implementation.

#### Visual feature extraction

A convolutional neural network (CNN) encodes the current camera observation:

$$
e_t = C_{\theta_C}(\tilde{o}_t), \qquad e_t \in \mathbb{R}^{m},
\qquad \tilde{o}_t = o_t / 255.
$$

Here, $e_t$ contains features extracted from the current image, and $m$ is the
visual-feature dimension. The CNN alone does not summarize earlier images or
actions; temporal information is incorporated by the memory update below.
Pixels are normalized to $[0,1]$. The implemented baseline uses four
Conv2d/ELU stages, flattening, and a linear projection into $m=256$ features.
Its layer dimensions and inspection command are documented in Step 2 below.

The complete training configuration applies parameter-free layer normalization
to each projected feature vector before memory initialization and updates:

$$
\bar e_t = \frac{e_t-\mu(e_t)}{\sqrt{\operatorname{var}(e_t)+10^{-5}}}.
$$

Here mean and variance are over that frame's 256 features (population variance).
In the GRU and initialization equations below, use $\bar e_t$ in place of
$e_t$ when `normalize_features=true`. This uses no batch statistics, future
frames, or extra learned parameters. The standalone encoder/memory inspections
preserve their original unnormalized behavior. Normalization was introduced
after the first training attempt saturated the GRU and lost encoder gradients;
the measured diagnostic is recorded below.

#### Recurrent memory update

For $t \geq 1$, a Gated Recurrent Unit (GRU) combines the current visual
features and previous action with the previous latent state:

$$
z_t = \operatorname{GRU}_{\theta_R}
\left([e_t, a_{t-1}], z_{t-1}\right).
$$

The brackets $[e_t, a_{t-1}]$ denote concatenation. The inputs are:

- $e_t$: features of the current observation.
- $a_{t-1}$: the action associated with the transition to the current observation.
- $z_{t-1}$: memory summarizing earlier observations and actions.

The GRU's updated hidden vector is the latent state $z_t$. Its learned gates
control how previous memory contributes to candidate new information and how
much old memory is retained or replaced. This is a learned transformation of
the inputs, not simply storage of their concatenation.

The representation may encode information about appearance, locations, motion,
and objects that are currently occluded. Prediction training encourages useful
information to be retained, but does not guarantee that every physical detail
is preserved or that individual coordinates have interpretable meanings.

#### Initial latent state

At the first available observation, there is no previous recorded action or
memory. We initialize the latent state from the first image's features:

$$
e_0 = C_{\theta_C}(\tilde{o}_0), \qquad
z_0 = I_{\theta_I}(e_0),
$$

where $I_{\theta_I}: \mathbb{R}^{m} \rightarrow \mathbb{R}^{d}$ is a learned
initialization mapping. The implemented baseline uses
$I_{\theta_I}(e_0)=\tanh(W_I e_0+b_I)$, with a linear layer from 256 features
to 128 latent coordinates followed by tanh.

Unrolling the memory updates gives a history-encoding system that can be
written conceptually as $z_t = E_{\theta_E}(h_t)$, with
$\theta_E = (\theta_C, \theta_R, \theta_I)$. The CNN, GRU, and initialization
mapping start with random weights and are trained jointly with the dynamics
model and decoder through prediction error.

The latent vector summarizes history rather than retaining all raw past images
and actions. Constructing $z_t$ uses observed evidence; predicting its future
evolution without new observations is the role of the dynamics model in
Section 5.

### 5. Latent dynamics and decoding

Starting from the latent state inferred from observed history, the world model
predicts the next latent state and decodes it into an image. We use a
deterministic latent dynamics model as the initial experimental baseline.

#### One-step latent prediction

The dynamics network receives the current latent state and supplied action:

$$
\hat{z}_{t+1} = T_{\theta_T}(z_t, a_t),
\qquad
T_{\theta_T}: \mathbb{R}^{d} \times \mathcal{A} \rightarrow \mathbb{R}^{d}.
$$

Unlike the GRU update in Section 4, this prediction does not receive the next
observation. The GRU incorporates observed evidence into memory; the dynamics
network predicts how the representation evolves before that evidence is
available. Both $z_t$ and $\hat{z}_{t+1}$ are vectors, while $T_{\theta_T}$ is
the trainable network acting on them.

#### Image decoding

The decoder maps the predicted latent state to a predicted RGB image:

$$
\hat{o}_{t+1} = D_{\theta_D}(\hat{z}_{t+1}),
\qquad
D_{\theta_D}: \mathbb{R}^{d} \rightarrow [0,1]^{64\times64\times3}.
$$

We express decoder outputs as continuous, normalized pixel values. Recorded
images $o_t$ retain the raw-image definition in Section 1; for training and
comparison with decoder outputs, we use $\tilde{o}_t = o_t / 255$. The hat
denotes a prediction rather than an observed target. Displaying the prediction
can convert normalized values back to the raw pixel range; integer conversion
is not part of the differentiable training path.

The exact dynamics and decoder layers remain implementation choices. Their
parameters start randomly and are trained jointly with the history-encoding
system. The complete one-step predictor can be written as:

$$
f_\theta(h_t, a_t)
= D_{\theta_D}\!\left(T_{\theta_T}(E_{\theta_E}(h_t), a_t)\right),
\qquad
\theta = (\theta_E, \theta_T, \theta_D).
$$

#### Recursive prediction without future observations

For a horizon $H \geq 1$, initialize the rollout from observed history:

$$
\hat{z}_t = z_t = E_{\theta_E}(h_t).
$$

For $k = 0, \ldots, H-1$, apply the supplied actions recursively:

$$
\hat{z}_{t+k+1}
= T_{\theta_T}(\hat{z}_{t+k}, a_{t+k}),
\qquad
\hat{o}_{t+k+1} = D_{\theta_D}(\hat{z}_{t+k+1}).
$$

The predicted latent state becomes the dynamics input for the next step.
Decoded images are outputs for inspection and evaluation; they are not fed
back through the CNN. Actual future images are also excluded from this
rollout. Observed-history encoding and prediction without new observations
are therefore distinct operations.

#### Consistency between inferred and predicted latent states

After observing the actual next image, the history-encoding system could
instead construct:

$$
z_{t+1}
= \operatorname{GRU}_{\theta_R}
\left([C_{\theta_C}(\tilde{o}_{t+1}), a_t], z_t\right).
$$

This observation-informed state $z_{t+1}$ and the prediction
$\hat{z}_{t+1} = T_{\theta_T}(z_t, a_t)$ belong to the same vector space,
but are not guaranteed to agree. In the initial image-loss-only baseline,
training supervises the decoded next image, not an explicit match between
these latent states. A vector that decodes to an accurate image is not
necessarily a useful input for another dynamics prediction.

This creates a potential mismatch: one-step training starts from latent states
inferred using real observations, whereas recursive prediction repeatedly
uses predicted latent states. We will measure its effect rather than assume
that good one-step images imply stable multi-step predictions.

Recurrent state-space models such as
[Dreamer](https://arxiv.org/html/1912.01603v3) address a related issue by training
observation-informed and observation-free latent distributions with a
consistency regularizer. Our deterministic baseline is simpler and is not an
implementation of that probabilistic formulation.

Section 6 will define the baseline training objective. A latent-consistency
objective and multi-step training are candidate follow-up experiments, not
part of the initial baseline. Any latent-matching objective needs a defined
target-update scheme and consideration of representation collapse; merely
making two jointly learned vectors equal does not guarantee useful features.

### 6. Training objective

#### Dataset and training examples

Let the training dataset contain $N$ recorded trajectories:

$$
\mathcal{D}_{\mathrm{train}} = \{\tau^{(i)}\}_{i=1}^{N},
\qquad
\tau^{(i)} = (o^{(i)}_0, a^{(i)}_0, \ldots,
o^{(i)}_{L_i-1}, a^{(i)}_{L_i-1}, o^{(i)}_{L_i}).
$$

Trajectory $i$ contains $L_i$ valid transitions and $L_i+1$ observations.
Each transition supplies a prediction example
$(h^{(i)}_t, a^{(i)}_t, o^{(i)}_{t+1})$. The GRU processes real observations
and past actions in chronological order to construct $z^{(i)}_t$. We reset
memory between trajectories and never carry it across unrelated recordings.
Image/action alignment and valid sequence lengths must be verified during
dataset preparation.

#### One-step image loss

The baseline minimizes mean squared error (MSE) between predicted and actual
normalized pixels. Let $P = 64\cdot64\cdot3$ be the number of image values:

$$
\ell^{(i)}_t(\theta)
= \frac{1}{P}
\left\|f_\theta(h^{(i)}_t,a^{(i)}_t)
- \tilde{o}^{(i)}_{t+1}\right\|_2^2,
\qquad \tilde{o}^{(i)}_{t+1}=o^{(i)}_{t+1}/255.
$$

The norm sums squared errors over all pixel locations and color channels.
The full empirical objective gives equal weight to each valid transition:

$$
\theta^* \in \operatorname*{arg\,min}_{\theta}
\mathcal{L}_{\mathrm{train}}(\theta),
\qquad
\mathcal{L}_{\mathrm{train}}(\theta)
= \frac{1}{\sum_{i=1}^{N} L_i}
\sum_{i=1}^{N}\sum_{t=0}^{L_i-1}\ell^{(i)}_t(\theta).
$$

This notation defines the optimization goal, not a guarantee that training
finds a global minimum. Mini-batch training approximates this objective;
padding is masked and does not contribute to the loss. Any nonuniform
sequence/window sampling must be documented because it can change which
transitions receive more weight.

#### Joint optimization and temporal gradients

All trainable parameters are optimized jointly:

$$
\theta=(\theta_C,\theta_R,\theta_I,\theta_T,\theta_D).
$$

Gradients flow from the image loss through the decoder and dynamics network
into the CNN, GRU, and initial-state mapping. Backpropagation through time
also accounts for the recurrent updates used to form the current state.
Optimizer, learning rate, batch size, context length, latent dimensions,
gradient clipping, and any truncation of temporal gradients will be fixed
and recorded before running the baseline.

We calculate one-step losses at multiple time points within a sequence. When
advancing to the next training time point, the GRU incorporates the next real
observation. That observation was the preceding prediction's target; it was
not available as an input to that preceding prediction. This is one-step
supervision across a sequence, not a loss on an observation-free recursive
rollout.

#### Baseline boundaries and limitations

The initial objective contains no latent-matching, multi-step rollout,
reward, or policy loss. It is self-supervised because recorded observations
provide the targets. Validation selects checkpoints without gradient updates;
test data is reserved for the final evaluation.

Pixel MSE can be dominated by static background and can favor averaged,
blurry images when several futures are compatible with the history. A small
image loss does not establish physical understanding or latent consistency.
We therefore evaluate movement and recursive behavior separately. Latent
consistency and multi-step losses remain controlled follow-up experiments;
their targets and weighting must be defined before using them.

### 7. Evaluation

#### Frozen splits and checkpoints

Use one selected camera consistently and preserve the supplied test split.
Create training and validation partitions from training recordings using a
recorded seed, keeping each complete trajectory in exactly one split.
Select an initial working subset by trajectory and save its identifiers,
source version, preprocessing, and action alignment. Overlapping windows
from one recording must not leak into another split.

Save random-weight, periodic, best-validation, and final checkpoints. Choose
the best checkpoint using one-step validation pixel MSE for the baseline;
report the selection rule and epoch. Evaluate all checkpoints being compared
on the same recorded windows and action sequences, with fixed weights and
without using future images as prediction inputs.

#### One-step and recursive horizon error

At each evaluation start, infer $z_t$ from a declared observed prefix. For
one-step prediction, apply $T$ once and decode. For longer horizons, apply
the observation-free rollout in Section 5. Proposed horizons are
$\mathcal{H}=\{1,5,10,20\}$, subject to the verified recording lengths.

For split $S$ and horizon $h$, let $\mathcal{W}_{S,h}$ be the eligible
trajectory/start pairs with enough context and at least $h$ future actions.
Measure endpoint pixel error as:

$$
\operatorname{MSE}_{S}(h)
= \frac{1}{|\mathcal{W}_{S,h}|}
\sum_{(i,t)\in\mathcal{W}_{S,h}}
\frac{1}{P}
\left\|\hat{o}^{(i,t)}_{t+h}
- \tilde{o}^{(i)}_{t+h}\right\|_2^2.
$$

Here, $\hat{o}^{(i,t)}_{t+h}$ is predicted by a rollout starting at time $t$
in trajectory $i$. Each scored future frame is withheld throughout that
rollout. Windows remain inside one trajectory, and padded or missing frames
are excluded. Report context length and sample counts at each horizon;
overlapping windows are not independent samples. For direct horizon
comparisons, also use a common set of starts valid at the largest horizon.

#### Copy-last-frame baseline

The baseline predicts no visual change at any horizon:

$$
\hat{o}^{\mathrm{copy}}_{t+h} = \tilde{o}_t,
\qquad h\geq1.
$$

Compute its error on exactly the same windows as the learned model. Beating
this baseline indicates better prediction than assuming nothing changes;
it does not establish correct dynamics in every scene.

#### Movement-sensitive diagnostics and visual inspection

In addition to whole-image MSE, report an error on pixels that change between
the starting image and the target. For a fixed, documented threshold
$\epsilon>0$, define the diagnostic mask:

$$
M^{(i,t,h)}_{uv}
= \mathbf{1}\!\left[
\max_{c\in\{1,2,3\}}
\left|\tilde{o}^{(i)}_{t+h,uvc}
-\tilde{o}^{(i)}_{t,uvc}\right|>\epsilon
\right].
$$

For each window with a nonempty mask, calculate:

$$
\operatorname{MSE}_{\mathrm{change}}(i,t,h)
= \frac{
\sum_{u,v}M^{(i,t,h)}_{uv}
\sum_{c=1}^{3}
\left(\hat{o}^{(i,t)}_{t+h,uvc}
-\tilde{o}^{(i)}_{t+h,uvc}\right)^2
}{3\sum_{u,v}M^{(i,t,h)}_{uv}}.
$$

Average this diagnostic over eligible windows and report their count; report
empty-mask windows separately. The mask is computed from real frames only
for scoring, never supplied to the predictor. It highlights observed image
change rather than supplying object labels or a precise motion estimate.
Camera noise and lighting changes can also affect it, and false predicted
motion outside the mask is captured by whole-image MSE.

Save side-by-side actual, predicted, and copy-baseline sequences, together
with error maps and error-by-horizon plots. Select example recordings by a
declared rule independent of prediction quality, such as fixed episode IDs.
Inspect blur, object disappearance, contact behavior, occlusion, and drift.

#### Testing whether history and actions are useful

After establishing the baseline, compare a single-frame variant with the
history-based model under matched data and training budgets. Separately,
evaluate the effect of removing or perturbing action inputs. These are
diagnostics of dependence on history and actions, not proof of causal
correctness; perturbed actions may fall outside the recorded distribution.

Any later distribution-shift benchmark must name the shift and verify that
the data supports it. We do not assume that a random trajectory split tests
unseen objects, viewpoints, or physical conditions. If evaluation results
guide model changes, treat that set as monitoring data and retain an
untouched set for a final comparison.


## Learning objective

```text
Input:  recorded image/action history + supplied next robot action
Output: predicted next camera image
```

Use existing BAIR camera recordings and action logs rather than generating a
synthetic environment. Learn the CNN, recurrent representation, latent dynamics,
and decoder from scratch. The objective is to understand what visual and
temporal information supports prediction, and where that prediction fails.


## What we will build

1. A reproducible loader and inspection tools for a selected BAIR subset.
2. A CNN/GRU latent world model with action-conditioned dynamics and decoding.
3. A training loop with checkpoints and held-out evaluation.
4. Side-by-side actual and predicted images, including recursive sequences.
5. Experiments that explain what the model learned and where it fails.

There is no planner or policy network in this practice. Data preparation and
evaluation code exist to train and inspect the world model.


## How we will measure progress

- Compare next-frame pixel error against copying the current image unchanged.
- Measure changed-region error separately, because a mostly static background
  can make an incorrect prediction look good under average pixel error.
- Inspect image sequences at several recursive prediction horizons.
- Keep entire trajectories in one dataset split to avoid episode leakage.
- Later test a verified distribution shift supported by available recordings,
  with a separate evaluation set and no updates using those evaluation targets.

See [PLAN.md](project-history.md) for implementation missions.


## Run Step 0: Python setup check

This practice uses Python 3.11.15, PyTorch 2.13.0 with CUDA 13.0, and `uv` for
the isolated environment and dependency lock. The CUDA build is sourced from
the [official PyTorch wheel index](https://pytorch.org/get-started/previous-versions/).
`uv.lock` fixes resolved package versions; `.python-version` pins Python 3.11.15.
The installed interpreter and package versions are recorded in each run.

From `visual-world-model/`:

```powershell
uv sync --locked
uv run --locked python -m src.setup_check
```

This creates `.venv/` and executes on `cuda:0` by default. Request a CPU check
explicitly with:

```powershell
uv run --locked python -m src.setup_check --device cpu
```

There is no silent device fallback. `uv` uses copy mode because this workspace
is under OneDrive, where hardlinks can cause installation failures. If a prior
installation was interrupted, repair it with `uv sync --locked --reinstall`.
Install `uv` following its [official instructions](https://docs.astral.sh/uv/getting-started/installation/)
if it is not available. Environment activation is optional when using `uv run`.

### Setup data flow

```text
configs/baseline.json → Config validation and tensor shapes
                     → New run directory
                     → Explicit device selection and seeding
                     → Tensor operation, known gradient, one SGD update
                     → Metrics, environment, source fingerprint, and summary
```

- `src/config.py` defines configuration and tensor conventions.
- `src/run.py` allocates unique run directories, writes JSON/JSONL records, and
  records successful completion or failures without swallowing exceptions.
- `src/setup_check.py` is the Step 0 entry point. It is not the future training
  entry point and does not instantiate a world model or load BAIR.
- Tests are in dedicated files under `tests/`.

The check evaluates a known scalar problem, $L(w)=(w-3)^2$, at $w=1$.
The expected loss is 4 and the gradient is -4. With learning rate 0.1, one SGD
update gives $w=1.4$ and loss 2.56. It also verifies a known matrix product
and allocates dummy tensors with the configured shapes.

Current shape conventions are:

| Quantity | Shape |
|---|---|
| Context images | `[B, T, 3, 64, 64]` |
| Past context actions | `[B, T-1, 4]` |
| Next supplied action | `[B, 4]` |
| Visual features | `[B, m]` |
| Latent memory | `[B, d]` |
| Target / predicted image | `[B, 3, 64, 64]` |

The current configuration uses B=2, T=8, m=256, d=128, seed 3, and float32.
Step 2 increased the original setup's feature size from 64 to 256.
These are starting dimensions for module development, not tuned model sizes.
`check_learning_rate` belongs only to the scalar setup exercise; it is not
the learning rate selected for future world-model training. T observed frames
contain T-1 past actions; the next action $a_t$ is supplied separately.

Every execution gets a new `evaluation/runs/run-NNN/` containing configuration,
environment/device/dependency metadata, source-file hashes and Git version if
available, tensor shapes, metrics, a status file, and a summary. The setup
dataset manifest explicitly says that no dataset was used. Predictions and
model checkpoints are added by later modules, not fabricated during setup.
Seeds and deterministic settings are recorded; identical results across
different hardware or software versions are not guaranteed.

Run focused checks with:

```powershell
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Modules live directly under `src/`. Run entry points as `python -m src.<module>`
from the project root, which keeps imports explicit without modifying
`sys.path` in the application. The training entry point is
`python -m src.train`; its settings are in `configs/train.json`.

### Verified setup result — 2026-10-16

The check passed on the NVIDIA GeForce RTX 5060 Laptop GPU (about 8 GB VRAM)
and CPU. The analytic gradient matched -4, and the SGD update reduced loss
from 4.00 to 2.56. All 22 tests, Ruff linting, and formatting checks pass.
The generated setup runs were subsequently cleared at the user's request.
Results are local and ignored by Git; rerun the setup command to generate
fresh records. The measurements above describe the completed verification.


## Step 1: Fetch and verify BAIR

```powershell
uv sync --locked
uv run --locked python -m src.inspect_data
```

The command discovers the first training and test TFRecord members in the
[original BAIR archive](https://sites.google.com/view/sna-visual-mpc/), then
uses HTTPS byte ranges to download those complete members. The selected
members total 379,704,320 bytes (about 362 MiB), rather than downloading the
entire 32,274,964,480-byte archive. Repeated runs verify and reuse cached
downloads and prepared files. Network access is needed to check archive
identity; the PyTorch dataset itself loads prepared data offline.

### Data layout and verified counts

```text
data/bair/
  raw/                      original TFRecord members and download receipts
  prepared/
    manifest.json           source ranges/ETag, SHA256, episode IDs, splits
    episodes/*.npz          raw uint8 camera frames, actions, diagnostic positions
evaluation/runs/run-002/
  dataset-manifest.json
  alignment.json
  sample.json
  inspection/
    image_main.gif
    image_aux1.gif
    image_main-frames.png
    image_aux1-frames.png
    context-and-target.png
    actions-and-positions.csv
```

Initial verification on 2026-10-16 produced:

| Split | Recordings | Frames | Valid transitions | Eight-frame context windows |
|---|---:|---:|---:|---:|
| Train | 205 | 6,150 | 5,945 | 4,510 |
| Validation | 51 | 1,530 | 1,479 | 1,122 |
| Test | 256 | 7,680 | 7,424 | 5,632 |

The selected training shard contains 256 recordings; seed 3 selects 51 whole
recordings for validation. The downloaded test shard contains 256 recordings
and stays entirely separate. This is a small development training subset,
not the complete BAIR training corpus or a verified OOD benchmark.

Every recording contains 30 raw RGB frames from each of two cameras, 30
four-component action vectors, and 30 three-component end-effector positions.
The loader uses `image_main` by default; `image_aux1` is also supported.
Action 29 is retained in storage but excluded from next-frame prediction,
because frame 30 is unavailable. Thus each recording has 29 valid transitions.
These keys and raw-image decoding follow the
[official TFDS reader](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/bair_robot_pushing_small/bair_robot_pushing_small_dataset_builder.py).

### Frame/action alignment

The released controller's `apply_act` uses horizontal x/y increments, a
gripper-close duration command, and a lift duration command. We retain all
four raw values without rescaling. They are commands, not measured joint
velocities or exact observed displacement. See the
[released controller](https://github.com/febert/visual_mpc/blob/master/python_visual_mpc/sawyer/visual_mpc_rospkg/src/visual_mpc_client.py).

As a supporting check on training recordings only, action[t] x/y correlated
with measured position[t+1]-position[t] at 0.888 / 0.804. The corresponding
correlations for action[t-1] were 0.385 / 0.569 and action[t+1] 0.491 / 0.233.
This supports using the same-index action for the next-frame transition.
Repeated commands and control lag prevent treating correlation as independent
proof of precise physical timing. The records do not provide a verified
per-frame timestamp; GIF playback speed is for inspection, not measured time.

### Public loader and tensor flow

```python
from pathlib import Path
from torch.utils.data import DataLoader
from src.data.dataset import BairDataset

dataset = BairDataset(Path("data/bair/prepared"), "train", context_length=8)
sample = dataset[0]
batch = next(iter(DataLoader(dataset, batch_size=2)))
```

Each sample returns eight frames o[start]..o[start+7], past actions
a[start]..a[start+6], next action a[start+7], and target o[start+8]. It includes
episode ID and frame indices for tracing predictions. Pixels become float32
CHW tensors in [0,1]. Actions stay float32. End-effector positions are saved
for diagnostics and are never returned as model inputs.

The context is a finite window: memory is initialized at its first frame,
rather than carrying state from earlier windows. Windows remain within a
single recording and may overlap within their assigned split. An eight-frame
context yields 22 one-step windows per recording. The last uses action 28
to predict frame 29. A one-frame context is also supported with no past actions.

### Integrity and recorded inspection

Verification checks exact HTTP range responses and archive ETag, tar-header
checksums, every TFRecord length/payload CRC32C, required feature types,
image/array shapes, finite values, file SHA256, split separation, and loader
tensor shapes. Local SHA256 receipts detect later modification; they are not
an independently published upstream member digest. NumPy loading disables
pickle. Corrupted caches and inconsistent existing manifests are rejected.

The source flow is `fetch.py` -> `records.py` -> `prepare.py` -> `dataset.py`;
`src/inspect_data.py` connects them and saves the inspection run. TensorFlow
is not installed: the lightweight `tfrecord` package supplies protobuf message
definitions, while our reader checks framing with CRC32C.

See the [verification summary](run-catalog.md#run-002),
[context/target example](run-catalog.md#run-002),
and [main-camera animation](run-catalog.md#run-002).
The fetch-only record is run-001; the full dataset verification is run-002.
All 46 tests pass, including corruption checks, action/target indexing, split
leakage checks, and an offline end-to-end inspection using synthetic fixtures.
No world-model weights were created or trained in this step.


## Step 2: CNN image encoder

`src/model/encoder.py` implements $e_t=C_{\theta_C}(\tilde{o}_t)$ as a
PyTorch `ImageEncoder`. Input pixels are float32 `[B,3,64,64]` tensors already
normalized by the loader. Output features are `[B,256]` and are not constrained
to the image pixel range.

| Layer | Output excluding batch |
|---|---|
| Input | `3 x 64 x 64` |
| Conv2d 3 -> 32 + ELU | `32 x 32 x 32` |
| Conv2d 32 -> 64 + ELU | `64 x 16 x 16` |
| Conv2d 64 -> 128 + ELU | `128 x 8 x 8` |
| Conv2d 128 -> 128 + ELU | `128 x 4 x 4` |
| Flatten | `2048` |
| Linear 2048 -> 256 | `256` |

All convolutions use kernel 4, stride 2, and padding 1. There is no pooling,
normalization, dropout, or activation after the final linear projection.
The 952,416 parameters use standard PyTorch random initialization, controlled
by the run seed. Feature size is configurable; the baseline sets it to 256.
The latent-memory size is separately configured as 128 and will be implemented
in Step 3. The encoder processes images only: it receives no action, previous
memory, or prediction target.

```python
from src.config import Config
from src.model import ImageEncoder

config = Config.load()
encoder = ImageEncoder(config.feature_size)
features = encoder(batch["context_images"][:, -1])  # Current frame only
```

For a sequence, the same encoder can independently process `B*T` frames and
reshape the output to `[B,T,256]`. That introduces no temporal memory; the GRU
will combine those features with actions over time.

Run the recorded inspection on real BAIR data:

```powershell
uv run --locked python -m src.inspect_encoder
# Optional CPU execution:
uv run --locked python -m src.inspect_encoder --device cpu
```

The inspection selects the first training windows without shuffling, traces
every layer on the configured device, and saves:

- Configuration, environment, source fingerprint, dataset manifest, and summary.
- `encoder-trace.json`: frame IDs, layer shapes/statistics, and gradient norms.
- `inspection/input-frame.png`: the actual current frame.
- `inspection/untrained-feature-maps.png`: the first eight channels at each
  convolutional stage after ELU, with independent per-tile contrast scaling.
- `inspection/features.npz`: current-image and per-context-frame feature arrays.
- `checkpoints/encoder-initial.pt`: random initial weights, verified by reloading
  and reproducing the same outputs.

A diagnostic objective, `mean(features**2)`, checks that gradients reach every
parameter and the input pixels. It is not our future image-prediction loss.
No optimizer step occurs and all weights remain unchanged. Untrained feature
maps are not learned object detectors, reconstructions, or evidence of useful
prediction. Feature quality will be evaluated during joint training later.

The GPU inspection completed in `evaluation/runs/run-003`, producing current
features `[2,256]` and sequence features `[2,8,256]`. All 58 tests, Ruff
linting, and formatting checks pass. See the
[encoder inspection summary](run-catalog.md#run-003) and
[activation maps](run-catalog.md#run-003).


## Step 3: GRU memory and initialization

`src/model/memory.py` implements `LatentMemory`, the observed-history portion
of $E(h_t)$. The baseline uses:

$$
z_0=\tanh(W_I e_0+b_I),\qquad
z_t=\operatorname{GRUCell}([e_t,a_{t-1}],z_{t-1}),\quad t\geq1.
$$

The initialization layer maps 256 image features to 128 latent coordinates.
The GRUCell receives 260 input values (256 features plus four previous-action
components) and a 128-value previous hidden state. Together, initialization
and GRU contain 182,656 trainable parameters. Parameters use standard PyTorch
initialization, controlled by the recorded seed.

| API | Inputs | Output |
|---|---|---|
| `initialize` | First features `[B,256]` | Initial state `[B,128]` |
| `step` | Current features `[B,256]`, previous action `[B,4]`, previous state `[B,128]` | Updated state `[B,128]` |
| `forward` | Features `[B,T,256]`, past actions `[B,T-1,4]` | All states `[B,T,128]` |

Each `forward` call initializes a fresh context; the module stores no mutable
episode memory. Callers continuing an observed sequence pass the previous state
explicitly to `step`. The sequence retains its gradient graph for learning
through time. A one-frame context returns only the initialized state and takes
an empty `[B,0,4]` past-action tensor. No fake action is inserted at time zero.

```python
from src.model import ImageEncoder, LatentMemory

encoder = ImageEncoder(config.feature_size)
memory = LatentMemory(config.feature_size, config.action_size, config.latent_size)
images = batch["context_images"]
B, T, C, H, W = images.shape
features = encoder(images.reshape(B * T, C, H, W)).reshape(B, T, -1)
states = memory(features, batch["context_actions"])
z_t = states[:, -1]
```

For the default context, frame 0 constructs z_0 without an action; frame 1 and
action 0 update z_1, continuing through frame 7 and action 6 to z_7. Next action
a_7 and target image o_8 are excluded from this history encoder. Step 4 will
use z_7 and a_7 to predict the next latent state.

Run the real-data inspection:

```powershell
uv run --locked python -m src.inspect_memory
# Optional CPU execution:
uv run --locked python -m src.inspect_memory --device cpu
```

The GPU inspection completed in `evaluation/runs/run-004`. It saves
`memory-trace.json`, numerical features/actions/states in
`inspection/memory.npz`, and `checkpoints/encoder-memory-initial.pt`, along with
the standard configuration, dataset manifest, environment, metrics, and summary.
The saved initial encoder/memory checkpoint reloads with identical outputs.

Checks verify that later features/actions do not change earlier states, a
truncated prefix gives the same prefix states, and explicit incremental updates
match sequence processing. A diagnostic `mean(final_state**2)` loss sends
gradients through memory and the CNN back to the earliest observed frames and
actions. No optimizer update occurs; these random vectors do not yet represent
learned motion or object persistence. The next-image training loss is still
reserved for the later training module.

All 73 tests, Ruff linting, and formatting checks pass. Tests include single-frame
contexts, causal indexing, influence of earlier observations/actions, gradient
flow through time, memory reset, batch independence, checkpoint reload, and
invalid inputs. See the [memory inspection summary](run-catalog.md#run-004).


## Steps 4–9: complete model, evaluation, and training

The complete model has **1,959,747 parameters**, all jointly trained from
random initialization. Each window starts a fresh observed-history memory.
The training configuration normalizes each 256-value CNN feature vector before
feeding it into memory; this adds no parameters and leaves the CNN unchanged.

| Module | Operation | Input -> output |
|---|---|---|
| CNN encoder | Four convolution/ELU stages and linear projection | `[B,8,3,64,64]` -> `[B,8,256]` |
| GRU memory | Learned initialization, then seven observed updates | Features and `[B,7,4]` past actions -> `[B,128]` |
| Dynamics | Linear(132,256), ELU, Linear(256,256), ELU, Linear(256,128), tanh | State and `[B,4]` next action -> predicted `[B,128]` |
| Decoder | Linear(128,2048), reshape to 128×4×4, four transposed convolutions | Predicted state -> `[B,3,64,64]` |

Decoder channels are 128 -> 128 -> 64 -> 32 -> 3, with kernel 4, stride 2,
padding 1. Hidden stages use ELU; the final stage uses sigmoid. Image sizes
are 4 -> 8 -> 16 -> 32 -> 64. There are no image skip connections.

```python
from src.model import WorldModel

model = WorldModel(config)
predicted_image = model(context_images, context_actions, next_action)
# context_actions: actions before the last observed image.
# next_action: action after the last observed image.

predicted_images = model.rollout(context_images, context_actions, future_actions)
# future_actions [B,H,4] -> predicted_images [B,H,3,64,64].
```

The CNN and GRU run once per rollout. Each later prediction advances only the
predicted latent state through the dynamics, then decodes it. Future images
are scoring targets; they are never passed to `rollout`. The predicted image
is never re-encoded. This is deterministic prediction, with no stochastic
latent sampling or additional latent-alignment loss.

### Training settings and entry point

`configs/train.json` specifies seed 3, batch size 16, eight context frames,
256 features, 128 latent coordinates, 10 epochs, and Adam with learning rate
0.0003. `normalize_features=true` prevents large CNN features from saturating
the GRU. Adam uses PyTorch defaults for its remaining settings. Gradients are
clipped to global norm 10. All computation uses float32, deterministic
algorithms, and the configured device. Training includes the final partial
batch, shuffles only the training windows, and uses zero DataLoader workers.
Prepared images/actions are preloaded after checksum validation to avoid
repeated NPZ decompression during shuffling. Positions remain excluded.

```powershell
uv sync --locked
uv run --locked python -m src.train
# Training and validation only, without final test scoring:
uv run --locked python -m src.train --skip-test
# Alternative settings:
uv run --locked python -m src.train --config configs/train.json
```

For each batch: clear gradients, infer the history state, predict the next
latent state and image, calculate mean squared RGB error, backpropagate through
all four components, clip gradients, and update with Adam. The loss covers one
future frame. Recursive predictions are evaluated but do not contribute to
this baseline's training loss.

Training MSE is averaged over examples as their updates happen. Validation MSE
uses fixed weights after the epoch, so these two curves do not measure the same
instant of training. Nonfinite losses/gradient norms stop the run and record
failure. Validation runs before training (epoch 0) and after every epoch.
The checkpoint with lowest **validation one-step MSE** is selected, including
random weights if training never improves that score. Test data is scored only
after this selection and does not determine the chosen checkpoint.

### Frozen evaluation windows and metrics

Every checkpoint uses all 1,122 validation one-step windows. Recursive
validation uses starts 0, 1, and 2 of each of the 51 validation recordings:
153 windows that all support horizons 1, 5, 10, and 20. Thus all recursive
horizons use the same histories. The recursive horizon-1 score can differ from
the all-window one-step score because their start sets differ.

Final test scoring uses all 5,632 one-step windows and 768 recursive starts
across the 256 official test recordings. For a context starting at frame 0,
the model observes frames 0–7 and actions 0–6, then receives actions 7–26 to
predict frames 8–27. Start 2 reaches target frame 29 with last action 28.
Action 29 is never used as a transition target.

Scores include whole-image MSE and changed-region MSE on normalized RGB.
The changed-region mask marks pixels whose maximum absolute channel change
between the last observed image and the actual target exceeds **0.05**. Error
is averaged over channels and changed pixels per nonempty window, then over
those windows. Empty masks are counted and omitted from that metric; an
entirely empty set is recorded as JSON null. Whole-image errors still measure
false changes outside the mask. The same masks/windows score the model and
copy-last-frame baseline. Aggregation weights each window equally, including
partial batches.

Standalone evaluation creates a new run and identifies each checkpoint by
absolute path, SHA256, epoch, training run, and dataset-manifest fingerprint:

```powershell
uv run --locked python -m src.evaluate --split validation --checkpoint evaluation/runs/run-007/checkpoints/best.pt
# Compare random and selected weights on the frozen test set:
uv run --locked python -m src.evaluate --split test --checkpoint evaluation/runs/run-007/checkpoints/epoch-000.pt evaluation/runs/run-007/checkpoints/best.pt
```

Replace the example run number with the desired training run. Evaluation uses
inference mode, preserves the model's previous train/eval mode, and performs
no optimization. Checkpoints from different configurations or a different
dataset-manifest hash are rejected for comparison.

### Recorded artifacts

Every training run preserves:

- `config.json`, `environment.json`, `dataset-manifest.json`, `status.json`,
  `summary.md`, and source fingerprints.
- `architecture.json` with parameter counts and the exact module structure.
- `evaluation-windows.json` with fixed recursive episode IDs/start indices.
- `metrics.jsonl`, `metrics.csv`, and `results.json` with epoch timing,
  validation scores, training scores, first-batch component gradient norms,
  selected epoch, and peak allocated CUDA memory.
- `validation-epoch-NNN.json` with complete one-step/recursive metrics.
- `checkpoints/epoch-NNN.pt` for random weights and every trained epoch,
  plus `best.pt` and `latest.pt` aliases. Checkpoints contain model weights,
  optimizer state, settings, epoch, provenance, and RNG/shuffle states.
  Automatic resume is not implemented; saved state is available for a future
  resume entry point.
- `predictions/epoch-NNN/`: the first three recursive windows, selected by
  fixed ordering. Each includes metadata and numerical NPZ arrays, actual and
  predicted GIFs, and panels showing actual, predicted, copy, and absolute
  error at horizons 1/5/10/20. Error panels use absolute normalized error
  without per-image contrast scaling.
- `plots/learning-curve.png` and `.svg`.

A separate final evaluation run preserves checkpoint identities, the same
split/window protocol, prediction artifacts, `results.json`, and recursive
error plots in PNG/SVG. Earlier runs are preserved. Generated runs, datasets,
and the local plan remain ignored by Git.

The dedicated tests now verify complete gradient flow to the earliest context
frame/action, action causality, one-step/rollout consistency, absence of image
feedback during rollout, scoring/mask aggregation, recording boundaries,
evaluation without weight updates, and exact checkpoint reloading.



### First experiment results - 2026-10-16

Completed training: [run-007](run-catalog.md#run-007).
Completed held-out comparison: [run-008](run-catalog.md#run-008).
All 10 epochs ran on the RTX 5060 Laptop GPU; epoch **9** was selected using
validation one-step MSE. Training plus validation took about 144 seconds;
the final test comparison took about 27 seconds. This is one seed on the
development subset, without hyperparameter selection on the test set.

| One-step setting | Validation MSE | Test MSE | Test changed-region MSE |
|---|---:|---:|---:|
| Random weights, epoch 0 | 0.093762 | 0.095703 | 0.066345 |
| Selected weights, epoch 9 | 0.030978 | 0.030591 | 0.056828 |
| Copy last frame | 0.019680 | 0.019174 | 0.081990 |

The model improves over random weights, but **does not beat copying the last
frame on whole-image one-step MSE**. Saved predictions remain blurry. Lower
changed-region error alone is insufficient evidence of good prediction: even
random gray outputs beat copying on that restricted mask here, while their
whole-image errors are much worse. Both metrics and the actual images matter.

Recursive test metrics use the same 768 starts at every horizon:

| Horizon | Random whole-image MSE | Selected whole-image MSE | Copy whole-image MSE | Selected changed MSE | Copy changed MSE |
|---|---:|---:|---:|---:|---:|
| 1 | 0.095931 | 0.030926 | 0.021251 | 0.058047 | 0.087563 |
| 5 | 0.095695 | 0.076839 | 0.041078 | 0.092769 | 0.115456 |
| 10 | 0.095777 | 0.085647 | 0.048593 | 0.100163 | 0.120669 |
| 20 | 0.095585 | 0.085965 | 0.055230 | 0.097442 | 0.124677 |

Whole-image recursive error increases sharply after the first step, and the
selected model trails copy-last-frame at all four horizons. This measures the
limits of the current one-step-only objective; it does not establish one
specific cause of rollout failure. Predicted/inferred latent mismatch remains
a plausible issue to study, alongside image blur and limited data.

Inspect the [learning curve](run-catalog.md#run-007),
[recursive comparison](run-catalog.md#run-008),
and [fixed-example GIF](run-catalog.md#run-008).
Metrics, checkpoint hashes, fixed window IDs, and numerical predictions are
saved alongside these visual artifacts. Peak allocated CUDA tensor memory
was about 241 MiB; this excludes CUDA context/driver memory.

#### Saturation diagnosis and correction

The first attempt, [run-005](run-catalog.md#run-005), exited with
code 1 after epoch 9 without a captured Python traceback. Its completed
checkpoints are preserved and its status is failed. The runtime termination
cause was not established. Its best validation MSE was 0.050074, and late
training measurements showed exactly zero CNN and GRU gradient norms.

The separate [run-006 probe](run-catalog.md#run-006)
reproduced zero history gradients on a fixed training batch. Raw CNN feature
magnitudes reached 529; 99.87% of recurrent-state coordinates were saturated
at absolute value at least 0.999. A per-frame layer-normalization intervention
on the same weights/batch reduced that fraction to 0.78% and restored CNN/GRU
gradients. This supports feature-scale saturation as the cause of the lost
gradients; it does not explain the separate process termination.

`normalize_features=true` now enables parameter-free per-frame feature
normalization in the complete model. It defaults to false for compatibility
with earlier module inspections/checkpoints, and is enabled explicitly in
`configs/train.json`. The corrected run started again from random weights.
The [run-009 health probe](run-catalog.md#run-009)
confirmed nonzero gradients in all four components of the selected checkpoint.
A regression test amplifies CNN outputs to reproduce the saturation pattern
and verifies that normalized history encoding retains encoder gradients.

To repeat a recorded checkpoint health check:

```powershell
uv run --locked python -m src.inspect_training --checkpoint evaluation/runs/run-007/checkpoints/best.pt
```

An unhealthy checkpoint exits with a failed assertion and a recorded diagnostic
run. No optimizer update occurs in this probe. All **88 tests**, Ruff linting,
and formatting checks pass. Steps 0-9 of the local implementation map are
complete; further model improvements are separate experiments.


## Controlled diagnosis experiments

`src/experiment.py` runs the follow-up experiment sequence. Settings live in
`configs/experiments.json`, separately from the original training baseline.
The architecture, feature normalization, optimizer, learning rate, and seed
remain fixed. Each stage gets its own run directory, linked by a parent suite
record. The sequence uses training and validation recordings only.

```powershell
# Complete sequence, including the trained zero-action ablation:
uv run --locked python -m src.experiment
# Individual stages:
uv run --locked python -m src.experiment --stage tiny
uv run --locked python -m src.experiment --stage history
uv run --locked python -m src.experiment --stage multistep
# Frozen action interventions on the configured baseline checkpoint:
uv run --locked python -m src.experiment --stage actions
```

The history/actions commands use `baseline_checkpoint` from the experiment
configuration. Update that path to inspect a different trained checkpoint.
The full sequence trains a matched one-step control, a three-step model, and
a zero-action control, then inspects their selected checkpoints.

### 1. Tiny-dataset fitting

A private NumPy generator selects eight training recordings using seed 3.
For each recording, rank its one-step windows by copy-last-frame MSE. Select
two random starts from the lower half and two from the upper half, giving
32 fixed examples with different amounts of image change. The halves are
relative to each recording, not global motion classes. Save all recording IDs,
start indices, change groups, and copy errors before training.

Train the unchanged world model from random weights with one-step MSE, batch
16, at most 2,000 optimizer updates. Score the same 32 windows every 50 updates.
Preserve checkpoints every 250 updates, plus initial, best, and latest states.
Stop early after at least 250 updates if aggregate MSE is at most 20% of the
fixed copy baseline and at least 90% of examples beat copying individually.
This is an operational fitting target, not a guarantee of perceptual quality.

Scores include per-window whole-image and changed-region errors, the fraction
beating copying, raw feature magnitudes, recurrent saturation, and sampled
component gradient norms. Intermediate visuals use a fixed example from each
recording. The best checkpoint saves all 32 examples as numerical arrays,
actual/predicted/copy/error panels, and GIFs. These scores measure memorization,
not generalization.

If the numerical target is not reached, the tiny stage also runs a one-window
fitting check (up to 1,000 updates) and a 32-image appearance reconstruction
check (up to 2,000 updates). Reconstruction removes temporal prediction:
CNN -> normalized features -> Linear/tanh latent projection -> decoder.
It targets the current image, with MSE target 0.001. Identity copying has zero
reconstruction error and is not a meaningful competitive baseline for this
diagnostic. These appearance-only checkpoints are explicitly distinguished
from world-model checkpoints.

### 2. Refreshed history versus recursive prediction

Freeze a checkpoint and score the same 153 validation starts at every horizon
from 1 to 20. Recursive prediction uses its initial eight real frames once,
then feeds predicted latent states into the dynamics. Refreshed prediction
uses a new sliding window of eight real images before each target prediction.
For target frame t+k, it receives real frames through t+k-1 and action a_(t+k-1).

Both modes use identical targets and the same changed-region mask, defined
relative to the original last observed image. Save endpoint error curves,
recursive-minus-refreshed MSE, per-window errors, recursive latent norms and
saturation, and fixed side-by-side videos. No gradients or optimizer updates
occur. Refreshed prediction has extra information; its advantage does not
establish a unique cause of recursive failure.

### 3. Matched one-step versus three-step training

Both models start from identical seeded parameters, verified using a SHA256
digest of parameter names and values. They use identical training windows,
shuffle seeds, batches, optimizer settings, and update budgets. Both use only
starts supporting three future targets: 20 per recording, or 4,100 windows.
Ten passes through these windows give 2,570 optimizer updates.

The control trains the first target only. The experimental model minimizes:

$$
L_3=\frac{1}{3}\sum_{k=1}^{3}
\operatorname{MSE}\big(D(\hat z_{t+k}),\tilde o_{t+k}\big),
\qquad \hat z_{t+k}=T(\hat z_{t+k-1},a_{t+k-1}).
$$

No intermediate predicted state is detached. Future observations remain loss
targets; neither real nor decoded future images enter the recursive training
path. Every pass scores all 1,122 one-step validation windows and the fixed
153 recursive starts at horizons 1/5/10/20. Each model's best checkpoint is
selected by validation one-step MSE. Additional frozen history diagnostics
score all 20 horizons on the selected checkpoints.

Equal update budgets do not equalize supervision or compute: the three-step
model receives three target frames per window and performs more transitions
and decoder operations. Timing and the exact objective are recorded. This
comparison is one seed, not a claim of a general improvement.

### 4. Action dependence and trained ablation

For each selected model, score correct actions, shuffled future actions,
shuffled past actions, both shuffled, and repetition of the first future
action. A seeded cycle pairs every validation recording with a different
recording. Donors have the same window start, and complete sequences stay
intact. Save every donor mapping; no shuffling occurs across data splits.

For each horizon, report prediction sensitivity to the intervention and
intervened target MSE minus correct-action target MSE. Positive differences
mean the correct actions were more accurate for those targets. Per-window
results and fixed videos accompany the aggregates. Physical plausibility is
not guaranteed for substituted action sequences, and sensitivity alone does
not establish useful action conditioning.

The full suite additionally trains a one-step zero-action control with the
same initial parameter values, training windows, and update budget as the
one-step action-conditioned control. `ignore_actions=true` replaces both
past and future actions with zeros during training and inference. The flag is
stored in configuration/checkpoints, so loading preserves the ablation.
Its frozen action diagnostic should show exactly zero sensitivity, providing
an internal check of the measurement. The default baseline keeps this flag
false.

Implementation files:

- `src/data/windows.py`: future targets, refreshed histories, and tiny selection.
- `src/training/experiments.py`: update-budget training, per-window fitting
  scores, checkpoints, and conditional fitting diagnostics.
- `src/evaluation/diagnostics.py`: observation-refresh and action interventions.
- `src/evaluation/experiment_plots.py`: experiment curves and comparison videos.
- `src/model/reconstruction.py`: appearance-only fallback diagnostic.
- `src/experiment.py`: configuration, stage sequencing, and linked run records.

Existing run directories are preserved. Parent and child records include
configuration, environment/source fingerprints, data provenance, measurements,
status, summaries, and relevant checkpoints/predictions. The local plan tracks
which experiments actually ran; an implemented experiment is not a measured
result until execution completes.

### Measured diagnosis results

The completed report is [recovery suite run-020](run-catalog.md#run-020),
which links all experiments from the original [suite run-010](run-catalog.md#run-010).
The original suite failed during final status recording after training; its
failure history is preserved. The recovery ran only the remaining frozen
diagnostic and reporting, without repeating training.

**Tiny fitting:** [run-011](run-catalog.md#run-011) reached its
numerical target at update 1,950: whole-image MSE **0.001376**, versus copy
**0.017638**, with **29/32** examples beating copying. Fixed inspected images
retain substantially more scene detail than the original broad-data model,
although they are not perfect. This demonstrates fitting on these examples,
not generalization. Single-window and reconstruction fallback experiments were
not triggered. The reconstruction path is covered by a dedicated execution test.

**Observation refresh:** [run-012](run-catalog.md#run-012) measured
the original selected checkpoint on matching validation targets. Horizon-20
MSE was **0.085870** recursively and **0.029994** with refreshed real histories.
The extra observations make a large difference; the gap alone does not isolate
uncertainty, representation mismatch, or accumulated model error.

**Matched training:** [run-013](run-catalog.md#run-013) and
[run-014](run-catalog.md#run-014) completed 2,570 updates each, with
identical initial parameter hashes and starting-window sets. Their best
validation one-step MSEs were **0.031672** and **0.030976**. Frozen recursive
diagnostics [run-015](run-catalog.md#run-015) and
[run-016](run-catalog.md#run-016) use the same 153 starts:

| Horizon | One-step-trained control | Three-step-trained model | Copy last frame |
|---|---:|---:|---:|
| 1 | 0.031916 | 0.030732 | 0.022116 |
| 5 | 0.068902 | 0.037814 | 0.042734 |
| 10 | 0.082178 | 0.046509 | 0.049771 |
| 20 | 0.077939 | 0.056005 | 0.058061 |

Three-step training improved recursive MSE in this experiment and beat copying
at horizons 5, 10, and 20 on these validation starts. One-step image quality
still trails copying, and inspected long-horizon outputs remain blurry. The
three-step model receives three times as many target frames per training
window and more computation per update; this does not isolate the effect from
additional supervision. There is one seed and no new held-out test claim.

**Actions:** [run-017](run-catalog.md#run-017) and
[run-018](run-catalog.md#run-018) separately shuffle past and future
action sequences. For the three-step model, substituting future actions
increased MSE by **0.002535** at horizon 1 and **0.003833** at horizon 20.
Past-action interventions had much smaller effects. This supports useful
future-action dependence within these frozen models, with the physical
plausibility limitation of substituted action sequences.

The trained zero-action control [run-019](run-catalog.md#run-019)
completed the same 2,570 updates and achieved one-step validation MSE
**0.031387**, comparable to the action-conditioned one-step control's
**0.031672**. Its recursive horizon-20 MSE was **0.089472**, versus **0.077939**
for that control. These single-seed comparisons do not establish a universal
advantage of action conditioning. The frozen [run-021 check](run-catalog.md#run-021)
confirmed exactly zero sensitivity and zero action-intervention advantage for
the zero-action model at every horizon.

Inspect [tiny-fit example](run-catalog.md#run-011),
[recursive/refreshed video](run-catalog.md#run-016),
and [matched learning curves](run-catalog.md#run-020).
All numerical predictions, per-window scores, donor mappings, checkpoint states,
source fingerprints, and split identities are retained in the linked runs.

#### Recording recovery

Windows denied replacing run-019's final status JSON with `PermissionError`
after its training updates, checkpoints, results, and CSV had been written.
The exact locking process was not identified. Atomic JSON replacement now
retries permission denial up to six attempts with bounded exponential delays,
then raises. Persistent failure preserves the previous complete JSON and
removes the temporary file. Tests exercise both cases at the recording seam.

The recovery command checks complete update counts, shared initial weights,
shared training-window identities, dataset fingerprints, and action-ablation
flags before running the remaining diagnostic. It creates a new parent record
and leaves the old failures visible:

```powershell
uv run --locked python -m src.experiment --recover-suite evaluation/runs/run-010 --ablation-run evaluation/runs/run-019
```

This recovery is for the completed comparison stages, not automatic resumption
of interrupted gradient updates. All **102 tests**, Ruff linting, and formatting
checks pass. The next research decision should use these measured results;
improved recursive MSE does not by itself resolve image blur.


## Expansion to 10,000 training recordings

`configs/scale.json` defines the larger-data experiment. Its target is
**10,000 complete training recordings**, excluding the original **51 validation**
and **256 test** recordings. The original prepared dataset and previous runs
remain available for comparison.

```powershell
# Complete pipeline: acquisition, integrity checks, packing, training, evaluation:
uv run --locked python -m src.scale
# Acquisition and verification only:
uv run --locked python -m src.prepare_expansion
# GPU integration check on the original small dataset:
uv run --locked python -m src.check_scaling
# Reuse a completed acquisition record, replacing the example ID as appropriate:
uv run --locked python -m src.scale --prepared-run evaluation/runs/run-NNN
```

### Acquisition and split preservation

The expansion reads checksummed headers from the same original BAIR archive
under its recorded ETag and archive size. It selects the first 40 training
members in archive order, reuses the original training member, and downloads
the 39 additional members. Each source retains its URL, exact byte range,
ETag, size, and local SHA256 receipt. The final member can contain unselected
records because the prepared training count stops at exactly 10,000.

Downloads use six workers and bounded retries. Each partial member has a
journal containing its committed byte count and prefix SHA256. On restart,
the committed prefix must pass verification; an uncommitted tail is truncated
before resuming from the correct archive offset. A partial file without valid
provenance is rejected. Completed cached members are checked before reuse.

The 205 original training IDs remain in training; 9,795 verified recordings
are added. Validation/test IDs, their ordering, episode-file checksums,
preprocessing, and action alignment remain unchanged. Selected new records
pass TFRecord length/payload CRC32C and strict array validation. Existing
prepared episodes are preserved across an interrupted conversion, with content
checks against their decoded source. Original held-out episode files are copied
without recompression, preserving byte identity. The expanded manifest records
the original validation split fraction separately from the new count ratio.

Expanded files live in `data/bair-10000/`. Its `raw/` contains new members;
the original two source members remain under `data/bair/raw/`. Prepared
episodes are under `data/bair-10000/prepared/episodes/`. The source is still
an archive-order subset, not the full BAIR training corpus or an OOD dataset.

### Disk-backed training

`src/data/packed.py` prepares read-only memory-mapped NumPy arrays under
`data/bair-10000/.cache/training/`:

- `images.npy`: uint8 main-camera images `[10000,30,64,64,3]`.
- `actions.npy`: float32 actions `[10000,30,4]`.
- `manifest.json`: file hashes, training ID order, and source-manifest fingerprint.

The images occupy about 3.69 GB in this cache. Training reads windows on demand;
it does not preload all recordings into Python memory. Two DataLoader workers
reopen their own read-only maps, with pinned batches for CUDA. Samples copy
their slices before normalization so callers cannot modify the cache.
Positions and the auxiliary camera remain in the prepared source episodes
for inspection but are not model inputs.

### Training and comparison

The model starts from random weights with the same architecture, seed 3,
feature normalization, Adam learning rate 0.0003, gradient clipping 10, batch
size 16, and three-step image MSE. Each recording supplies 20 eligible windows:
**200,000 windows, 12,500 optimizer updates, and 600,000 future-frame target
exposures in one complete pass**. Overlapping windows reuse recorded frames;
these exposures are not 600,000 distinct observations. No future observation enters the recursive
prediction path. The first larger-data run uses one epoch; more epochs can be
configured explicitly. The previous 205-record comparison used 2,570 updates,
so the expansion also increases optimization and target exposure.

Validation and checkpoints occur every 1,000 updates and at the final update.
Lightweight progress is saved every 100 updates. Initial, periodic, best,
and latest checkpoints include optimizer/RNG state and provenance. Best-model
selection continues to use validation one-step MSE; recursive horizons
1/5/10/20 and fixed visual examples are recorded throughout. Metrics include
examples and supervised frames processed, timing, and sampled component gradients.

Final validation and test runs compare random weights, the selected expanded
model, and the prior 205-record three-step checkpoint. Their training-manifest
hashes intentionally differ. Cross-dataset comparison therefore verifies each
checkpoint's saved manifest snapshot and requires exactly matching holdout
IDs/order/episode hashes, image preprocessing, action timing, and model input
conventions. It does not bypass data provenance by ignoring the manifest mismatch.
The original strict evaluator remains available for comparisons within one dataset.

Every acquisition, training, evaluation, and parent pipeline has its own run
record under `evaluation/runs/`. The parent links all children and preserves
the expanded manifest and packed-data receipts. Numerical predictions, images,
GIFs, plots, checkpoint identities, metrics, status, and failures remain local.

Implementation: `src/data/expansion.py`, `src/data/packed.py`,
`src/prepare_expansion.py`, `src/evaluation/scaling.py`, and `src/scale.py`.
`src/training/experiments.py` supports the worker configuration and records
sample/target counts. Atomic file replacement uses bounded retries for
transient Windows permission denial.

### Git exclusions

The local `.gitignore` excludes plan files/directories, downloaded/prepared
data, generated runs, Python bytecode, virtual environments, common tool and
notebook caches, and log files. A test invokes Git's actual ignore matcher in
an isolated temporary repository, checking both exclusions and that README,
AGENTS.md, source, configuration, and uv.lock remain eligible for version control.
The current workspace itself has no Git repository; no files were uploaded.

The GPU integration check completed in run-026/run-027: disk-backed windows,
two spawned workers, three-step backpropagation, validation, and checkpoint
saving processed 32 windows / 96 future targets in two optimizer updates.
This verifies the infrastructure on the original small dataset and is separate
from the expanded-data experiment.

### Completed larger-data results

The [end-to-end pipeline run-028](run-catalog.md#run-028) completed
acquisition verification, packing, training, validation, and test comparison.
The original acquisition is run-025; canonical verification is run-029,
training is run-030, validation is run-031, and test evaluation is run-032.
Earlier acquisition attempts and their errors/restarts remain recorded.

The final dataset contains **10,000 training, 51 validation, and 256 test
recordings**. The original manifest is preserved. Validation/test episode
IDs, order, file checksums, preprocessing, and action alignment match the
smaller experiment. Canonical metadata records both the original validation
split setting (0.2) and the expanded count ratio (51/10,051), and distinguishes
CRC verification of selected records from unused records in the last shard.

Training started from the same initial parameter values as the 205-record
three-step reference. It completed one full pass: **12,500 updates, 200,000
windows, and 600,000 future-frame target exposures**. The final checkpoint at
update 12,500 was selected by validation one-step MSE **0.025020**. Training
plus scheduled validation took about **481 seconds** on the RTX 5060 Laptop.
The best checkpoint is [run-030/checkpoints/best.pt](run-catalog.md#run-030).

Frozen test results, with lower MSE better:

| Metric | 205-record three-step reference | 10,000-record model | Copy last frame |
|---|---:|---:|---:|
| One-step, all 5,632 windows | 0.030325 | 0.023891 | 0.019174 |
| Recursive horizon 1, fixed 768 starts | 0.030640 | 0.023922 | 0.021251 |
| Recursive horizon 5 | 0.037360 | 0.029247 | 0.041078 |
| Recursive horizon 10 | 0.045715 | 0.036243 | 0.048593 |
| Recursive horizon 20 | 0.054927 | 0.046178 | 0.055230 |

The expanded setup reduced test one-step MSE by about **21%** and horizon-20
MSE by about **16%** relative to the previous three-step model. It beats
copying at recursive horizons 5/10/20, but still trails copying at one step.
Test changed-region one-step MSE is 0.041453 versus reference 0.055360 and
copy 0.081990. Inspected generated frames remain blurry: lower error does not
establish detailed or physically accurate video generation.

This is one seed with both more data and more optimizer updates than the old
run (12,500 versus 2,570). It does not isolate sample diversity from additional
optimization. Checkpoint selection used validation only; no gradient updates
or hyperparameter adjustments followed the final test comparison.

Inspect the [learning curve](run-catalog.md#run-030),
[test recursive errors](run-catalog.md#run-032),
and [fixed prediction video](run-catalog.md#run-032).
All 110 tests and Ruff checks pass. The end-to-end command reuses verified
downloads, prepared episodes, and packed arrays on subsequent runs; it starts
a new recorded training run from random weights.

## Fixed validation benchmark for blur experiments (B0)

```powershell
uv run --locked python -m src.benchmark_blur
# Compare a later checkpoint using the same frozen protocol:
uv run --locked python -m src.benchmark_blur --checkpoint PATH
```

`configs/blur-protocol.json` freezes the dataset/checkpoint identities, seed,
eight-frame context, all 1,122 validation one-step windows, 153 common recursive
starts, horizons 1/5/10/20, metrics, planned experiment budgets and six visual
examples. The command creates this file only when absent and rejects conflicting
contents. Both reference and candidate must match the dataset manifest. Candidate
latent dimensions may differ, but input context and scoring settings stay fixed.
The reference is run-030/best.pt. No training or test evaluation is performed.

Visual selection uses only real images before loading prediction weights:
three lowest-change windows, followed by three highest-change windows, all from
distinct validation recordings. Change is the mean squared difference between
the initial image and all 20 future targets; seed 3 breaks ties. This is an
image-change proxy affected by lighting and noise, not an object-motion label.
Selection is diagnostic; aggregate metrics include the complete eligible split.

In addition to existing pixel/changed-region MSE, boundary error is:

$$
E_{\mathrm{gradient}}(\hat o,o)
=\tfrac12\left(
\operatorname{mean}|\Delta_x\hat o-\Delta_x o|
+\operatorname{mean}|\Delta_y\hat o-\Delta_y o|
\right).
$$

Here the differences are signed forward differences between adjacent RGB pixels
in [0,1]. Average each direction over valid neighbors and channels, then average
directions equally and windows equally. No padding, smoothing or edge threshold
is used. Lower is better. This is target-referenced edge accuracy, not a
standalone sharpness score; constant brightness offsets are invisible to it,
so always inspect MSE and images too. Target frames are used only for selection
and scoring; recursive prediction receives initial context and future actions.

Each invocation records a new run: immutable protocol copy and SHA, checkpoint
SHA, manifest, environment/source hashes, aggregate and per-recursive-window
metrics, horizon curves, and six matched actual/predicted/copy/error PNG/GIF
panels. PT artifacts preserve normalized context, past/future actions, targets,
and predictions; NPZ artifacts preserve the displayed streams. Overlapping
windows are not independent observations. Planned training budgets in the
protocol are not executed by this command.

Reference execution **run-034** reproduced validation MSE **0.02501980** exactly.
One-step gradient MAE was **0.05793638** for the model and **0.03802815** for copy.
At recursive horizon 20, model MSE was **0.04781138** versus copy **0.05806142**,
but gradient MAE was **0.06086142** versus copy **0.05532964**. Thus lower
long-horizon pixel MSE does not establish better boundary accuracy. Inspected
low/high-change examples remain blurry; this benchmark does not identify the
cause. Next planned step: the separate appearance reconstruction diagnostic.


## Controlled blur experiments (B1-B5)

Run the authorized suite with:

```powershell
uv run --locked python -m src.blur_suite
```

Configuration is `configs/blur-experiments.json`. It uses the same 10,000 training
recordings and B0 validation protocol, batch 16, seed 3, float32, Adam at 0.0003,
gradient clipping at 10, and two read-only mmap loader workers. Each training
experiment has 12,500 optimizer updates and validation every 1,000 updates plus
the initial/final states. Source: `src/training/blur_experiments.py`; orchestration:
`src/blur_suite.py`. Checkpoints and previous runs are preserved.

- **B1 reconstruction:** train a separate random-initialized CNN -> normalized
  features -> Linear/tanh 128-value latent -> decoder. It reconstructs the last
  observed frame, without memory or dynamics. It uses the same decoder design
  and 200,000 eligible windows as the prediction experiments: current frames
  7 through 26 per recording, so neighboring images are correlated. Its 200,000
  target exposures are fewer than the 600,000 three-frame prediction exposures.
  Validation scores reconstruct current frames 7 through 28. Six fixed panels
  show real current/reconstructed current/real next/predicted next. This is a
  task diagnostic, not a competitive baseline; copying current has zero
  reconstruction error. A lower reconstruction error does not establish that
  all future-prediction blur comes from dynamics.
- **B2 continuation:** restore run-030's model, Adam and Python/NumPy/Torch/CUDA
  RNG and shuffle state. Train one additional epoch with unchanged three-step
  MSE. The legacy checkpoint is accepted only at a complete epoch boundary.
  Continuation uses an explicit deterministic permutation sampler, rather than
  relying on the old DataLoader's implicit cursor. New checkpoints persist the
  full order, consumed-sample cursor, epoch and generator state; prefetching
  cannot advance the saved consumed cursor. Exact new-format resume is tested
  against uninterrupted training, including Adam and RNG state. No claim of
  bitwise identity with an unexecuted continuation of the legacy loader is made.
- **B3 objective:** branch twice from B2's validation-selected checkpoint with
  identical model/Adam/RNG/cursor state and equal 12,500-update budgets. Compare
  three-step MSE with `L1 + 1.0 * gradient_MAE`, averaged over all three predicted
  frames. Gradient MAE is the signed-neighbor difference error defined in B0;
  it remains differentiable during training. Log MSE, L1 and gradient components.
  All checkpoints continue to be selected by validation one-step MSE; objectives
  have different scales, so their raw training-loss values are not comparable.
- **B4 capacity:** execute if at least one B3 branch still has higher one-step
  boundary error than copying. This is a predeclared boundary-error proxy, not
  a definitive blur detector. Train new 128- and 256-latent models from scratch
  with matched seed, data order, MSE objective and update budget. Random weights
  differ across incompatible shapes; initial weights cannot be identical.
  Record parameters, elapsed time and CUDA tensor memory. The valid comparison
  is the two fresh models, not a fresh wide model versus a resumed narrow model.
- **B5 selection:** choose the lowest validation one-step MSE among the original
  reference and all world-model candidates. Preserve the selection and checkpoint
  SHA before a single final test comparison of reference and selected models.
  The final comparison uses all 5,632 one-step windows, 768 common recursive
  starts, and the first six official test recordings at start zero for visuals.
  Test results never alter the recorded selection. No stochastic model or
  spatial/detail bypass is introduced in this suite.

Each stage receives its own recorded run; the parent links children and records
the capacity decision, validation candidates, final selection and test results.
Training stores initial/periodic/best/latest checkpoints, optimizer/RNG/sampler
states, metrics CSV/JSONL, component losses, progress, learning curves and source
provenance. Evaluations save matched images, tensors and boundary/pixel metrics.
The selected checkpoint is referenced by SHA in `configs/viewer.json`; subsequent
`python -m src.viewer` sessions use it by default. Existing viewer sessions keep
their already-loaded model. Pass `--checkpoint` to inspect another saved model.
New objectives and capacity experiments still require visual inspection: lower
MSE or sharper edges alone cannot establish correct physical predictions.

### Measured B1-B5 results — 2026-10-17

All six training experiments completed their 12,500-update budgets. Original
suite run-035 and evaluation run-039 failed after B2 training when JSON recording
encountered the new sampler tensor. Those records and completed training were
preserved. `metadata_summary` now records tensor shape/dtype/SHA while retaining
exact values in checkpoints; viewer and existing evaluation entry points also
use it. Recovery suite **run-040** reused completed children without retraining.
Recovery command: `python -m src.blur_suite --resume-from evaluation/runs/run-035`.
This reuses completed stages; an interrupted training stage requires an explicit
training resume from its latest checkpoint rather than being silently skipped.

| Experiment | Validation MSE | Boundary MAE | Recursive h20 MSE |
|---|---:|---:|---:|
| Original run-030 reference | 0.025020 | 0.057936 | 0.047811 |
| B2 extra epoch | 0.021353 | 0.056374 | 0.040388 |
| B3 continued MSE | **0.019315** | 0.055413 | 0.041038 |
| B3 L1 + gradient | 0.020882 | **0.053507** | 0.041083 |
| B4 fresh latent 128 | 0.025457 | 0.058209 | 0.042992 |
| B4 fresh latent 256 | 0.025289 | 0.058170 | 0.048136 |

B1 reconstruction reached MSE **0.009935** and gradient MAE **0.050867** on
1,122 validation current-frame windows. Fixed panels retain more detail than
future predictions but remain blurry. This supports further appearance and
optimization investigation, not attribution of all prediction error to dynamics.

B3 initial weights, Adam, RNG and sampler matched exactly, verified against
saved initial checkpoints in `run-040/loss-branch-initial-equivalence.json`.
Both selected states are their final 12,500-update states, so the comparison
also matches endpoint budgets. L1+gradient slightly improves boundary error but
worsens MSE. B4 permutation/shuffle equality is recorded in
`run-040/capacity-data-equivalence.json`. The 256-latent model has 2,568,515
parameters versus 1,959,747; its one-step gain is small and its h20 error is
worse. Neither change resolves blur for this single seed and budget.

**Selected:** `run-042/checkpoints/best.pt`, global update 37,000, using the
predeclared validation one-step MSE rule. B2's selected parent was global
update 24,500, followed by 12,500 control updates. No test results altered this
selection. Final test comparison **run-050** used identical 5,632 one-step
windows and 768 recursive starts:

| Test metric | Original model | Selected model | Copy baseline |
|---|---:|---:|---:|
| One-step MSE | 0.023891 | **0.018641** | 0.019174 |
| Changed-region one-step MSE | 0.041453 | **0.029638** | 0.081990 |
| One-step boundary MAE | 0.055105 | 0.052638 | **0.036477** |
| Recursive h20 MSE | 0.046178 | **0.040293** | 0.055230 |
| Recursive h20 boundary MAE | 0.058482 | 0.059246 | **0.054258** |

One-step MSE improves about **22%** and h20 MSE about **13%** relative to the
reference. The selected model slightly beats copying on aggregate one-step
MSE, while boundary error still trails copying and h20 boundary error worsens
slightly relative to the reference. Inspected images remain blurry. More
optimization helped pixel prediction; sharpness and physical fidelity are not
established. Single-seed results do not identify one definitive blur cause.

The parent report is `evaluation/runs/run-040/summary.md`; test artifacts are in
`run-050/reference/` and `run-050/selected/`. Default viewer configuration now
pins the selected checkpoint by SHA; a verified session is running at
`http://127.0.0.1:8766` (run-051). Existing sessions retain their original loaded
model. All **122 tests**, Ruff linting and formatting checks pass. The local
plan marks B0-B5 complete; spatial appearance preservation is a separate future
experiment to discuss, not an executed change.


## Appearance and motion experiments (C0-C5)

The new experiment family preserves observed appearance while predicting its
evolution. Baseline run-042 and all B0-B5 artifacts remain the reference history.

```powershell
uv run --locked python -m src.appearance_suite
# Reuse completed stages from an earlier invocation:
uv run --locked python -m src.appearance_suite --resume-from evaluation/runs/run-NNN
```

`--through c0`, `c1`, or `c3` limits an invocation to that stage; default is c5.
Settings are in `configs/appearance-experiments.json`; the immutable
`configs/appearance-protocol.json` copies B0 windows/examples and records the new
reference checkpoint, budgets, metric definitions and selection rules.

The original `src.train` entry point also honors `Config.architecture` through
the shared model factory, and its checkpoints reload the selected variant.
That entry point retains its documented one-step MSE training/selection;
use `src.appearance_suite` for the matched three-step C0-C5 protocol below.

### C0: selection measures appearance and movement together

Evaluate all 1,122 validation one-step windows and the 153 common recursive
starts at horizons 1/5/10/20. Keep MSE, changed-region MSE and signed-neighbor
boundary MAE. Add RGB SSIM using uniform 7x7 valid windows, population moments,
data range 1, C1=0.01^2 and C2=0.03^2. Average channels, locations and windows
equally; clamp small negative numerical variances to zero. This explicit variant
must not be compared directly with paper scores using other SSIM conventions.

Eligibility requires every one-step/recursive MSE and changed-region MSE to be
at most 1.02 times reference, mean recursive SSIM at least reference minus 0.01,
and one-step/h20 changed-region error better than copying. Rank eligible states
by mean boundary MAE across horizons, then higher mean SSIM, lower h20 changed
error and lower one-step MSE. Promotion requires at least 1% boundary improvement.
These tolerances were fixed before candidate training. Fixed-example inspection
and action diagnostics corroborate quality; sharp static copying is insufficient.

### C1-C3: observed appearance and original-source rendering

The temporal state initially stays a 128-value GRU vector. For each observed
history, the encoder exposes feature maps of shapes [B,32,32,32], [B,64,16,16],
[B,128,8,8] and [B,128,4,4]. Zero-initialized 1x1 adapters add last-observation
features at matching decoder scales; the candidate initially reproduces the
original model exactly. Both candidate and continued control warm-start shared
weights from run-042 and start fresh Adam. Match batch 16, seed/data order,
three-step MSE, lr=0.0003, clip=10, 12,500 updates and 1,000-update validation.
Save and compare fully trained endpoints as well as eligibility-selected states:
the selected checkpoint may remain the initial state when detail and movement
trade off. Both C2 branches warm-start the same C1 appearance endpoint, with
fresh Adam; an endpoint used for this mechanism study still must pass all C0
gates before final promotion.

Let A_t be the appearance pyramid computed before the prediction rollout:

$$
\hat z_{t+k}=T_\theta(\hat z_{t+k-1},a_{t+k-1}),\qquad
\hat o_{t+k}=D_\theta(\hat z_{t+k},A_t,o_t).
$$

The motion variant predicts pixel flow F_k [B,2,64,64], generated RGB G_k
[B,3,64,64], and three softmax masks [B,3,64,64]:

$$
\hat o_{t+k}=m_c\odot\tilde o_t+
m_w\odot\operatorname{Warp}(\tilde o_t,F_k)+m_g\odot G_k.
$$

Backward flow means target location (x,y) samples source (x+dx,y+dy).
Displacements are bounded by 64*tanh; masks start with logits (-2,-2,2),
favoring generation while flow starts at zero. Sampling is bilinear with border
extension and pixel-center semantics equivalent to align_corners=False. Each
horizon samples the original observed image, not a previously sampled output.
Generated content handles appearance unavailable in the source; borders and
occlusions can still fail. Flow is cumulative relative to the source at t.

The custom sampler treats raw observed RGB as constant and differentiates flow
through bilinear weights. It avoids nondeterministic CUDA source-pixel scatter
gradients; CUDA deterministic algorithms remain enabled. Flow-coordinate
gradients are zero on/outside source borders. Encoder/adapter gradients still
flow through appearance features and the temporal path. No future images or
future features enter inference, and decoded images are not CNN inputs.

Model variants and sampling live in `src/model/appearance.py`; legacy encoder
forward output remains compatible while `pyramid` exposes intermediate maps.
Evaluations preserve flow/masks/warped/generated tensors and component panels.
C3 compares correct, zero, repeated and substituted future actions on matching
validation targets, with seeded donor recordings. Record action sensitivity,
correct-action error advantage, mask mass and out-of-bounds flow. These are
diagnostics, not proof of correct counterfactual physics or labeled occlusion.

### Conditional C4 and final C5

If observed appearance helps but the h20 changed-error/copy ratio exceeds 0.35
or future-action advantage is nonpositive, test spatial memory. Its learned
state is [B,64,8,8], initialized from the first image's spatial features. A
ConvGRU consumes each subsequent observed feature grid and previous action;
a separate ConvGRU transition consumes predicted state plus broadcast next
action. Decode the predicted grid without flattening spatial positions. Compare
fresh vector/spatial models using the same renderer, objective and order.
Shared later decoder layers retain matched initial weights. Implementation:
`src/model/spatial.py`. Channels have no predefined physical interpretation.
The diagnostic renderer family is chosen by boundary/structural fidelity even
if its movement errors fail promotion gates, so motion failures can trigger the
spatial study. Final selection always applies the complete C0 eligibility rule.

Confirm a promising pair with seed 7 where applicable; warm-start repetitions
share pretrained weights and vary data order/new-layer initialization. Freeze
final selection before test evaluation. The official test split has already
been used in earlier project phases; do not call it newly untouched data.
Preserve all parent/child records, optimizer/RNG/consumed sampler cursors,
protocol/source/dataset/checkpoint hashes, metric histories and fixed examples.
Update `configs/viewer.json` only when a candidate qualifies for promotion.
Execution outcomes are recorded in the local plan and each suite summary.

### Measured C0-C5 results — 2026-10-17

The [completed suite run-059](run-catalog.md#run-059), recovering
completed C0/C1 stages from run-052, selects **C1 appearance run-056, update
5,000**. All eight controlled training branches finished 12,500 updates each:
100,000 experimental updates and 4.8 million future-frame target exposures.
These are separate branches with reused recordings, not 100,000 updates applied
to the selected model. Its shared weights warm-start from run-042 at update
37,000, with fresh Adam; selected weights incorporate 5,000 new local updates.

The selected checkpoint passes every frozen validation guard: one-step MSE
0.00815827, mean recursive boundary MAE 0.04730393 and mean SSIM 0.70150.
The second-ordering appearance repeat, run-077 at update 3,000, also passes
all guards and improves mean boundary MAE to 0.04901026 versus its paired
control's 0.05862586. [Audit run-078](run-catalog.md#run-078)
verifies identical initial parameter values and a different training-window
shuffle: zero adapters and the shared pretrained parent make this a data-order
replication, not independent training from random weights.

Selection was written before both final test runs. The official test split had
already been evaluated in earlier phases. [Reference run-080](run-catalog.md#run-080)
and [selected run-081](run-catalog.md#run-081) score all 5,632 one-step
windows and 768 common recursive starts:

| Test metric | Reference run-042 | Selected appearance model |
|---|---:|---:|
| One-step MSE | 0.018641 | **0.007607** |
| One-step changed-region MSE | 0.029638 | **0.026199** |
| Horizon-20 MSE | 0.040293 | **0.027806** |
| Horizon-20 changed-region MSE | 0.057824 | **0.055780** |
| Mean recursive boundary MAE | 0.056089 | **0.045472** |
| Mean recursive SSIM | 0.453448 | **0.711745** |

Whole-image MSE decreases about 59.2% at one step and 31.0% at horizon 20.
Changed-region gains are smaller: 11.6% and 3.5%, respectively. Inspected fixed
images preserve much more background/object appearance, while moving robot
contours still blur or ghost over long rollouts. These results do not establish
that blur or physical prediction is solved. [Selected-model action diagnostics
run-082](run-catalog.md#run-082) find positive correct-action error
advantages over zero/repeated/shuffled controls at applicable horizons, without
establishing counterfactual physics. This descriptive check did not change the
frozen selection.

The motion-rendering and spatial-state modules are implemented and evaluated,
but neither branch qualifies for promotion. Motion run-064's stored checkpoint
has mean boundary MAE 0.03695036 but fails h20 changed-region error. Its aggregate
move-mask weight is only 0.007856, with 93.48% out-of-bounds flow samples; the
composite largely suppresses the poor warp. Components and controls are in
[run-068](run-catalog.md#run-068) and [run-069](run-catalog.md#run-069).
The spatial endpoint lowers one-step MSE to 0.00604718 versus fresh vector
control 0.00858331, but still fails long-horizon guards. Spatial/vector parameter
counts are 1,098,920 / 2,008,680, and training plus validation takes 923 / 890
seconds. [Audit run-073](run-catalog.md#run-073) verifies their shared
initial renderer weights and shuffle. No general spatial-memory advantage or
optimization convergence is inferred from this budget.

Exact tensors, actions, components, images/GIFs, source/checkpoint/data hashes,
optimizer/RNG/sampler states and measured compute remain in the run records.
The suite's `visual-review.md` documents actual inspections. Each C-phase
training record includes a selection-label correction for a legacy metadata
label; actual checkpoint decisions and raw MSE histories remain intact. Future
runs record the correct label and confirm a winning control against its distinct
paired counterpart. All **141 tests**, lint and formatting checks pass.


## Detail and transport experiments (phase D)

Phase D tests the causes of remaining moving-object blur before scaling a new
recipe. Settings are in `configs/detail-experiments.json`; the immutable
`configs/detail-protocol.json` forks the earlier fixed examples and pins the
current run-056 checkpoint. Separate training, component diagnostics and frozen
evaluations create their own run records. Earlier checkpoints remain intact.

```powershell
uv run --locked python -m src.detail_suite
# Smaller diagnostic stages:
uv run --locked python -m src.detail_suite --through tiny
uv run --locked python -m src.detail_suite --through flow
# Reuse completed stages; replace the example parent ID as appropriate:
uv run --locked python -m src.detail_suite --resume-from evaluation/runs/run-NNN
```

### Capacity and transport diagnostics

`src/training/detail.py` trains configurable horizon prefixes and records
optimizer, RNG, consumed sampler cursor, initialization hashes, exposures,
metrics, checkpoints and images. Tiny fitting selects high-change windows from
32 distinct training recordings using targets only for selection. The one-window
fit passes at 600 updates (MSE 0.00006831, SSIM 0.99637). The 32-window fit plus
lower-LR extension reaches MSE 0.00051563, SSIM 0.97918 and edge-energy ratio
0.99683, but misses the strict combined pixel/gradient accuracy gate. Inspected
images are sharp; these are memorization diagnostics, not generalization scores
or proof that optimization has converged. Records: run-087 through run-091.

`src/training/transport.py` trains known translations without blending masks.
The source bank uses 64 BAIR **training** recordings: 48 for flow training and
16 held out from that component training. All were eligible for earlier BAIR
model training. Held-out translation offsets are fractional values absent from
the component training offsets. Synthetic action encoding is an explicitly
defined toy convention, not a real robot displacement calibration. Run-093
reaches 0.1541-pixel held-out endpoint error and warped-image MSE 0.000321.
Its checkpoints are tagged as diagnostic and cannot be loaded as trained world
models by the viewer. This result does not establish real object-motion accuracy.

### Bounded flow and conditioned appearance

`src/model/transport.py` keeps the vector GRU/history and latent transition.
For horizon k, renderer conditioning includes predicted z, the cumulative
**supplied action prefix** q_k and elapsed horizon. It never sees later actions
when producing an earlier frame:

$$
q_k=\sum_{j=0}^{k-1}a_{t+j},\qquad
r_k=[\hat z_{t+k},q_k,k/20].
$$

The flow predictor combines an action-conditioned global field with a local
8x8 field conditioned on z and normalized observed features, then upsamples
the field to [B,2,64,64]. For real-data experiments:

$$
F_k=b_k\frac{u_k}{1+|u_k|},\qquad b_k=\min(32,8+2(k-1)).
$$

The softsign tail preserves a polynomial coordinate gradient for large raw
outputs. The older synthetic diagnostic uses its recorded tanh/4-pixel profile;
checkpoint configuration preserves both conventions. Synthetic output layers
are damped to 0.1 of their learned values before real-data initialization.
Both compared branches receive the identical initialization.

The `transport` variant keeps source-position appearance features; `aligned`
computes appearance features from the **warped original real observation**:

$$
A_t^{(k)}=C_{\mathrm{pyramid}}(\operatorname{Warp}(o_t,F_k)).
$$

Both apply zero-initialized state/control-conditioned feature gates, initially
equal to one. The aligned variant re-encodes transformed observed RGB at each
horizon. No decoded/generated image or actual future observation becomes a CNN
input. Raw RGB sampling remains deterministic and differentiable with respect
to flow; the CNN receives gradients through the warped source. Alignment adds
compute relative to cached fixed skips, which is recorded rather than claimed
to be identical. Three softmax masks blend original source, warped source and
generated missing content. Initial logits (-2,2,-2) favor the transport path.

Flow learning also receives a mask-independent auxiliary pixel loss (weight
0.05), a pre-clamp out-of-bounds coordinate penalty (0.01), and a small spatial
variation penalty (0.0001). The coordinate penalty can correct out-of-bounds
samples even where border-clamped image sampling has zero coordinate gradient.
Generation remains available for occlusions and unavailable source appearance.

### Objective, curriculum and selection

The controlled objective comparison uses MSE, a half whole-image / half
changed-static-balanced MSE, and that balanced objective plus signed-neighbor
gradient error with weight 0.1. Empty changed/static groups fall back to the
available group. Change masks are computed from training targets for the loss;
they are never inference inputs or claimed object labels.

The next comparison tests fixed H3 supervision against the **1/3/5/10 curriculum
plus a persistent one-step MSE anchor**. It is a comparison of that training
package, not an isolated estimate of horizon length alone. Both branches use
identical H10-eligible windows and private shuffle seeds. Curricula have
different target exposures/compute; the records include those differences.

Evaluation keeps all 1,122 validation one-step windows and the 153 common
recursive starts at horizons 1/3/5/10/20. Each individual pixel/change error
must stay within 1.02 times run-056; mean SSIM must stay within -0.01, and
one-step/h20 changed error must beat copy. Eligible states rank by the equal
average of normalized mean changed error and boundary error. Promotion requires
at least 1% improvement in both unnormalized means. The frozen settings and
actual comparison designs are saved before their experiments.

An unqualified renderer is not carried into later objective/curriculum studies.
The full-data pair starts again from the same original run-056 shared weights,
with fresh Adam: original appearance/H3/MSE versus the screened recipe. Screening
weights are diagnostic; they do not give one full-data branch extra base-model
optimization. Identical recipes skip redundant duplicate branches. A promising
full-data result triggers seed-7 confirmation. Promotion requires the same
recipe to win that paired repeat and pass the improvement gates; the original
seed's checkpoint is retained rather than selecting the best repeat seed.
Selection freezes before final
test scoring; previous official-test reuse is disclosed. Stochastic modeling is
deferred when these diagnostics cannot isolate uncertainty from geometry,
optimization and generalization errors. Final action controls and the viewer
configuration use frozen selected weights.

### Phase D measured results

Parent `evaluation/runs/run-094` links the completed diagnostic and comparison
records. D0/D1 records are run-087 through run-091; D2 is run-092/run-093.
The following are **trained endpoints**, including rejected states:

| Comparison | Updates per branch | Endpoint one-step MSE | Outcome |
|---|---:|---:|---|
| Current appearance reference | 0 additional | 0.0081583 | Retained |
| Fixed transport / aligned transport | 1,500 | 0.0070137 / 0.0071512 | Rollout guards fail |
| MSE / balanced / balanced + boundaries | 1,500 | 0.0080914 / 0.0082788 / 0.0082071 | Selected states remain initial weights |
| Fixed H3 / curriculum + one-step anchor | 3,000 | 0.0081833 / 0.0076717 | Selected states remain initial weights |

The fixed transport endpoint improves mean recursive boundary MAE from
0.0467324 to 0.0373498, but fails horizon-10 pixel/change accuracy. The aligned
variant also fails movement guards. A replay on the same six saved validation
examples reduces out-of-bounds flow from 92.29% in the earlier motion model to
20.62%, and mean absolute flow from 48.13 to 7.28 pixels. These are descriptive
six-example statistics, not full-split physical-flow accuracy; the inspected
moving robot remains blurry or misplaced. Exact replay tensors/statistics are
in run-098. Initial equality audits are run-099 and the parent record.

Lower one-step MSE alone did not yield a qualified replacement. Selection
retains run-056 with the existing viewer pin. The unchanged screened recipe
skips duplicate 12,500-update full-data branches and seed repeats. Stochastic
modeling is deferred: neither sharp single-window memorization nor successful
toy translations isolates irreducible future uncertainty. The 32-window fit
still misses its accuracy gate. These experiments repair a flow failure and
establish sharp-output capacity, but **do not resolve general prediction blur**.

The curriculum endpoint is a useful lead despite rejection: horizon-20 MSE
falls from 0.0278699 to 0.0242023 (13.16%), changed-region MSE falls 14.03%, and
boundary MAE falls 6.11%. Its horizon-3 MSE rises 2.47% and changed-region MSE
rises 4.69%, exceeding the frozen 2% tolerance. Mean recursive changed-region
error improves 8.29% and boundary error 5.12%. Those tradeoffs are saved in
`run-094/endpoint-diagnostics.json`; the endpoint remains available as
`run-109/checkpoints/update-003000.pt`. It was not selected or tested as a
replacement. A follow-up could retain explicit H3 supervision while extending
the horizon, with its own frozen protocol.

The suite completed with selection frozen before test run-111. Since the
selected model is the reference, one test evaluation serves both identities:
one-step MSE 0.0076068431 and horizon-20 MSE 0.0278058123. Run-112 records
positive correct-action advantage over zeroed/shuffled actions; these
perturbations do not establish causal physics. Previous official-test reuse
still applies. All **154 tests**, Ruff lint and formatting checks pass.
`run-094/verification.json` records the selection timing and viewer SHA audit.
The verified retained-model viewer is at `http://127.0.0.1:8767/`.


## Conditional diffusion pilot (phase E)

Phase E tests one-step conditional generation against run-056 and a matching
deterministic U-Net. It is a bounded pilot; multistep training and automatic
viewer promotion are outside this experiment. The existing world-model weights
stay frozen. The new head starts random and receives observed multiscale
features, predicted latent state, next action, source RGB, last observed frame
difference, and the coarse predicted next image. No future image enters this
conditioning path. The deterministic control receives exactly the same context.

Let b be the frozen coarse next prediction, y the recorded target, and
r_0=2(y-b). For diffusion index n and independent Gaussian noise epsilon:

$$
r_n=\sqrt{\bar\alpha_n}r_0+\sqrt{1-\bar\alpha_n}\epsilon,\qquad
v_n=\sqrt{\bar\alpha_n}\epsilon-\sqrt{1-\bar\alpha_n}r_0.
$$

Train the head to predict v_n with squared error. This v denotes a diffusion
parameterization, not physical velocity. Training uses the actual target to
construct noisy residuals and supervision. At inference, start from Gaussian
noise and apply 50 eta=0 DDIM steps from a 100-step cosine schedule. Recover
clean residuals using r_0=sqrt(alpha_bar)*r_n-sqrt(1-alpha_bar)*v_hat. Clip
estimated residuals to [-2,2], then output clip(b+r_0/2,0,1). Every individual
sample is an RGB prediction; samples are never averaged for display.

The formulation follows [DDPM](https://arxiv.org/abs/2006.11239),
[DDIM](https://arxiv.org/abs/2010.02502), and the v parameterization discussed
in [Progressive Distillation](https://arxiv.org/abs/2202.00512). This small
residual-conditioned experiment is not a replication of those papers or MCVD.

```powershell
uv run --locked python -m src.diffusion_suite --through probe
uv run --locked python -m src.diffusion_suite
# Reuse completed stages from a matching parent:
uv run --locked python -m src.diffusion_suite --resume-from evaluation/runs/run-NNN
```

Settings are frozen in configs/diffusion-experiments.json and the generated
immutable diffusion-protocol.json. Each full branch trains 6,000 updates at
batch 16, Adam 0.0002, using the same private data permutation and identical
initial head parameters. EMA uses decay min(0.999,(1+update)/(10+update)).
The deterministic control directly predicts the residual with RGB MSE;
diffusion learns v. This compares objectives and inference procedures while
holding conditioning and head architecture fixed. It does not isolate all
effects of generative modeling or test joint retraining of dynamics.
Hidden layers use random initialization; the final convolution is zero in
both heads, so the initial deterministic prediction equals run-056 exactly.

Checkpoint selection uses 51 fixed validation starts, one per recording,
every 1,000 updates. Final comparison scores all 1,122 validation windows.
Three fixed per-window noise seeds score diffusion samples individually.
The larger validation comparison includes the selection windows; it is not
an independent final test. Windows from the same recording overlap.
Report MSE, changed-region MSE, signed-gradient MAE, SSIM, edge-energy ratio,
LPIPS and sample diversity. Eligibility requires >=5% LPIPS and >=1% boundary
improvement, with MSE/changed-region MSE within +5% of run-056. These are pilot
criteria; the original viewer remains pinned. No official test data is used.

LPIPS uses the official frozen SqueezeNet version-0.1 metric on RGB [-1,1]
without resizing. These external pretrained weights are evaluation tools;
they do not train the prediction head. Run records pin their state SHA and
package versions. Downloaded weights are cached under ignored data/.cache.
Checkpoint files include head, EMA, optimizer, RNG, consumed sampler cursor,
backbone/data identities, and settings. Load with training.diffusion.load_head;
generic world-model loading rejects head-only checkpoints.
Final reports also record the fraction of RGB channels at 0 or 1 as a
saturation proxy; it is not an exact count of pre-clamp values outside range.

### Phase E results

Parent **run-116** completed the pilot. The deterministic and diffusion heads
(run-118/run-121) each trained 6,000 updates, observing 96,000 windows spanning
all 10,000 training recordings. Each head has 1,934,819 trainable parameters;
the backbone stays frozen. Initial head tensors, complete data permutations,
backbone identity and fresh Adam match exactly. Both selected EMA states are
their update-6,000 states. No official test data was used.

| Model | One-step MSE | Changed-region MSE | Boundary MAE | LPIPS | SSIM |
|---|---:|---:|---:|---:|---:|
| Original run-056 | 0.0081583 | 0.0260891 | 0.041479 | 0.122070 | 0.829122 |
| Deterministic residual U-Net | 0.0053881 | 0.0189077 | 0.028624 | 0.056710 | 0.904001 |
| Diffusion, all three samples | 0.0101739 | 0.0367802 | 0.036284 | 0.046433 | 0.848224 |

The deterministic control passes the pilot criteria: MSE improves 33.96%,
changed-region error 27.53%, boundary error 30.99% and LPIPS 53.54%. Diffusion
has the lowest LPIPS (61.96% better than the original) and visibly sharper
robot boundaries/markings in the inspected fixed examples, but MSE rises
24.71% and changed-region error 40.98%. Sampled object placement and details
vary. Diffusion therefore fails the frozen accuracy guards. More visual detail
does not establish correct motion; the deterministic head is the stronger
accuracy/appearance compromise in this pilot.

Final evaluations are run-119, run-122 and run-124. Paired-noise zero-action
controls (run-120/run-123) show positive correct-action advantage for both
heads; perturbed actions may be implausible and do not establish causality.
Exact tensors and six matched comparison panels are in
`evaluation/runs/run-116/comparisons/`, with source, actual target, original,
control and every diffusion sample. Curves and verification are in the parent.
The diffusion evaluation took 175.24 seconds for all windows/three samples,
versus 1.96 seconds for the deterministic head, including metrics/artifact
work; these are evaluation totals, not isolated single-image latency.

An algebraic diagnostic decomposes diffusion sample MSE 0.0101739 into
within-sample variance 0.0024158 and empirical sample-mean error 0.0077581.
No sample-mean image is displayed or selected. This does not prove calibrated
uncertainty or justify choosing the best sample with target knowledge.

All **163 tests**, lint and formatting checks pass, including exact stochastic
training continuation, EMA reload, target-free conditioning and CUDA gradients.
The current viewer still uses run-056. These one-step validation results do
not authorize automatic replacement of the multistep model; temporal rollout
quality, confirmation seeds and final test evaluation remain future work.


## Interactive prediction viewer

To browse the new saved diffusion/U-Net comparisons:

```powershell
uv run --locked python -m src.result_viewer --run evaluation/runs/run-116 --port 8768
```

Open `http://127.0.0.1:8768/`. This comparison viewer shows the six fixed
validation examples, actual next image, original prediction, deterministic
U-Net and all three diffusion samples, recorded action and observed history.
It verifies saved-tensor hashes and matching contexts. Every session records
source identities and viewed examples. It performs no inference or training.

From this directory, run:

```powershell
uv run --locked python -m src.viewer
```

This opens a local browser GUI at `http://127.0.0.1:8765`. It uses the selected
checkpoint pinned in `configs/viewer.json` (currently `run-056/checkpoints/best.pt`)
and frozen test recordings. Without that config, it falls back to run-030.
Choose a recording, use Previous/Next or the arrow keys, or request a random
example. It displays the current image, four raw recorded action values, actual
next image, generated prediction, and the complete eight-frame observed context.
Prediction and copy-last-frame MSE both compare with the actual next image,
using float pixels in [0,1] before PNG rounding.

Each example uses eight real context frames `[1,8,3,64,64]`, seven past actions
`[1,7,4]`, and the next action `[1,4]`. At current frame t, the action is a_t and
the target is frame t+1. The target never enters inference. Stepping refreshes
the observed history; it does not continue a recursive latent rollout. Valid
current frames are 7 through 28, with targets 8 through 29.

The viewer uses frozen weights, CPU inference by default, verified episode
checksums, and loopback-only serving. There are no new dependencies or external
web assets. Backend/inference code lives in `src/viewer.py`; the GUI is
`src/viewer.html`. Viewer inference never updates checkpoint weights.

Optional arguments: `--split validation`, `--device cuda:0`, `--port 8766`,
`--no-browser`, `--checkpoint PATH`, and `--dataset PATH`. The dataset manifest
must match the supplied checkpoint's training manifest; for older checkpoints,
pass their original prepared dataset. Press Ctrl+C in the launch terminal to
finish the session. Closing the browser tab alone leaves the server running.

Every session creates an ignored `evaluation/runs/run-NNN/`, recording checkpoint
SHA, configuration, dataset manifest, environment/source hashes, page SHA and
snapshot, viewed-window metrics, and exact context/action/target/prediction
tensors. These user-selected examples are exploratory, not new aggregate test
results. Session status remains running until the server stops normally.
