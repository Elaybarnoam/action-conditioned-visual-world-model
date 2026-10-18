# Asset provenance and licenses

The project code is MIT licensed, copyright 2026 Elay Barnoam. This release
does not confer ownership of external datasets, papers, or pretrained metrics.
Research documentation and newly generated plots may be reused under the MIT
license; source imagery retains its upstream terms below. Released model
parameters are provided under MIT, with the training-data provenance recorded.

## BAIR Robot Pushing

The inputs, targets, contact sheets, demo image strings and saved image tensors
derive from the BAIR Robot Pushing dataset, original v0, image_main camera.
Credit: Frederik Ebert, Chelsea Finn, Alex X. Lee, Sergey Levine,
*Self-Supervised Visual Planning with Temporal Skip Connections*, 2017.

- [Paper](https://arxiv.org/abs/1710.05268)
- [Dataset homepage](https://sites.google.com/view/sna-visual-mpc/)
- [BAIR dataset licensing statement](https://sites.google.com/berkeley.edu/robotic-interaction-datasets/home)
- [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

BAIR's robotic interaction dataset page states that its data is CC BY 4.0.
Changes in this repository: frame/window selection, normalization, image
enlargement, layout/caption additions, prediction generation and error scoring.
Raw dataset shards are downloaded from the original host; they are not in Git.
The frozen split is an archive-order subset, not the full BAIR corpus.

## LPIPS and torchvision

LPIPS 0.1.4 is an external evaluation dependency. The official SqueezeNet
version-0.1 LPIPS metric and ImageNet-pretrained SqueezeNet are downloaded by
their libraries. They are frozen evaluation tools, not learned world-model
parameters. Their weights are not redistributed in our model bundles.

- [LPIPS source/license, BSD-2-Clause](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE)
- [LPIPS paper](https://arxiv.org/abs/1801.03924)
- [torchvision source/license, BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE)

Evaluation preprocessing and the complete loaded metric-state SHA-256 are
recorded in results/protocols/phase-e.json. The relevant libraries retain their
own notices. Other runtime packages are referenced in pyproject.toml/uv.lock;
their source or binary distributions are installed through package registries.

## DROID imagery and derived camera-outage artifacts

DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset, Khazatsky et al.
(2024), is released under Creative Commons Attribution 4.0 International.

- [DROID project and authors](https://droid-dataset.github.io/)
- [Original paper and licensing statement](https://www.jiajunwu.com/papers/droid_rss.pdf)
- [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/)

Our artifacts select recordings, decode the publisher's face-blurred 320 by 180
images, select temporal windows, generate predictions and calculate errors.
Viewer panels, GIFs and replay fixtures include attributed derived imagery.
Raw recordings remain on the original host and in ignored local data storage.
Code licensing does not supersede the data license. DROID evaluation uses frozen
AlexNet LPIPS and torchvision weights; those external weights are not bundled.

## Algorithms and authorship

DDPM, DDIM, v parameterization, CNNs, GRUs and U-Nets are established methods.
This project is an empirical implementation and comparison, not a claim of
inventing those methods. See docs/related-work.md for citations. Coding and
documentation were developed with Codex assistance under the maintainer's
direction; no academic affiliation, degree, peer review or benchmark record
is asserted by this repository.
