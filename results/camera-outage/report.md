# Camera-outage pilot results

**Checkpoint selection:** 12 of 12 runs retained update 0 (initial weights). Selected test results therefore do not establish successful outage training. Unselected latest-state validation and training memorization are separate descriptive diagnostics.

All predictions scored before receiving their target observations.

| Split / method | H1 MSE | H10 MSE | H40 MSE |
|---|---:|---:|---:|
| validation / copy | 0.001105 | 0.010309 | 0.020442 |
| validation / image_feedback | 0.000980 | 0.011644 | 0.024938 |
| validation / vector_outage (3-seed mean) | 0.001105 | 0.010374 | 0.021453 |
| validation / spatial_outage (3-seed mean) | 0.001105 | 0.010379 | 0.021270 |
| validation / spatial_observed (3-seed mean) | 0.001105 | 0.010379 | 0.021270 |
| validation / spatial_detail (3-seed mean) | 0.001105 | 0.010379 | 0.021270 |
| test / copy | 0.001238 | 0.006844 | 0.016974 |
| test / image_feedback | 0.001503 | 0.010106 | 0.023071 |
| test / vector_outage (3-seed mean) | 0.001238 | 0.006911 | 0.018043 |
| test / spatial_outage (3-seed mean) | 0.001238 | 0.006901 | 0.017733 |
| test / spatial_observed (3-seed mean) | 0.001238 | 0.006901 | 0.017733 |
| test / spatial_detail (3-seed mean) | 0.001238 | 0.006901 | 0.017733 |

Seed-summary.json reports seed variation; child results retain recording bootstrap intervals.
New test recordings are disjoint; scene/session independence is unestablished.
Matched windows/actions/horizons per seed; parameter counts and compute differ.
Training uses 600 updates per branch; convergence is not established.
No operator benefit or calibrated uncertainty is established.
