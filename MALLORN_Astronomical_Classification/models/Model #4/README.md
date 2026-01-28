## Results

Optuna trial result (XGB tuning):
- Trial 204: value **0.61347**
- Params:
  - n_estimators: 4770
  - learning_rate: 0.009408
  - max_depth: 5
  - min_child_weight: 38
  - subsample: 0.9580
  - colsample_bytree: 0.5860
  - gamma: 8.6793
  - reg_alpha: 17.5374
  - reg_lambda: 24.2249
  - max_delta_step: 2
  - grow_policy: depthwise

OOF blend calibration:
- OOF best alpha: **0.20**
- OOF best threshold: **0.187286**
- OOF blended best F1: **0.57988**

Public leaderboard submissions:
- **0.6309**, **0.6024**, **0.6009**
This was enough for ~**130th / 925** on the public leaderboard.

---
