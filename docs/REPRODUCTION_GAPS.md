# Reproduction gaps: paper versus released code

Found by reading upstream `93f3eec` against arXiv:2504.14305v3. Each gap is marked **blocking** (the paper number can't be produced without new work) or **deviation** (the result can be produced, but not under exactly the paper's conditions).

| ID | Gap | Type | Plan |
|---|---|---|---|
| G1 | No evaluation script for Tables 4–6. The `play_curriculum*.py` scripts only visualize and export. | blocking | Write `scripts/eval_almi.py` from the §6 metric definitions and Table 3 difficulty levels. |
| G2 | The evaluation set is CMU MoCap (1122 clips) retargeted to H1-2, and it isn't released. | blocking | Retarget CMU clips from AMASS with the PHC pipeline (needs an AMASS licence), or evaluate on held-out ALMI-X motions and report that deviation. |
| G3 | Only `all_wave.pkl` (88 motions, with a matching `mean_episode_length.csv`) is released. The paper's curriculum sorts the full retargeted AMASS set. `mean_episode_length_all_motion_new.csv` (8000 rows) refers to an `all_motion.pkl` that isn't shipped. | deviation | Phase 1 trains on `all_wave.pkl`. Later, `select_motions.zip` from ALMI-X may provide a larger motion set; check its format against the env loader. |
| G4 | Baselines (ALMI-whole, Exbody, ExBody2, OmniH2O), G1 configs and the Appendix E adversary variants aren't released. | blocking for those rows | Out of scope for now. |
| G5 | No metric scripts for the foundation-model Tables 12–13. Upper-body "success" (waving the correct hand) is judged by eye. | blocking | Log base velocity and survival automatically, and record upper-body success manually from recorded videos. |
| G6 | Only rounds lower-1, upper-1 and lower-2 are documented. The paper evaluates lower-3 + upper-2. | deviation (small) | Rounds 2–3 reuse `h1_2_upper` and `h1_2_lower` with the policy paths advanced. |
| G7 | `max_iterations = 100000` in every config. The paper says about 10⁴ steps to converge and about 17 h in total. | deviation | Stop on reward plateau. Record the chosen iteration in `RUN_LOG.md`. |
| G8 | The paper trained on unspecified GPUs. We have one Quadro RTX 6000 (24 GB, Turing). | deviation | Keep 4096 envs if they fit in memory; otherwise note the reduction. |
| G9 | `Data_Collection/check_motions.py` and `mujoco/configs/h1_2_21dof.yaml` hard-code `/home/bcj/...` paths. | bug | Fix only if we regenerate ALMI-X. We use the released dataset instead. |
| G10 | Deploy code for the real robot isn't released. | n/a | Not reproducible here (no H1-2). |
