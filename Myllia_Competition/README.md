# Myllia Cell Competition

I spent a month participating in this competition. There was a brutal drop from my public score to my private score. I dropped from 5th down to 9th. If I were to do something differently I would drop the h5ad augmentation. It improved public score off of a sheer coincidence, and I took a risk and hoped that the private perts would have similar data to the public perts. That risk did not pay off, as they were vastly different. I was doomed to lose when I made the decision to keep the augmentation and make improvements based off of the private score. I should have focused more on generalization instead of chasing public LB score. In the MALLORN Astronomy competition, I went from ~160th to 29th due to generalizing very well. I got caught up in trying to find major improvements to having other competitors around my score on the leaderboard getting improvements. I was optimizing my model for the public leaderboard instead of generalizing to the private leaderboard. There were many different approaches and models I used. I did not take organized notes during this competition, so I will only document my final model.

## Submission Results
Private perts were realeased with a week remaining so the majority of the submissions do not contain a private score.

| Submission Name | Public Score | Private Score |
|---|---:|---:|
| ZETRO4.csv | 4.51113 | 3.01241 |
| ZETRO3.csv | 4.46724 | 3.35202 |
| ZETRO2.csv | 4.43912 | 2.80835 |
| ZETRO2.csv | 4.61487 | 2.96500 |
| proper_refit_submission.csv | 4.35613 | 3.66766 |
| Knn_run_dirmagloss.csv | 3.76852 | 3.22656 |
| Knn_run.csv | 4.28731 | 3.55318 |
| basis_hypernet_submission.csv | 4.33438 | NA |
| z_test.csv | 4.23419 | NA |
| Zero_Augmentation2.csv | 4.18593 | NA |
| Zero_Augmentation.csv | 4.15891 | NA |
| Zero_CTRL.csv | 4.31855 | NA |
| w1_5_opt.csv | 4.19878 | NA |
| w2_opt.csv | 4.09875 | NA |
| w2.csv | 4.18224 | NA |
| optimized_dorothea_aligned.csv | 4.14695 | NA |
| h5adblendrisk.csv | 2.75345 | NA |
| optimize_dorothea.csv | 4.11779 | NA |
| Submission_proper_norm.csv | 3.88230 | NA |
| Submission_Dorothea.csv | 3.90602 | NA |
| submission_zctrl_0.9.csv | 3.74685 | NA |
| submission_ztrl_1.csv | 3.77714 | NA |
| submission_with_all_external_data.csv | 3.80899 | NA |
| submission_hybrid_string_e30_a0.778.csv | 3.21146 | 1.91973 |
| submission_hybrid_all_links_B_rank1_eg2_b21_b30.5_knnK8_t0.07_e15_a0.660.csv | 2.77870 | 1.37864 |
| submission_hybrid_string_e30_a0.778.csv | 3.21146 | 1.91973 |
| submission_bilinear_refit_oofalpha.csv | 3.43548 | NA |
| submission_bilinear_refit_oofalpha.csv | 3.22415 | NA |
| bilinear_h5ad_goat_sub.csv | 1.61559 | NA |
| submission_hybrid_mh_residual.csv | 0.12295 | NA |
| submission_bilinear_ensemble.csv | 2.33401 | NA |
| submission_bilinear.csv | 2.61072 | NA |
| submission_bilinear_v2.csv | -0.30954 | NA |
| myllia_direction_bilinear_factorization_1.csv | -0.10055 | NA |
| sub_internal_hypernet.csv | -0.68462 | NA |
| model4_submission.csv | 0.02630 | NA |
| model_full_external_ctrl_genept_sig_tfidf.csv | -0.94074 | NA |
| baseline.csv | 0.05213 | NA |

Squaring the gate improved performance, but was prone to overfitting.

Heavily debated between w, w**1.5 and w**2. From the tiny bit of information I gathered due
    to the limited amount of submissions, w**2 overfitted and caused LB performance to drop and
    w**1.5 had a higher CV score and LB score than the other two.

