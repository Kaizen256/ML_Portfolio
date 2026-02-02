# Model 5: Teacher Features + Feature Selection + Optuna (OOF F1) + Multiseed XGBoost

This notebook contains Model 5 for the MALLORN challenge. Highest performing model out of all of them. I was able to place 29th out of 2389 entrants and 893 teams with limited time (2 weeks instead of the entire 4 months).

Model 5 builds on the teacher-stacking concept from Model 4, but pushes further in three directions:
1) Richer time-series and cross-band physics-inspired features
2) Feature selection using XGBoost gain importance
3) Direct optimization for OOF F1 using split-aware Optuna, followed by multiseed training/inference

## Results

Best parameters:
- n_estimators: 9476
- learning_rate: 0.0024306289953670325
- max_depth: 7
- min_child_weight: 6
- subsample: 0.5344962939912224
- colsample_bytree: 0.464696420753079
- colsample_bylevel: 0.8146569410634974
- colsample_bynode: 0.7285475291884695
- max_bin: 181
- gamma: 8.476938947246458
- reg_alpha: 0.44957196104419117
- reg_lambda: 5.23806334613521
- max_delta_step: 0
- grow_policy: depthwise

OOF multiseed best threshold: 0.419  
OOF multiseed best F1: 0.6243705941591138  
OOF AP (aucpr-ish): 0.5164263302782434  

| Submission | Public LB F1 | Private LB F1 |
|-------------|--------------|----------------|
| 1 | 0.6222 | 0.6231 |
| 2 | 0.6358 | 0.6413 |
| 3 | 0.5830 | 0.5962 |
| 4 | 0.6304 | 0.6491 |
| 5 | 0.5840 | 0.6100 |


Key upgrades vs Model 4:
- Expanded feature set (seasonality, structure function, SED slope fits, Bazin fits + rise/fall asymmetry)
- SpecType teacher grouping expanded (includes SLSN, SNII)
- Missing-value indicators added (`*_isnan`)
- Gain-based Top-K feature selection before Optuna
- Optuna objective returns OOF F1 (not AP/logloss)
- Final prediction is multiseed averaged XGB + global OOF-optimized threshold

# Features

Differences vs Model 4:
 - Adds raw (un-corrected) flux stats alongside de-extincted stats, plus "delta" features (deextinct minus raw).
 - Adds observation seasonality features globally and per band (counts of seasons, gap fractions, season spans).
 - Adds structure function features per band at multiple time lags (captures variability vs timescale).
 - Always fits Bazin per band (A, t0, trise/tfall, B, chi2red) and adds rest-frame versions of rise/fall.
 - Adds rise-time metrics (to 20%/50%) and asymmetry ratios (fall/rise) per band.
 - Adds cross-band ratios (amp_pre, AUC, width50, asym50) and band-to-band correlations (g-r, r-i, i-z).
 - Adds wavelength-trend features (peak time vs lambda, peak flux vs lambda) and SED slope fits at r-peak and +20d.
 - SpecType teacher: richer label mapping (splits out SLSN and SNII), uses missing-value flags, exposes spec_topprob.

## New Features

| Feature | Meaning | Why it helps |
|---------|----------|--------------|
| `flux_mean_raw` | Mean flux before dust de-extinction correction | Lets the model compare raw vs corrected brightness scale and learn dust-impact patterns |
| `flux_std_raw` | Standard deviation of raw flux | Captures variability before correction to detect dust-driven distortions |
| `snr_max_raw` | Maximum SNR using raw flux and raw error | Measures best raw detection strength as a correction sanity check |
| `fvar_raw` | Fractional variability using raw flux and error | Provides intrinsic-variability proxy before dust adjustment |
| `flux_mean_deext_minus_raw` | Difference between corrected and raw mean flux | Direct signal of how strongly extinction correction shifts brightness |
| `snrmax_deext_minus_raw` | Difference between corrected and raw max SNR | Measures how much detectability improves after correction |
| `n_seasons_global` | Number of observing seasons inferred from large time gaps | Separates single-season vs multi-season coverage patterns |
| `gap_frac_gt90` | Fraction of time gaps greater than 90 days | Flags strongly seasonal sampling |
| `gap_frac_gt30` | Fraction of time gaps greater than 30 days | Captures moderate sampling fragmentation |
| `n_seasons_{b}` | Number of observing seasons in band b | Band-specific sampling structure can differ by class |
| `season_maxspan_{b}` | Longest continuous season span in band b | Measures longest uninterrupted coverage window |
| `season_meanspan_{b}` | Mean season span in band b | Captures typical continuous coverage length |
| `sf_medabs_5_{b}` | Median absolute flux difference at ~5-day lag | Measures short-timescale variability strength |
| `sf_n_5_{b}` | Number of pairs used for 5-day lag SF | Reliability indicator for short-lag estimate |
| `sf_medabs_10_{b}` | Median absolute flux difference at ~10-day lag | Captures slightly longer-timescale changes |
| `sf_n_10_{b}` | Pair count for 10-day lag SF | Reliability indicator |
| `sf_medabs_20_{b}` | Median absolute flux difference at ~20-day lag | Mid-scale variability measure |
| `sf_n_20_{b}` | Pair count for 20-day lag SF | Reliability indicator |
| `sf_medabs_50_{b}` | Median absolute flux difference at ~50-day lag | Long-timescale variability proxy |
| `sf_n_50_{b}` | Pair count for 50-day lag SF | Reliability indicator |
| `sf_medabs_100_{b}` | Median absolute flux difference at ~100-day lag | Very long-timescale variability proxy |
| `sf_n_100_{b}` | Pair count for 100-day lag SF | Reliability indicator |
| `bazin_A_{b}` | Bazin model amplitude parameter | Smooth transient strength estimate |
| `bazin_t0_{b}_obs` | Bazin peak-time parameter (observed frame) | Parametric peak timing estimate |
| `bazin_trise_{b}_obs` | Bazin rise timescale (observed frame) | Encodes rise speed |
| `bazin_tfall_{b}_obs` | Bazin decay timescale (observed frame) | Encodes decay speed |
| `bazin_B_{b}` | Bazin baseline parameter | Estimates underlying baseline level |
| `bazin_chi2red_{b}_obs` | Reduced chi-square of Bazin fit | Fit quality indicator |
| `bazin_trise_{b}_rest` | Bazin rise timescale (rest frame) | Intrinsic rise speed |
| `bazin_tfall_{b}_rest` | Bazin fall timescale (rest frame) | Intrinsic decay speed |
| `t_rise50_{b}_obs` | Time from baseline to 50% amplitude (observed) | Measures rise speed |
| `t_rise20_{b}_obs` | Time from baseline to 20% amplitude (observed) | Early-rise behavior |
| `t_rise50_{b}_rest` | Rise time to 50% amplitude (rest frame) | Intrinsic rise speed |
| `t_rise20_{b}_rest` | Rise time to 20% amplitude (rest frame) | Intrinsic early-rise behavior |
| `asym50_{b}_obs` | Fall50 / Rise50 ratio (observed) | Captures peak asymmetry |
| `asym50_{b}_rest` | Fall50 / Rise50 ratio (rest frame) | Intrinsic asymmetry measure |
| `amppreratio_{a}{b}` | Ratio of pre-baseline amplitudes between bands | Color-dependent peak strength comparison |
| `aucratio_{a}{b}_obs` | Ratio of positive AUC between bands | Relative emitted-energy proxy |
| `width50ratio_{a}{b}_obs` | Ratio of 50% widths between bands | Cross-band duration contrast |
| `asym50ratio_{a}{b}_obs` | Ratio of asymmetry metrics between bands | Cross-band shape contrast |
| `corr_gr_obs` | Correlation between g and r band lightcurves | Measures multi-band coherence |
| `corr_ri_obs` | Correlation between r and i bands | Same, redder wavelengths |
| `corr_iz_obs` | Correlation between i and z bands | Same, further red |
| `tpeak_vs_lambda_slope_obs` | Slope of peak-time vs wavelength fit | Detects chromatic timing trends |
| `tpeak_vs_lambda_intercept_obs` | Intercept of that regression | Baseline timing offset |
| `tpeak_vs_lambda_r2_obs` | R² of peak-time vs wavelength fit | Reliability of chromatic timing trend |
| `peakflux_vs_lambda_slope` | Slope of peak-flux vs wavelength fit | Spectral energy trend |
| `peakflux_vs_lambda_intercept` | Intercept of flux–wavelength fit | Baseline spectral level |
| `peakflux_vs_lambda_r2` | R² of flux–wavelength fit | Reliability of spectral slope |
| `sed_logflux_loglambda_slope_rpeak` | Slope of log(flux) vs log(wavelength) at r-peak | Spectral slope at peak |
| `sed_logflux_loglambda_r2_rpeak` | R² of SED fit at r-peak | Fit reliability |
| `sed_logflux_loglambda_nbands_rpeak` | Number of bands used in SED fit | Coverage reliability |
| `sed_slope_rpeak_p20` | SED slope at r-peak + 20 days | Spectral evolution rate |
| `sed_r2_rpeak_p20` | R² of SED fit at +20 days | Reliability indicator |
| `sed_nbands_rpeak_p20` | Bands used at +20 days | Coverage indicator |
| `spec_topprob` | Maximum teacher-model class probability | Teacher confidence summary for meta-learning |

## Features from other models

| Feature | Meaning | Why it helps |
|---------|----------|--------------|
| `n_obs` | Total number of observations across all filters | Coverage proxy; some classes are observed more densely |
| `total_time_obs` | Total observed duration (max time − min time) | Separates long-timescale variability from short transients |
| `total_time_rest` | Duration corrected by (1+z) time dilation | Makes durations comparable across redshift |
| `flux_mean` | Mean dust-corrected flux | Overall brightness level |
| `flux_median` | Median dust-corrected flux | Robust brightness estimate |
| `flux_std` | Standard deviation of corrected flux | Overall variability strength |
| `flux_min` | Minimum corrected flux | Captures deep dips / noise floor |
| `flux_max` | Maximum corrected flux | Captures peak brightness |
| `flux_mad` | Median absolute deviation | Robust variability measure |
| `flux_iqr` | Interquartile range | Robust spread measure |
| `flux_skew` | Skewness of flux distribution | Detects asymmetric burst-like shapes |
| `flux_kurt_excess` | Excess kurtosis | Detects heavy-tailed spike behavior |
| `flux_p5` | 5th percentile flux | Robust low level |
| `flux_p25` | 25th percentile flux | Lower quartile |
| `flux_p75` | 75th percentile flux | Upper quartile |
| `flux_p95` | 95th percentile flux | Robust high level |
| `robust_amp_global` | p95 − p5 | Stable global amplitude proxy |
| `neg_flux_frac` | Fraction of flux values below zero | Noise-dominated vs real detection signal |
| `snr_median` | Median signal-to-noise ratio | Typical detection quality |
| `snr_max` | Maximum signal-to-noise ratio | Strongest detection strength |
| `median_dt` | Median time gap between observations | Sampling cadence proxy |
| `max_gap` | Largest time gap | Detects large seasonal breaks |
| `eta_von_neumann` | Von Neumann eta statistic | Smoothness vs randomness indicator |
| `chi2_const_global` | Chi-square vs constant model | Detects variability vs flat signal |
| `stetsonJ_global_obs` | Stetson J index (observed frame) | Robust correlated variability measure |
| `stetsonJ_global_rest` | Stetson J index (rest frame) | Intrinsic variability measure |
| `max_slope_global_obs` | Maximum absolute slope (observed) | Fastest brightness change |
| `max_slope_global_rest` | Maximum slope (rest frame) | Intrinsic fastest change |
| `med_abs_slope_global_obs` | Median absolute slope (observed) | Typical change rate |
| `med_abs_slope_global_rest` | Median absolute slope (rest) | Intrinsic change rate |
| `slope_global_obs` | Linear trend slope (observed) | Long-term drift indicator |
| `slope_global_rest` | Linear trend slope (rest) | Intrinsic drift |
| `fvar_global` | Fractional variability | Noise-corrected variability strength |
| `Z` | Redshift | Distance and time-dilation proxy |
| `log1pZ` | log(1+Z) | Stabilized redshift scale |
| `Z_err` | Redshift uncertainty | Reliability of distance estimate |
| `log1pZerr` | log(1+Z_err) | Stabilized uncertainty scale |
| `EBV` | Dust extinction value | Measures dust impact |
| `n_filters_present` | Number of filters with data | Multi-band coverage indicator |
| `total_obs` | Total observations across bands | Coverage strength |
| `n_{b}` | Number of observations in band b | Band completeness differs by class |
| `p5_{b}` | 5th percentile flux in band b | Robust low level |
| `p25_{b}` | 25th percentile | Lower quartile |
| `p75_{b}` | 75th percentile | Upper quartile |
| `p95_{b}` | 95th percentile | Robust high level |
| `robust_amp_{b}` | p95 − p5 in band b | Stable band amplitude |
| `mad_{b}` | Median absolute deviation | Robust variability |
| `iqr_{b}` | Interquartile range | Robust spread |
| `mad_over_std_{b}` | MAD / std ratio | Outlier sensitivity indicator |
| `eta_{b}` | Von Neumann eta | Smoothness vs noise |
| `chi2_const_{b}` | Chi-square vs constant | Variability detector |
| `stetsonJ_{b}_obs` | Stetson J (observed) | Correlated variability |
| `stetsonJ_{b}_rest` | Stetson J (rest) | Intrinsic correlated variability |
| `fvar_{b}` | Fractional variability | Normalized variability strength |
| `snrmax_{b}` | Maximum SNR | Best detection strength |
| `baseline_pre_{b}` | Estimated pre-peak baseline | Reference level for amplitude |
| `amp_{b}` | Peak − median flux | Simple amplitude |
| `amp_pre_{b}` | Peak − pre-peak baseline | Cleaner transient amplitude |
| `tpeak_{b}_obs` | Peak time (observed frame) | Band timing behavior |
| `tpeak_{b}_rest` | Peak time (rest frame) | Intrinsic timing |
| `peak_dominance_{b}` | Peak / baseline noise scale | Peak significance |
| `std_ratio_prepost_{b}` | Pre/post peak std ratio | Stability vs post-peak chaos |
| `width50_{b}_obs` | Width above 50% amplitude (obs) | Duration at mid level |
| `width80_{b}_obs` | Width above 80% amplitude (obs) | Peak sharpness |
| `width50_{b}_rest` | Width50 (rest) | Intrinsic duration |
| `width80_{b}_rest` | Width80 (rest) | Intrinsic peak shape |
| `t_fall50_{b}_obs` | Fall time to 50% (obs) | Decay speed |
| `t_fall20_{b}_obs` | Fall time to 20% (obs) | Late decay |
| `t_fall50_{b}_rest` | Fall50 (rest) | Intrinsic decay |
| `t_fall20_{b}_rest` | Fall20 (rest) | Intrinsic late decay |
| `sharp50_{b}_obs` | Amplitude / width50 (obs) | Spike sharpness |
| `sharp50_{b}_rest` | Amplitude / width50 (rest) | Intrinsic sharpness |
| `postpeak_monotone_frac_{b}` | Fraction monotonic after peak | Smooth decay vs noisy |
| `n_peaks_{b}` | Significant peak count | Multi-peak vs single transient |
| `n_rebrighten_{b}` | Rebrightening count | Secondary bump behavior |
| `decay_pl_slope_{b}_obs` | Power-law decay slope (obs) | Decay steepness |
| `decay_pl_r2_{b}_obs` | Fit R² (obs) | Fit reliability |
| `decay_pl_npts_{b}_obs` | Points used (obs) | Support size |
| `decay_pl_slope_{b}_rest` | Power-law slope (rest) | Intrinsic decay |
| `decay_pl_r2_{b}_rest` | Fit R² (rest) | Reliability |
| `decay_pl_npts_{b}_rest` | Points used (rest) | Support size |
| `tpeak_std_obs` | Std of peak times across bands (obs) | Peak alignment indicator |
| `tpeak_std_rest` | Std of peak times (rest) | Intrinsic alignment |
| `tpeakdiff_{a}{b}_obs` | Peak time difference (obs) | Chromatic lag signal |
| `tpeakdiff_{a}{b}_rest` | Peak time difference (rest) | Intrinsic lag |
| `peakratio_{a}{b}` | Peak flux ratio | Peak color proxy |
| `color_gr_at_rpeak_obs` | g−r color at r-peak | Spectral color at peak |
| `color_ri_at_rpeak_obs` | r−i color at r-peak | Red color proxy |
| `color_gr_rpeak_p20_obs` | g−r at +20d | Color evolution |
| `color_ri_rpeak_p20_obs` | r−i at +20d | Color evolution |
| `color_gr_rpeak_p40_obs` | g−r at +40d | Slower evolution |
| `color_ri_rpeak_p40_obs` | r−i at +40d | Slower evolution |
| `color_gr_slope20_obs` | g−r slope over 20d | Early color change rate |
| `color_ri_slope20_obs` | r−i slope over 20d | Early color change |
| `color_gr_slope40_obs` | g−r slope over 40d | Longer color trend |
| `color_ri_slope40_obs` | r−i slope over 40d | Longer trend |
| `p_spec_{c}` | Teacher probability for class c | Soft-label prior signal |
| `spec_entropy` | Entropy of teacher probs | Teacher uncertainty |
| `spec_topprob` | Max teacher probability | Teacher confidence summary |


## Training Setup

### Train / Validation Split
- Split-aware cross-validation using `StratifiedGroupKFold`
- `groups = split` (keeps each provided split together across folds)
- stratified by target to preserve class balance per fold

This matters because the dataset is imbalanced and naive random CV can leak information and inflate validation.

### Class Imbalance
The dataset is imbalanced, so the model uses:
- `scale_pos_weight = (#neg / #pos)` computed per fold during training

### Photo-z Augmentation
Training feature rows are augmented by simulating photo-z noise:
- sample `sigma` from the distribution of test `Z_err`
- create new training rows with `z_sim = z0 + Normal(0, sigma)`
- mark augmented rows with `photoz_aug = 1`

### Missingness Handling
Instead of filling missing values immediately, the pipeline adds explicit missing indicators:
- for each feature column `f`, also add `f_isnan` (0/1)

This helps tree models learn patterns like:
- missing bands
- failed curve fits
- missing cross-band comparisons


## Pipeline Summary

1. **Build feature table**
   - per-object: global + per-band + cross-band features
   - apply de-extinction correction using EBV
   - optionally add photo-z augmentation rows

2. **Add SpecType teacher features**
   - train LGBM multiclass on `SpecTypeGroup` (train only)
   - append OOF probabilities to train and full-fit probabilities to test
   - add `spec_entropy` and `spec_topprob`

3. **Feature selection**
   - train a baseline XGB across folds
   - rank features by aggregated `gain`
   - keep Top-K features (FS_TOPK)

4. **Optuna hyperparameter tuning (OOF F1)**
   - split-aware CV
   - build OOF probabilities
   - choose global best threshold on OOF
   - return OOF F1 as the trial objective

5. **Final multiseed model**
   - train multiple XGB seeds and average probabilities
   - pick final threshold from OOF
   - produce submission CSV
