# Selective Pose Repair: Development Results and Interpretation

**Status: exploratory development evidence only. Not official test-set performance.**

This note records the results from the 30-sequence, action-stratified Penn Action official-training development subset. It is intended to prevent overclaiming and to guide the next experiment. Four of the 30 sequences do not provide valid torso scales for the primary metric, so the paired comparisons contain 26 scoreable paired sequences. The threshold summary's `n_sequences=30` is the number of sequence IDs represented; it should not be interpreted as 30 valid torso-scored sequences for every condition.

## Experimental setup

The sequence-cross-fitted selector predicts normalized localization errors for candidate outputs from inference-time pose features. Each held-out sequence is predicted by a model fitted on other selected development sequences. Candidate methods are raw output, linear interpolation, missing-only interpolation, confidence-aware repair, One Euro filtering, and Savitzky–Golay smoothing. The primary measure is torso-normalized PCK at threshold 0.20; missing predictions count as errors. Reported intervals resample whole sequences, not individual frame/joint observations.

The independent left/right mapping audit was inconsistent for frontal-view sequences. Mapping-sensitive estimates below therefore remain provisional.

## Paired comparison: selective repair minus each baseline

Positive PCK deltas favor selective repair. Negative mean-error deltas favor selective repair. PCK deltas below are converted to percentage points; confidence limits are from the supplied sequence-bootstrap CSV.

| Condition | Baseline | PCK delta (pp) | 95% CI (pp) | Mean normalized error delta | 95% CI |
|---|---|---:|---:|---:|---:|
| Clean | Raw | -0.003 | [-0.460, 0.373] | -0.00387 | [-0.00846, -0.00089] |
| Clean | Linear interpolation | -0.127 | [-0.569, 0.244] | -0.00227 | [-0.00405, -0.00077] |
| Clean | Confidence-aware | +0.520 | [0.058, 1.030] | -0.00190 | [-0.00353, -0.00059] |
| Blur | Raw | +0.260 | [-0.446, 1.224] | -0.00719 | [-0.01483, -0.00178] |
| Blur | Linear interpolation | -0.840 | [-2.430, 0.246] | +0.00653 | [-0.00141, 0.02175] |
| Blur | Confidence-aware | -0.184 | [-1.714, 0.833] | +0.00633 | [-0.00132, 0.02080] |
| Low resolution | Raw | +0.729 | [-0.100, 1.859] | -0.01601 | [-0.03216, -0.00444] |
| Low resolution | Linear interpolation | +0.016 | [-0.375, 0.324] | -0.00196 | [-0.00346, -0.00076] |
| Low resolution | Confidence-aware | -0.235 | [-0.577, 0.115] | +0.00006 | [-0.00094, 0.00118] |
| Missing observations | Raw | +13.829 | [11.452, 15.851] | -0.15077 | [-0.16933, -0.12764] |
| Missing observations | Linear interpolation | +0.005 | [-0.233, 0.245] | -0.00154 | [-0.00290, -0.00056] |
| Missing observations | Missing-only interpolation | -0.026 | [-0.281, 0.221] | -0.00183 | [-0.00377, -0.00038] |
| Missing observations | Confidence-aware | +1.076 | [0.293, 2.040] | -0.00385 | [-0.00647, -0.00173] |
| Occlusion | Raw | +2.098 | [0.714, 3.809] | -0.05929 | [-0.09824, -0.02607] |
| Occlusion | Linear interpolation | -0.919 | [-2.285, 0.033] | -0.00242 | [-0.00866, 0.00482] |
| Occlusion | Confidence-aware | -0.764 | [-2.142, 0.163] | +0.00089 | [-0.00580, 0.00828] |

These are exploratory pairwise intervals with multiple comparisons and no multiplicity correction. They should not be interpreted as confirmatory significance tests.

### Interpretation

1. **Missing observations:** selective repair strongly improves over raw coordinates, but its PCK is effectively indistinguishable from linear interpolation and missing-only interpolation. It has a small improvement in mean normalized error over these fixed baselines; that difference needs independent replication and should not be exaggerated.
2. **Occlusion:** selective repair improves over raw coordinates, but no reliable PCK advantage over the fixed interpolation or confidence-aware baselines is established.
3. **Blur and low resolution:** PCK differences versus raw are uncertain. Relative to linear interpolation, PCK is essentially tied under low resolution and worse by point estimate under blur. Some mean-error comparisons favor selective repair, but the pattern is not consistent across conditions.
4. **Clean video:** PCK is effectively unchanged versus raw. Small improvements over particular baselines are not sufficient to establish a practically important general advantage.
5. **One Euro:** the selector has markedly higher localization PCK, but the One Euro baseline itself is weak on this localization metric in this experiment. It should not be the main evidence for the method.

## Threshold sweep: descriptive risk/coverage trade-off

The supplied threshold sweep reuses the same out-of-fold predictions. It is useful for inspecting behavior but **must not be used to claim an unbiased optimal margin**.

At the default margin of 0.01, mean repair fractions are approximately 22.8% (clean), 26.8% (blur), 37.3% (low resolution), 40.7% (missing observations), and 50.7% (occlusion). Raising the margin generally reduces repair fraction and false-correction rate, often with only small changes in PCK. For instance:

- **Missing observations:** margin 0.01 gives PCK 0.7241, repair fraction 0.4066, and false-correction rate 0.00967. Margin 0.20 gives PCK 0.7247, repair fraction 0.2101, and false-correction rate 0.00092.
- **Occlusion:** margin 0.01 gives PCK 0.3303, repair fraction 0.5069, and false-correction rate 0.0367. Margin 0.10 gives PCK 0.3330, repair fraction 0.1939, and false-correction rate 0.00294.
- **Low resolution:** very large margins reduce repair and false corrections but also reduce coverage; at margin 0.20, mean coverage is about 0.963 versus about 0.999 at margin 0.01. Abstention can therefore leave missing/raw-invalid outputs unrepaired.

These are descriptive observations, not validated operating points. Selecting a threshold requires nested sequence-level validation (or a separate validation partition) followed by a single locked evaluation.

## Current scientific conclusion

The current results **do not support the claim that learned selective repair generally outperforms conventional fixed repair methods**. The strongest defensible finding is that temporal repair can recover many deliberately omitted pose observations relative to leaving raw predictions unrepaired; simple interpolation achieves comparable PCK in that condition.

A narrower research question remains plausible: can an error-aware selector reduce unnecessary corrections or improve a combined localization/risk-coverage objective without sacrificing performance against strong fixed baselines? To make that defensible:

1. resolve the Penn Action left/right landmark mapping, including front-view cases, using an independent documented audit;
2. define the primary outcome and threshold-selection rule before the next analysis;
3. evaluate the selector against the strongest fixed baselines with nested sequence-level validation;
4. report PCK, mean normalized error, coverage, repair fraction, and false-correction rate together;
5. only then lock the protocol and evaluate the official test split once.

No values in this note should be described as official test-set performance or as evidence of state-of-the-art results.
