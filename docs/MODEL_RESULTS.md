# Model results

Three-class motor imagery (**rest / imagined left fist / imagined right fist**) on PhysioNet
EEGMMIDB runs 4, 8 and 12, 106 usable subjects (see [`DATA.md`](DATA.md)). Every number on
this page comes from a file in `ml/results/` produced by the commands below; the tables at
the bottom are generated from those files, not typed in.

**Research / portfolio model. Not a medical device.**

## Summary

| | |
|---|---|
| Served model | **EEGNet, 21 motor channels** (`models/eegnet-motor-1.0.1`) |
| Cross-subject balanced accuracy (unseen people) | **45.7%** (95% CI 43.7% to 47.7%), chance = 33.3% |
| Subjects above chance | 97 / 106 |
| vs. CSP+LDA baseline, same channels (paired) | **+2.4 pt** (95% CI +1.1 to +3.8) |
| Compute per window, filter + ONNX inference (0.5 s pad) | **0.90 ms p50, 1.52 ms p99** (1 CPU thread, offline benchmark) |
| Window job size / Redis traffic per session (0.5 s pad) | 102.4 KB, 204.8 KB/s as sent today (64 ch); 33.6 KB, 67.2 KB/s if only the 21 model channels were sent |
| End-to-end latency through the running system | see [`BENCHMARKS.md`](BENCHMARKS.md) |

The cross-subject numbers pool the 5 fold models, each scored on the people it never saw:
that is the estimate of how the method does on a new person. The exported file is the
fold-0 model, which scored 43.7% on its own 22 held-out subjects (fold-to-fold range
43.3% to 50.0%). Retraining one model on all 106 subjects for serving is a possible next
step, but it would have no held-out score of its own, so it was not done here.

## Setup

- **Windows:** 2 s of EEG (320 samples at 160 Hz), a new one every 0.5 s, plus 1 s of
  extra signal before it so the causal filter has settled (the "pad"). A window is labelled
  only if its whole 2 s core carries one label ("pure" windows).
- **Preprocessing:** causal 4th-order Butterworth band-pass 8 to 30 Hz (mu and beta bands),
  steady-state initial conditions, then per-window normalization. It is the same code
  (`neurostream_common.preprocessing`) that the inference workers will run, so training and
  serving cannot drift apart.
- **Cross-subject:** 5 folds over subjects. Each fold trains on 76 to 77 people,
  early-stops on 8 others and tests on 21 to 22 nobody saw in training. Every subject is tested exactly once,
  so pooled numbers cover all 106.
- **Within-subject:** each person separately; test on run 12, the chronologically last run,
  as in a real session where you calibrate first and then use the system.
- **Models:** CSP+LDA (classic spatial filtering + linear classifier, one-vs-rest) and
  EEGNet (a compact CNN: 2,051 parameters on 21 channels, 2,739 on 64), with class-weighted loss, AdamW,
  batch 64, up to 30 epochs, early stopping after 6 epochs without validation improvement.

### Why balanced accuracy, not accuracy

About half of all windows are `rest`, so always predicting rest scores **49.5% accuracy**
(the "majority baseline" column) while being useless. Balanced accuracy is the mean of the
per-class recalls: that same always-rest model scores 33.3%. Plain accuracy is reported, but
every comparison and the headline number use balanced accuracy.

### Why confidence intervals over subjects, and paired differences

Windows from one person are strongly correlated, so treating thousands of windows as
independent samples would give falsely tight intervals. The CIs are bootstrapped over
**subjects**. Model comparisons are **paired** per subject (the same person is hard or easy
for every model), which is what exposes a consistent 2-point gain hidden inside a 10-point
spread between people.

## Findings

1. **EEGNet beats CSP+LDA across subjects, modestly.** +2.4 pt on the 21 motor channels,
   +1.5 pt on 64 channels, both with CIs excluding zero. It is also above chance for more
   people (101/106 vs. 92/106 on 64 channels).
2. **All 64 channels do not clearly help EEGNet.** 64 vs. 21 channels: +1.1 pt, CI −0.2 to
   +2.4, so the difference is not distinguishable from zero. For CSP+LDA the 64-channel
   gain is real (+2.0 pt cross-subject, +3.4 pt within-subject).
3. **Served model: EEGNet on 21 motor channels.** For a ~1 pt accuracy difference that the
   data cannot confirm, the 64-channel model costs 3x the data per window job (102 to 123 KB
   vs. 34 to 40 KB), 3x the Redis traffic, and twice the latency (1.85 ms vs. 0.93 ms p50).
   The 21 channels (FC, C and CP rows) cover the motor cortex, where imagined hand movement
   shows up.
4. **Cross-subject is hard: ~46% is the realistic level.** Motor-imagery EEG varies a lot
   between people (subject std ~10 pt), and the model has never seen the test person.
   Subjects never cross the train/test split (tested in
   `test_subject_folds_never_share_subjects`), so nothing leaks to inflate the number.
5. **Within-subject EEGNet fails at chance (34.7%), and that is a real result, not a bug.**
   Each person has only one training run: 128 to 133 heavily overlapping windows from
   about 2 minutes of EEG, with 13 to 15 imagery trials for both hands together. The training loss barely moves: median 1.103 after the first
   epoch and 1.079 at the end, against 1.099 (ln 3) for uniform guessing. 13 subjects
   early-stopped at epoch 0, and the model collapses onto `right` (66% of predictions vs.
   25% of labels). CSP+LDA has far fewer free parameters and gets 50.0% to 53.6% from the
   same data. The sensible fix is fine-tuning the cross-subject EEGNet on a person's short
   calibration recording (the spec's "per-user calibration" stretch goal), not training a
   CNN from scratch per person. Note that CSP+LDA also trains on run 8 while EEGNet uses
   run 8 only for early stopping, so EEGNet sees half the data here.
6. **Rest is the hardest class for the served model.** Pooled recall: rest 36.5%, left
   46.3%, right 54.3%. Rest windows get spread over left and right, which in a live system
   means false "imagery" detections while the user is idle. Temporal smoothing of
   predictions (Phase 4/6) is the usual remedy.
7. **EEGNet early-stops early and validation scores are noisy.** On 64 channels three of
   five folds picked epoch 1 as best. Validation is only 8 people, so per-epoch balanced
   accuracy moves by a few points and early stopping is a noisy choice. A larger
   validation set or longer patience was not tried, to keep the protocol fixed across
   folds; it is the first thing to try for a better model.

## Serving path

- **ONNX export** (`neurostream_ml.export`): fold-0 checkpoints exported with the PyTorch
  `dynamo` exporter, opset 18, dynamic batch size. Each export is checked against PyTorch
  with ONNX Runtime before it is written: max probability difference 8.9e-8 (motor) and
  1.5e-7 (64 ch). Weights are embedded in `model.onnx` (~28 KB), so the SHA-256 in
  `model_card.json` covers them. The model card also records the channels, filter and
  normalization a worker must apply.
- **Exact serving path, held-out subjects:** running the exported ONNX model with the
  worker's float32 preprocessing on fold 0's 22 test subjects reproduces the training-time
  evaluation (motor: 43.7% both ways; 64 ch: 45.5% both ways at the training pad of 1 s).
- **Filter settling and pad length:** the band-pass impulse response falls below 1% of its
  peak after 0.32 s and below 0.1% after 0.51 s. With 0.5 s of pad, the filtered window
  differs from a fully settled one by 0.03% RMS (median). Accuracy is within 0.1 pt from 0.375
  to 1 s of pad and within 0.2 pt at 0.25 s. **Applied in Phase 4:** the system serves with a
  0.5 s pad instead of 1 s, which makes window jobs 17% smaller (64 channels: 102.4 vs.
  122.9 KB) at no measurable accuracy cost. `eegnet-motor-1.0.1` is the same model
  (identical ONNX file and SHA-256) with the 0.5 s serving pad in its card.
- **Latency:** per-window preprocessing + inference in one process with one ONNX Runtime
  thread on a laptop CPU, 2000 windows. This is compute only; queueing and Redis round trips
  are measured end to end in Phases 4 and 7.

## Reproduce

```powershell
python scripts/download_data.py                       # all subjects, ~1 GB
python -m neurostream_ml.experiments csp              # CSP+LDA, both channel sets
python -m neurostream_ml.experiments eegnet-cross --channels motor --epochs 30 --patience 6
python -m neurostream_ml.experiments eegnet-cross --channels all   --epochs 30 --patience 6
python -m neurostream_ml.experiments eegnet-within --channels motor --epochs 30 --patience 6
python -m neurostream_ml.settling
python -m neurostream_ml.export --channels motor --fold 0 --version 1.0.0
python -m neurostream_ml.export --channels all   --fold 0 --version 1.0.0
python -m neurostream_ml.serving_eval pad-sweep --model models/eegnet-motor-1.0.0 --model models/eegnet-all-1.0.0
python -m neurostream_ml.serving_eval latency
python -m neurostream_ml.report_tables --update docs/MODEL_RESULTS.md
```

On this laptop (8 CPU threads, no GPU) one cross-subject EEGNet fold takes 7 to 12 minutes
on 21 channels and 15 to 32 minutes on 64. Training is deterministic (fixed seeds): a run
restarted from scratch repeated the interrupted run's per-epoch losses exactly. EEGNet runs
save results after every fold and the full training state after every epoch, so an
interrupted run resumes where it stopped and produces the same model.

Training times in the tables are wall-clock on a shared laptop, not a controlled benchmark.

<!-- BEGIN GENERATED TABLES -->
<!-- python -m neurostream_ml.report_tables --update docs/MODEL_RESULTS.md -->

## Cross-subject (5 folds, each subject tested once)

| Model / channels | Balanced acc. | 95% CI (subjects) | Subject std | Macro-F1 | Accuracy | Majority baseline | Subjects > 33.3% |
|---|---|---|---|---|---|---|---|
| CSP+LDA, 64 ch | **45.3%** | 43.3% to 47.3% | 10.7% | 44.1% | 45.2% | 49.5% | 92/106 |
| CSP+LDA, 21 motor ch | **43.3%** | 41.4% to 45.1% | 9.8% | 42.1% | 43.0% | 49.5% | 89/106 |
| EEGNet, 64 ch | **46.7%** | 44.8% to 48.7% | 10.3% | 45.3% | 46.4% | 49.5% | 101/106 |
| EEGNet, 21 motor ch | **45.7%** | 43.7% to 47.7% | 10.7% | 43.1% | 43.4% | 49.5% | 97/106 |

## Within-subject (test run 12)

CSP+LDA trains on runs 4 and 8. EEGNet trains on run 4 only and uses run 8 for early stopping, so it sees half the training data: the EEGNet row is a lower bound.

| Model / channels | Balanced acc. | 95% CI (subjects) | Subject std | Macro-F1 | Accuracy | Majority baseline | Subjects > 33.3% |
|---|---|---|---|---|---|---|---|
| CSP+LDA, 64 ch | **53.6%** | 50.9% to 56.3% | 14.4% | 54.0% | 59.8% | 49.5% | 100/106 |
| CSP+LDA, 21 motor ch | **50.0%** | 47.8% to 52.6% | 12.6% | 50.1% | 54.2% | 49.5% | 97/106 |
| EEGNet, 21 motor ch | **34.7%** | 33.7% to 35.9% | 5.7% | 28.4% | 33.5% | 49.5% | 52/106 |

## Paired per-subject differences (balanced accuracy, 95% bootstrap CI)

| Comparison | Mean difference | 95% CI | Subjects |
|---|---|---|---|
| cross: EEGNet, 64 ch minus EEGNet, 21 motor ch | +1.1 pt | -0.2 to +2.4 | 106 |
| cross: CSP+LDA, 64 ch minus CSP+LDA, 21 motor ch | +2.0 pt | +0.9 to +3.1 | 106 |
| cross: EEGNet, 21 motor ch minus CSP+LDA, 21 motor ch | +2.4 pt | +1.1 to +3.8 | 106 |
| cross: EEGNet, 64 ch minus CSP+LDA, 64 ch | +1.5 pt | +0.1 to +2.8 | 106 |
| within: CSP+LDA, 64 ch minus CSP+LDA, 21 motor ch | +3.4 pt | +1.8 to +5.1 | 106 |
| within: EEGNet, 21 motor ch minus CSP+LDA, 21 motor ch | -15.4 pt | -17.9 to -12.9 | 106 |

## EEGNet per fold (cross-subject)

| Channels | Fold | Balanced acc. | Macro-F1 | Best epoch | Val bal. acc. | Train time |
|---|---|---|---|---|---|---|
| all | 0 | 45.5% | 42.7% | 1 | 42.2% | 15.4 min |
| all | 1 | 46.6% | 45.6% | 1 | 45.1% | 17.6 min |
| all | 2 | 41.9% | 41.0% | 1 | 41.9% | 17.1 min |
| all | 3 | 47.0% | 44.9% | 8 | 45.7% | 31.7 min |
| all | 4 | 52.5% | 51.6% | 13 | 44.1% | 31.9 min |
| motor | 0 | 43.7% | 40.7% | 10 | 40.0% | 11.6 min |
| motor | 1 | 46.2% | 45.0% | 6 | 41.4% | 9.0 min |
| motor | 2 | 43.3% | 41.7% | 7 | 39.4% | 9.7 min |
| motor | 3 | 45.4% | 42.0% | 7 | 42.6% | 9.8 min |
| motor | 4 | 50.0% | 46.0% | 5 | 39.8% | 7.2 min |

## Confusion matrices (cross-subject, pooled)

**CSP+LDA, 64 ch** (rows = true, columns = predicted, row-normalized)

| | rest | left | right |
|---|---|---|---|
| **rest** | 44.9% | 29.1% | 26.0% |
| **left** | 28.2% | 43.7% | 28.1% |
| **right** | 23.0% | 29.8% | 47.3% |

**CSP+LDA, 21 motor ch** (rows = true, columns = predicted, row-normalized)

| | rest | left | right |
|---|---|---|---|
| **rest** | 42.1% | 32.4% | 25.5% |
| **left** | 29.4% | 42.3% | 28.3% |
| **right** | 22.2% | 32.2% | 45.5% |

**EEGNet, 64 ch** (rows = true, columns = predicted, row-normalized)

| | rest | left | right |
|---|---|---|---|
| **rest** | 45.6% | 28.6% | 25.8% |
| **left** | 25.4% | 46.4% | 28.3% |
| **right** | 22.2% | 29.8% | 48.0% |

**EEGNet, 21 motor ch** (rows = true, columns = predicted, row-normalized)

| | rest | left | right |
|---|---|---|---|
| **rest** | 36.5% | 32.8% | 30.7% |
| **left** | 21.9% | 46.3% | 31.8% |
| **right** | 16.7% | 29.0% | 54.3% |

## Filter settling

Impulse response: amplitude < 1% of peak after 0.319 s, < 0.1% after 0.506 s; 99% of its energy within 0.169 s, 99.9% within 0.250 s.

Relative RMS error of the filtered window versus a long-settled reference (1200 windows):

| Pad | Init | Core median | Core p95 | First 100 ms median | First 100 ms p95 |
|---|---|---|---|---|---|
| 0 s | steady | 15.27% | 28.08% | 75.14% | 96.40% |
| 0.0625 s | steady | 6.32% | 13.95% | 30.67% | 58.78% |
| 0.125 s | steady | 3.15% | 6.58% | 14.97% | 33.53% |
| 0.25 s | steady | 0.67% | 1.56% | 3.11% | 7.09% |
| 0.375 s | steady | 0.15% | 0.36% | 0.69% | 1.74% |
| 0.5 s | steady | 0.03% | 0.07% | 0.15% | 0.38% |
| 0.75 s | steady | 0.00% | 0.00% | 0.01% | 0.02% |
| 1 s | steady | 0.00% | 0.00% | 0.00% | 0.00% |
| 0 s | zero | 19.34% | 39.51% | 90.29% | 176.87% |
| 0.0625 s | zero | 8.99% | 23.37% | 41.87% | 105.90% |
| 0.125 s | zero | 4.32% | 11.41% | 20.63% | 54.68% |
| 0.25 s | zero | 0.98% | 2.31% | 4.63% | 11.81% |
| 0.375 s | zero | 0.21% | 0.54% | 1.01% | 2.93% |
| 0.5 s | zero | 0.04% | 0.11% | 0.21% | 0.60% |
| 0.75 s | zero | 0.00% | 0.01% | 0.01% | 0.03% |
| 1 s | zero | 0.00% | 0.00% | 0.00% | 0.00% |

## Pad sweep: eegnet-all 1.0.0 (exact float32 serving path, 22 held-out subjects)

| Serving pad | Windows | Balanced acc. | Macro-F1 | Accuracy |
|---|---|---|---|---|
| 0.25 s | 8619 | 45.6% | 42.8% | 43.7% |
| 0.375 s | 8619 | 45.5% | 42.7% | 43.6% |
| 0.5 s | 8619 | 45.5% | 42.7% | 43.6% |
| 0.75 s | 8619 | 45.5% | 42.7% | 43.6% |
| 1 s | 8619 | 45.5% | 42.7% | 43.6% |

## Pad sweep: eegnet-motor 1.0.0 (exact float32 serving path, 22 held-out subjects)

| Serving pad | Windows | Balanced acc. | Macro-F1 | Accuracy |
|---|---|---|---|---|
| 0.25 s | 8619 | 43.7% | 40.7% | 40.9% |
| 0.375 s | 8619 | 43.7% | 40.7% | 40.9% |
| 0.5 s | 8619 | 43.7% | 40.7% | 40.9% |
| 0.75 s | 8619 | 43.7% | 40.7% | 40.9% |
| 1 s | 8619 | 43.7% | 40.7% | 40.9% |

## Latency and data volume (single process, 1 ORT thread, this laptop CPU, 2000 windows)

| Model | Pad | Job size (model ch.) | Job size (64 ch.) | Redis per session | Preprocess p50 | Inference p50 | Total p50 | Total p95 | Total p99 |
|---|---|---|---|---|---|---|---|---|---|
| eegnet-motor | 0.5 s | 33.6 KB | 102.4 KB | 67.2 KB/s | 0.51 ms | 0.39 ms | 0.90 ms | 1.10 ms | 1.52 ms |
| eegnet-motor | 1 s | 40.3 KB | 122.9 KB | 80.6 KB/s | 0.53 ms | 0.40 ms | 0.93 ms | 1.16 ms | 1.51 ms |
| eegnet-all | 0.5 s | 102.4 KB | 102.4 KB | 204.8 KB/s | 0.77 ms | 1.04 ms | 1.82 ms | 2.15 ms | 2.76 ms |
| eegnet-all | 1 s | 122.9 KB | 122.9 KB | 245.8 KB/s | 0.81 ms | 1.03 ms | 1.85 ms | 2.32 ms | 2.99 ms |

<!-- END GENERATED TABLES -->
