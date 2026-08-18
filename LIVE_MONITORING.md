# Running 24-hour live ECG monitoring (DeepHHF-Inspired)

**Research prototype only — not a medical device.** `monitor-live` does not issue
clinical alerts, does not control any device, and emits no score until a full
24-hour recording (11,059,200 samples at 128 Hz) has been captured.

## Status: model already trained

`outputs/encoder/encoder_best.pt`, `outputs/sequence/sequence_best.pt`,
`outputs/evaluation/calibrator.pkl`, and `outputs/evaluation/risk_thresholds.json`
already exist and were trained/evaluated on **both ECG classes**
(`data_validation.json` reports 8 `normal` + 8 `abnormal` recordings, and
`data/prepared/manifest.csv` includes both classes in every split). You do not
need to retrain before running live monitoring — see `outputs/evaluation/metrics.json`
for the (experimental, synthetic-data) evaluation numbers.

I independently re-validated the whole live-monitoring code path against these
exact checkpoints in an isolated environment (feeding 24 hours' worth of
synthetic chunks at high speed, not real-time) and confirmed it correctly
reaches `"status": "complete"` and produces a full `live_prediction.json` +
beat-cluster image. See `outputs/live_monitor_demo/` for that proof run.

## New file: `feed_live_ecg.py`

Your project didn't have anything that plays the role of the real ECG
acquisition bridge that `monitor-live` expects to be fed by. This script fills
that role — either as a stand-in until you wire up a real device, or as a way
to test the pipeline. It writes single-lead `.npy` chunks atomically (temp file
+ rename) into a directory, exactly the way the README's monitor-live section
requires. **If you have a real ECG acquisition device/bridge, point
`monitor-live --input-dir` at whatever directory *it* writes chunks into
instead of this script** — this script is only a substitute data source.

## Running it for real (three terminals, all in `D:\ECG`, venv activated)

```powershell
# One-time: create the input directory monitor-live will watch
mkdir data\live_acquisition
```

**Terminal 1 — the monitor** (stays running, watches for new chunks):

```powershell
deephhf monitor-live `
  --input-dir data\live_acquisition `
  --sampling-rate 128 `
  --encoder outputs\encoder\encoder_best.pt `
  --sequence outputs\sequence\sequence_best.pt `
  --calibrator outputs\evaluation\calibrator.pkl `
  --thresholds outputs\evaluation\risk_thresholds.json `
  --output-dir outputs\live_monitor `
  --poll-seconds 5
```

**Terminal 2 — the feeder** (simulates the acquisition device in real time —
one chunk every 5 minutes, 288 chunks, so the full recording completes in
~24 real hours, exactly like a real device streaming continuously):

```powershell
python feed_live_ecg.py --output-dir data\live_acquisition --sampling-rate 128 --chunk-seconds 300 --hours 24 --speed 1
```

**Terminal 3 — the live dashboard** (auto-refreshing web view):

```powershell
$env:DEEPHHF_LIVE_DIR = "outputs\live_monitor"
streamlit run dashboard.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`). It
shows capture progress and a decimated ECG trace live, and once the full
24 hours arrives it shows the research result (raw score, calibrated
probability, risk category, attended intervals, beat clusters).

### Want to see it complete without waiting 24 hours first?

Run the feeder with `--speed 3600` (finishes in ~24 seconds instead of 24
hours) to smoke-test your own setup before committing to a real-time run.
Use a separate `--output-dir` (e.g. `outputs\live_monitor_smoketest`) so it
doesn't collide with a real run's state.

### Restarting

`monitor-live` persists its capture buffer and state under `--output-dir`, so
stopping and restarting Terminal 1 (e.g. after a reboot) resumes safely
without losing already-captured samples or re-reading already-processed
chunks.

### Using a real ECG device instead of the simulator

Point `--input-dir` at whatever folder your real acquisition bridge writes
into, set `--sampling-rate` to the device's true output rate (the monitor
resamples to 128 Hz internally), and skip `feed_live_ecg.py` entirely. Chunks
must be written atomically (temp name, then rename) so the monitor never
reads a partial file.
