# dashboard – PolyRoute web app (Flask)

One page: inputs on the left, results on the right.

```
form ─► classifier records ─► comp_analyzer ─► optimiser ─► explainer ─► page
        (integrations.py)     (real)           (stand-in)   (stand-in)
```

## Run it

From `src/`:
```bash
pip install flask
python -m comp_analyzer.gen      # sample classifier outputs, if not done yet
python -m dashboard.app          # http://127.0.0.1:8501
```

On LinuxONE, inside the `app` container (port 8501 is already mapped to the VM's localhost):
```bash
docker exec app pip install flask
docker exec -it -w /shared/<repo>/src -e POLYROUTE_HOST=0.0.0.0 app python -m dashboard.app
```
Then on your laptop: `ssh -L 8501:localhost:8501 linuxone` and open http://localhost:8501.
The footer shows `Running on IBM LinuxONE (s390x)` there.

## Files

| File | Job |
|---|---|
| `app.py` | Flask routes: `GET /` (empty page), `POST /analyse` (run the chain, show results); host/port from `POLYROUTE_HOST` / `POLYROUTE_PORT` |
| `pipeline.py` | Reads the form, runs comp_analyzer → optimiser → explainer, builds everything the page shows (`run()`, `read_inputs()`, `build_view()`) |
| `integrations.py` | **The only place that talks to other teammates' modules.** Each has a stand-in until the real module is ready |
| `templates/index.html` | The page |
| `static/style.css` | Styles (no chart library; bars are HTML/CSS, so nothing depends on a CDN except the font) |

## Swapping in the real modules (`integrations.py`)

| Function | Stand-in now | Replace with |
|---|---|---|
| `classify_photos(files)` | not connected | Classifier: photos → list of records (format in `comp_analyzer/CLASSIFIER_OUTPUT_SPEC.md`) |
| `run_optimiser(composition_data, limits)` | `_standin_optimiser`: greedy, placeholder factors, not optimal | `from optimiser import optimise; return optimise(composition_data, limits)` |
| `explain_plan(plan_result)` | text template | Granite call; keep the template as a fallback if Granite is slow |

`run_optimiser` must return:
```python
{"plan": {"PET": {"recycle": 20.0, "combust": 11.2}, ...},
 "optimised_kgco2e": -12.4, "baseline_kgco2e": 0.9, "saved_kgco2e": 13.3, "notes": ["..."]}
```
Routes: `recycle`, `compost`, `combust`, `landfill`. If the optimiser's names differ, map them inside `run_optimiser`.

## Inputs on the page

- **Classifier output:** a sample from `comp_analyzer/data/synthetic/`, or an uploaded JSON file
- **Weight:** weighed (kg), volume (m³), item count, or "the photos show the whole batch"
- **Facility limits:** recycling capacity, composting capacity, landfill limit (defaults 40 / 10 / 15 kg)

Bad input (missing weight, non-numbers, invalid JSON, wrong class names) shows a clear message instead of crashing.
