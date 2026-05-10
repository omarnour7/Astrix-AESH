# GeoRadar — Demo Video Script
**Team Astrix · AESH 2026 · Target duration: 2–3 minutes**

---

## Segment breakdown

| Time | What to show on screen | Script |
|---|---|---|
| 0:00 – 0:20 | Title slide: "GeoRadar — Underground Water Detection" + team name + SDG badges | "We are Team Astrix. GeoRadar is a software-simulated Ground Penetrating Radar system that locates hidden groundwater in drought-affected regions — using 70% less power than conventional always-on GPR." |
| 0:20 – 0:45 | Show the pipeline architecture diagram | "Our system has four stages: FMCW signal generation at 200 MHz to 1 GHz, signal preprocessing with bandpass filtering and time-gain correction, then our adaptive duty-cycle gate — which only activates the radar when terrain entropy suggests an anomaly." |
| 0:45 – 1:15 | Live terminal: run `python simulation/run_pipeline.py --skip_train` | "Here we run the full pipeline. The signal generator creates synthetic GPR A-scans across configurable soil layers using dielectric constants from NASA and ESA open datasets. The duty-cycle controller decides which scan positions actually need radar." |
| 1:15 – 1:45 | Show power_comparison.csv / power comparison bar chart | "The power comparison tells the core story. Conventional GPR draws 2 watts continuously. GeoRadar activates for roughly 30% of positions, bringing average consumption down to about 0.6 watts — a 70% reduction." |
| 1:45 – 2:15 | Launch Streamlit dashboard: `streamlit run src/dashboard.py` — show 3D map rotating | "The Streamlit dashboard renders an interactive 3D groundwater map. Blue voxels are water-bearing layers, brown is dry soil, gray is rock. The CNN classifier assigns each scan position to one of these three classes." |
| 2:15 – 2:45 | Show CNN confidence heatmap, briefly show classification report | "Our CNN achieves over 85% accuracy on the held-out synthetic test set. The confidence map shows where the model is certain — highest confidence over the central water zone." |
| 2:45 – 3:00 | Closing slide with GitHub link + SDG alignment | "All code, notebooks, datasets, and results are in our public repository. We're honest that these are simulated results — but the pipeline is complete, reproducible, and directly applicable to arid regions across North Africa and the Middle East." |

---

## Recording tips

- Run `python simulation/run_pipeline.py --skip_train` live — it's fast and shows real terminal output.
- Launch the dashboard before recording and have it ready to switch to at the 1:45 mark.
- Rotate the 3D map slowly with the mouse during the dashboard segment — motion makes it readable.
- Keep the `results/power_comparison.csv` and `results/cnn_accuracy_report.txt` open in a text editor as a fallback visual.
- Add subtitles if recording in a noisy environment.

## File name

```
Astrix_GeoRadar_Phase1_Video.mp4
```
