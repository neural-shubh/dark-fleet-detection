# Data

This project uses two open-access datasets.

## 1. SARscope (CNN Branch)

SAR ship detection dataset combining HRSID and OPEN-SSDD corpora.

- Download from Roboflow: https://universe.roboflow.com
- Search for "SARscope"
- Export in COCO format
- Place in `data/sarscope/`

## 2. Global Fishing Watch — Sentinel-1 SAR Vessel Detections (LSTM/RNN Branches)

- Download from: https://globalfishingwatch.org/data-download/datasets/public-sentinel-1-detections:v20231026
- File used: `sar_vessel_detections_pipev4_202603.csv`
- Place in `data/gfw/`

## Indian Ocean Filter

Applied in the notebook:
- Latitude: −30° to 30°N
- Longitude: 40° to 100°E
- Yields 16,795 records from 107,257 total
