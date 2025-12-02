<img src="images/logo-compact.png" style="height:64px;margin-right:32px" alt=""Logo COMPACT>

# Project Documentation: Meniscus Displacement Analysis

The following project contains the files for analyzing the displacement of a meniscus in a microfluidic channel.

## Main Analysis File: `main.py`

- **Input**: Video to analyze; set the path in the `VIDEO_SOURCE` variable.
- **Output**: Generates an output folder containing:
    - `log.csv` – list of operations performed during the analysis.
    - `processed_video.avi` – the video with the detected features overlaid.
    - `automated.csv` – list of automatic measurements (e.g., meniscus position, frame number, etc.).


## Merging Datasets: `merge_datasets.py`

This file allows merging the manual measurements (for the three proposed videos) with the automatic measurements obtained from the developed software.
It therefore combines: `automated.csv` + `manual.csv`.

## Error Comparison: `compare_error.py`

This file generates:

- `compare.jpg` – plot of the observations obtained manually and automatically.
- `quadratic_regression.jpg` – plot showing the quadratic regressions of the deviations obtained manually and automatically.
- `error.txt` – contains the error analysis of the software’s predictions with respect to the max-min range, mean, and median of the observed values.

## Folders

In the folder optical_data there is the diagram with the specifications of the microscope used: Dino‑Lite.

In the video folder there are three of the videos used during the experiments.

	- 20251015_154007768-15fps-x20RTL.mov
	- 20241023_174413659_5ul_0glic-x20RTL.mov
	- 20241022_172649156_5ul_20glic-x20RTL.mov
	
It is possible to obtain example videos of some experiments by writing to gabriele.surano@cnr.it or hpc.nanotec@cnr.it
