# LPPL Analysis Pipeline: Logarithmic Prime Phase Law

This repository contains the computational pipeline used to generate, analyze, and fit the data for the **Logarithmic Prime Phase Law (LPPL)** project. The core task is calculating the unwrapped cumulative phase $\Theta(X)$ of the prime vector sum $S(X) = \sum_{p \le X} p^i$ for massive ranges $X$.

The pipeline is designed for **memory efficiency** and **scalability**, handling huge data streams by processing prime angles in streamable chunks.

---

## 1. Workflow Overview

The standard execution pipeline follows three main stages, often used for ranges up to $X=10^{12}$:

1.  **Generation:** Create memory-safe, gzipped angle files ($\phi(p) = \log p \pmod{2\pi}$) for sequential large ranges (e.g., $10^{10}$ to $2 \times 10^{10}$).
2.  **Analysis (Core):** Stream the angle files, calculate circular statistics (KDE, resultant vector $V$, phase $\Theta(X)$) at specified checkpoints $X$.
3.  **Fitting:** Merge the final analysis data and perform logarithmic regression to verify the LPPL constants ($\Gamma_p$ and $k \equiv 1$).

---

## 2. Script Descriptions

### `generate_custom_sane.py`

| Purpose | Generate raw prime phase angle data in memory-safe chunks. |
| :--- | :--- |
| **Function** | Uses a **segmented Sieve of Eratosthenes** to find primes in a range $(X_{\min}, X_{\max}]$. It calculates the angle $\phi(p) = \log p \pmod{2\pi}$ for each prime and writes the result **streamingly** to disk as a gzipped CSV (`.csv.gz`). This is the starting point of the pipeline. |
| **Key Args** | `X_max` (positional), `--X-min`, `--segment-size`, `--dtype` (`float32` recommended for space saving). |

### `phase_stream_runner_fast_chunked.py`

| Purpose | **Core Analysis Tool:** Calculate phase $\Theta(X)$ at a single checkpoint. |
| :--- | :--- |
| **Function** | A **memory-safe streamer** designed to process massive angle files incrementally. It builds circular histograms, performs a deterministic **Train/Test split**, calculates the resultant vector $V$, and determines the phase $\Theta(X)$ and resultant length $R$. It uses **Circular Gaussian KDE** for smoothing, controlled by the `sigma` parameter. |
| **Usage** | This script is typically called repeatedly by the `run_checkpoints_over_chunks.py` orchestration script. |
| **Key Args** | `--checkpoints` (list of X values), `--angles-files` (X:path mapping), `--sigmas` (smoothing parameter list). |

### `run_checkpoints_over_chunks.py`

| Purpose | **Workflow Orchestration:** Manage analysis over multiple data chunks. |
| :--- | :--- |
| **Function** | The **wrapper script** for production runs. It takes a list of **angle file chunks** (`LO:HI:FILE`) and a list of target **checkpoints** $X$. It intelligently maps checkpoints to the correct files and executes `phase_stream_runner_fast_chunked.py` for each required analysis run, appending all results to a single analysis CSV file. |
| **Usage** | Primary command used to launch large-scale computations. |
| **Key Args** | `--chunks` (e.g., `1e10:2e10:file.csv.gz`), `--checkpoints`, `--out-csv`. |

### `merge_csv_gz.py`

| Purpose | Utility to merge analysis results efficiently. |
| :--- | :--- |
| **Function** | Merges multiple CSV or gzipped CSV files with the same column schema into a single, master GZipped CSV. It correctly handles the header, writing it only once, even when using streaming chunks to manage memory. |
| **Usage** | Used to consolidate analysis outputs from different batches or nodes. |

### `fit_from_analysis_cli.py`

| Purpose | Post-processing: Perform regression analysis and verify the LPPL. |
| :--- | :--- |
| **Function** | Reads the final aggregated analysis CSV, filters by the desired smoothing parameter (`--sigma`), and performs **logarithmic regression** (linear and quadratic fits on $\Theta(X)$ vs. $\log X$). It determines the LPPL constants (`a` and `b`), calculates $R^2$ scores, and generates plots of the fit and the residuals. |
| **Usage** | The final step to verify the Logarithmic Prime Phase Law against the data. |
| **Key Args** | `--analysis`, `--sigma`, `--theta-col` (e.g., `theta_star_deg_test`), `--save-plot`, `--save-json`. |

---

## 3. Example Execution Flow

This sequence demonstrates the typical flow for analyzing a range, where the data is segmented into 10 billion chunks:

1.  **Generate a Data Chunk (e.g., 7e10 to 8e10):**
    ```bash
    python generate_custom_sane.py 80000000000 --X-min 70000000000 --outdir data --prefix angles_7e10_8e10_
    ```

2.  **Run Orchestration over All Chunks:**
    (The `run_checkpoints_over_chunks.py` command is launched once, pointing to all generated files and desired checkpoints $X$).

    ```bash
    python run_checkpoints_over_chunks.py \
      --chunks \
        1e10:2e10:angles_1e10_2e10.csv.gz \
        2e10:3e10:angles_2e10_3e10.csv.gz \
        # ... list continues ... \
      --checkpoints 10000000000 20000000000 30000000000 ... \
      --sigmas 0.03 \
      --out-csv analysis_master.csv
    ```

3.  **Perform Final Logarithmic Fit:**
    ```bash
    python fit_from_analysis_cli.py \
      --analysis analysis_master.csv \
      --sigma 0.03 \
      --theta-col theta_star_deg_test \
      --save-plot final_lp_fit.png
    ```
