# CS-479-Assignment-2

## [Report][report/report.pdf]
## Running Code

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it:

- Windows (PowerShell)
```powershell
.venv\Scripts\Activate.ps1
```

- Windows (bash)
```bash
source .venv/Scripts/activate
```

- macOS/Linux
```bash
source .venv/bin/activate
```

### GPU Requirements

GPU acceleration requires a CUDA-compatible NVIDIA GPU with the matching CUDA Toolkit installed.

First, check your CUDA version:

```bash
nvcc --version
```

Update requirements.txt file to reflect the version you have installed. 

Verify CuPy can see your GPU:

```bash
python gpu_check.py
```

If you see a `DLL load failed` error, the installed cupy version does not match your CUDA toolkit — uninstall cupy and reinstall with the correct version above, or use `--cpu`.

## Running the Experiments

Install dependencies:

```bash
pip install -r requirements.txt
```

By default, the experiment auto-detects GPU availability and falls back to CPU if unavailable.

- **Default:**
```bash
python experiment_1.py
```

- **Force CPU (NumPy):**
```bash
python experiment_1.py --cpu
```

- **Force GPU (CuPy) — exits with an error if unavailable:**
```bash
python experiment_1.py --gpu
```
