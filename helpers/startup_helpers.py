def select_array_module(args):
    """Return numpy or cupy as `xp` based on --cpu / --gpu args."""
    if args.cpu:
        import numpy as xp
        print("Using NumPy (CPU) — forced via --cpu")
    elif args.gpu:
        try:
            import cupy as xp
            xp.linalg.cholesky(xp.eye(2, dtype=float))  # validates cublas
            print("Using CuPy (GPU) — forced via --gpu")
        except Exception as e:
            print(f"GPU requested but CuPy is unavailable: {e}")
            print("Check your CUDA version:  nvcc --version")
            print("Install matching CuPy:    pip install cupy-cuda11x  (CUDA 11.x)")
            print("                       or pip install cupy-cuda12x  (CUDA 12.x)")
            print("Or run on CPU:            python <script>.py --cpu")
            raise SystemExit(1)
    else:
        try:
            import cupy as xp
            xp.linalg.cholesky(xp.eye(2, dtype=float))  # validates cublas
            print("GPU detected — using CuPy")
        except Exception:
            import numpy as xp
            print("No GPU detected — using NumPy")
    return xp
