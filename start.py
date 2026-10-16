"""
Railway startup script for Student Academic Performance Predictor.
Safely extracts PORT from environment and launches Gunicorn with optimized memory workers.
"""

import os
import sys
import subprocess

def main():
    port = os.environ.get('PORT', '5000')
    print(f"[STARTUP] Launching application on port {port} with 1 worker and 4 threads for memory efficiency...")

    # Train or verify model artifacts exist before booting workers
    try:
        from app import load_artifacts
        load_artifacts()
    except Exception as e:
        print(f"[WARNING] Pre-load check encountered: {e}. Retrying via train_models...")
        try:
            from train_models import train_and_save
            train_and_save()
        except Exception as te:
            print(f"[ERROR] Could not train models: {te}")

    cmd = [
        sys.executable,
        "-m", "gunicorn",
        "app:app",
        "--bind", f"0.0.0.0:{port}",
        "--workers", "1",
        "--threads", "4",
        "--timeout", "120",
        "--access-logfile", "-",
        "--error-logfile", "-"
    ]
    
    sys.exit(subprocess.call(cmd))

if __name__ == '__main__':
    main()
