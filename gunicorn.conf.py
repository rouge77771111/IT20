import os

# Automatically extract Railway's PORT environment variable
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Memory-optimized worker configuration for Railway free container
workers = 1
threads = 4
timeout = 120

accesslog = "-"
errorlog = "-"
