import os

# Railway default port is 8080
port = os.environ.get("PORT", "8080")
bind = f"0.0.0.0:{port}"

workers = 1
threads = 4
timeout = 120

accesslog = "-"
errorlog = "-"
