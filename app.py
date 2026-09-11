"""
Entrypoint wrapper for DepthWizard Web Application.
Launches the server from web/backend/app.py.
"""
from web.backend.app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
