"""
Railway deployment entry point.
Imports the standalone FastAPI app to avoid complex package dependencies.
"""

from api.youtube import app

# This allows Railway to auto-detect and start the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
