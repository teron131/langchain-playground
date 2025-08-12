"""
Railway deployment entry point.
Imports the FastAPI app from api/youtube_with_ui.py for Railway's auto-detection.
"""

from api.youtube_with_ui import app

# This allows Railway to auto-detect and start the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
