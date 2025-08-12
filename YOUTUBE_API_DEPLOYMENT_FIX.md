# YouTube API Deployment Fix Guide

## ❌ Issues Identified

The initial deployment failed due to two critical problems:

### 1. Missing `__init__.py` File
- **Error**: `ERROR: Error loading ASGI app. Could not import module "api.youtube_main"`
- **Cause**: The `api` directory lacked an `__init__.py` file, preventing Python from treating it as a package
- **Fix**: ✅ Created [`api/__init__.py`](api/__init__.py)

### 2. Incorrect PYTHONPATH Configuration
- **Error**: `python: can't open file '/opt/render/project/src/api/youtube_main.py': [Errno 2] No such file or directory`
- **Cause**: Original `render.yaml` set `PYTHONPATH: /opt/render/project/src` but files are in project root, not `/src`
- **Fix**: ✅ Removed incorrect PYTHONPATH and used proper uvicorn command

## ✅ Fixed Configuration

### Corrected Render Configuration (`render-youtube.yaml`)

```yaml
services:
  - type: web
    name: youtube-summarizer-youtube
    runtime: python
    region: singapore
    plan: starter
    buildCommand: "pip install --upgrade pip && pip install -r requirements-youtube.txt"
    startCommand: "uvicorn api.youtube_main:app --host 0.0.0.0 --port $PORT"
    healthCheckPath: /health
    envVars:
      - key: PYTHONUNBUFFERED
        value: "1"
      - key: GEMINI_API_KEY
        sync: false
      - key: FAL_KEY
        sync: false
```

### Key Changes Made

| Issue | Before | After |
|-------|--------|-------|
| **Start Command** | `python api/youtube_main.py --host 0.0.0.0 --port $PORT` | `uvicorn api.youtube_main:app --host 0.0.0.0 --port $PORT` |
| **PYTHONPATH** | `/opt/render/project/src` (incorrect) | Removed (not needed) |
| **Package Structure** | Missing `api/__init__.py` | Added `api/__init__.py` |
| **Requirements** | Heavy dependencies | Lightweight `requirements-youtube.txt` |

## 🚀 Deployment Instructions

### 1. Use the Fixed Configuration
Deploy using the corrected configuration file:
```bash
# Use the fixed render configuration
render deploy --config render-youtube.yaml
```

### 2. Required Files
Ensure these files exist:
- ✅ [`api/__init__.py`](api/__init__.py) - Package marker
- ✅ [`api/youtube_main.py`](api/youtube_main.py) - Main FastAPI app
- ✅ [`requirements-youtube.txt`](requirements-youtube.txt) - Dependencies
- ✅ [`render-youtube.yaml`](render-youtube.yaml) - Deployment config

### 3. Environment Variables
Set these in your Render dashboard:
- `GEMINI_API_KEY` - Your Google Gemini API key
- `FAL_KEY` - Your Fal AI API key

## 🔧 Dependencies

The [`requirements-youtube.txt`](requirements-youtube.txt) includes:

```txt
# Core API framework
fastapi>=0.115.12
uvicorn>=0.34.3
pydantic>=2.5.0

# YouTube processing
yt-dlp>=2024.8.6
pydub>=0.25.1
requests>=2.31.0

# LLM and AI processing
google-generativeai>=0.8.3
langchain>=0.3.25
langchain-openai>=0.2.8
langchain-core>=0.3.15
fal-client>=0.6.0

# Text processing and utilities
opencc>=1.1.7
python-dotenv>=1.0.0
```

## 🎯 API Endpoints

After successful deployment, the following endpoints will be available:

- `GET /` - Health check and API info
- `GET /health` - Service health status
- `POST /api/youtube` - Process YouTube videos
- `GET /api/test` - API functionality test

## 🐛 Troubleshooting

### Import Errors
If you see `Could not import module "api.youtube_main"`:
1. ✅ Ensure `api/__init__.py` exists
2. ✅ Use `uvicorn api.youtube_main:app` (not direct Python execution)
3. ✅ Don't set incorrect PYTHONPATH

### File Not Found Errors
If you see `No such file or directory`:
1. ✅ Remove `/src` from PYTHONPATH 
2. ✅ Use correct file paths relative to project root
3. ✅ Use `uvicorn` instead of `python` command

### Dependency Errors
If build fails:
1. ✅ Use `requirements-youtube.txt` (lightweight)
2. ✅ Don't use heavy browser dependencies for this API
3. ✅ Ensure all required packages are listed

## 📊 Expected Deployment Flow

```
1. Build Phase:
   ✅ pip install --upgrade pip
   ✅ pip install -r requirements-youtube.txt

2. Start Phase:
   ✅ uvicorn api.youtube_main:app --host 0.0.0.0 --port $PORT

3. Health Check:
   ✅ GET /health returns {"status": "healthy"}

4. Ready for requests:
   ✅ POST /api/youtube processes YouTube videos
```

## 🔄 Next Steps

After this fix is deployed:
1. Test the API endpoints
2. Monitor the deployment logs
3. Verify YouTube video processing works
4. Update any client applications to use the new endpoint

---

**Fix Applied**: 2025-08-12T03:09:00Z
**Files Modified**: `api/__init__.py` (created), `render-youtube.yaml` (created)
**Status**: Ready for deployment ✅