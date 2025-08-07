# 🔧 Render.com Deployment Debugging Guide

## 🚨 Current Issue: Deployment Stuck/Error

### Step 1: Simplified Deployment Test

Since the full Chrome installation might be causing issues, let's test with a minimal setup first:

1. **Use Simplified Config**: Rename `render-simple.yaml` to `render.yaml`
2. **Update Requirements**: Change build command to use `requirements-simple.txt`
3. **Test Basic Deployment**: Get the basic app running first

```bash
# In Render Dashboard -> Service Settings:
Build Command: pip install -r requirements-simple.txt
Start Command: uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

### Step 2: Check Build Logs

In Render Dashboard:
1. Go to your service
2. Click "Logs" tab
3. Look for specific error messages in build phase

**Common Error Patterns:**
```bash
# UV not found
/bin/sh: uv: command not found

# Chrome installation timeout
E: Could not get lock /var/lib/dpkg/lock-frontend

# Memory issues
Build failed due to resource constraints

# Python package conflicts
ERROR: pip's dependency resolver does not currently take into account...
```

### Step 3: Progressive Enhancement

Once basic deployment works:

1. **Add Browser Dependencies**:
   ```bash
   # Update requirements-simple.txt to include:
   playwright>=1.40.0
   ```

2. **Add Chrome Installation**:
   ```bash
   # Build Command becomes:
   apt-get update && apt-get install -y wget && pip install -r requirements-simple.txt && playwright install chromium
   ```

3. **Test Each Addition** before proceeding

### Step 4: Alternative Approaches

#### Option A: Use Docker Deployment
```yaml
# render.yaml
services:
  - type: web
    name: youtube-summarizer
    runtime: docker
    dockerfilePath: ./Dockerfile
```

#### Option B: Native Python with Staged Build
```yaml
# render.yaml - Staged approach
services:
  - type: web
    name: youtube-summarizer
    runtime: python
    buildCommand: |
      pip install --upgrade pip &&
      pip install -r requirements-simple.txt &&
      playwright install-deps &&
      playwright install chromium
```

### Step 5: Debug Commands

If you can SSH into the build environment:

```bash
# Check available space
df -h

# Check memory
free -m

# Test Chrome installation manually
google-chrome --version

# Test Python packages
python -c "import pyppeteer; print('Success')"
```

### Step 6: Fallback Strategy

If Chrome installation keeps failing:

1. **Deploy without browser dependencies**
2. **Use the fallback yt-dlp extraction** (which already works)
3. **Add browser support later** once basic app is stable

### 🎯 Quick Fix Commands

**For immediate deployment:**

1. Update `render.yaml`:
```yaml
buildCommand: "pip install -r requirements-simple.txt"
```

2. Comment out Chromium code in `api/main.py`:
```python
# Temporarily disable Chromium
CHROMIUM_AVAILABLE = False
```

3. Deploy and verify basic functionality first

### 📊 Resource Requirements

**If build keeps failing, upgrade plan:**
- Starter: 0.5 CPU, 512MB RAM - might be insufficient for Chrome
- Standard: 1 CPU, 2GB RAM - recommended minimum
- Pro: 2 CPU, 4GB RAM - ideal for browser automation

### 🔄 Recovery Steps

1. **Stop current deployment** if stuck
2. **Use simplified config** to get basic app running
3. **Add complexity incrementally**
4. **Monitor logs at each step**

This staged approach will help identify exactly where the deployment fails!