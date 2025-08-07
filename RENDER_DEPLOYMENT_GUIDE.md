# 🚀 Render.com Deployment Guide for YouTube Summarizer

## Overview
This guide will help you deploy the YouTube Summarizer to Render.com with persistent container support and full Chromium capabilities to solve YouTube's bot detection issues.

## 📋 Prerequisites

1. **Render.com Account**: Sign up at [render.com](https://render.com)
2. **GitHub Repository**: Your code should be pushed to GitHub
3. **Environment Variables**: Have your API keys ready

## 🔧 Deployment Steps

### Step 1: Create New Web Service

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select your repository containing this code

### Step 2: Configure Service Settings

**Basic Settings:**
- **Name**: `youtube-summarizer` (or your preferred name)
- **Region**: Choose closest to your users (e.g., `Oregon`)
- **Branch**: `main` (or your default branch)

**Build & Deploy:**
- **Runtime**: `Docker`
- **Build Command**: Leave empty (uses Dockerfile)
- **Start Command**: Leave empty (uses Dockerfile CMD)

### Step 3: Environment Variables

Add these environment variables in Render dashboard:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here
FAL_KEY=your_fal_key_here

# Optional
PYTHONPATH=/app
PYTHONUNBUFFERED=1
```

### Step 4: Advanced Settings

- **Auto-Deploy**: ✅ Enable
- **Health Check Path**: `/health`
- **Plan**: Start with **"Starter"** ($7/month) - can upgrade if needed

### Step 5: Deploy

1. Click **"Create Web Service"**
2. Wait for build process (5-10 minutes for first deploy)
3. Monitor build logs for any issues

## 🔍 Expected Build Process

```bash
[Build] Installing system dependencies for Chromium...
[Build] Installing Google Chrome...
[Build] Installing Python dependencies...
[Build] Installing Playwright browsers...
[Build] Build completed successfully
[Deploy] Container started successfully
[Deploy] Health check passed ✅
```

## 📊 Resource Requirements

### Minimum Plan Recommendations:
- **Starter Plan** ($7/month): 0.5 CPU, 512MB RAM - Good for testing
- **Standard Plan** ($25/month): 1 CPU, 2GB RAM - Recommended for production
- **Pro Plan** ($85/month): 2 CPU, 4GB RAM - Best performance

### Why Container Environment Solves YouTube Issues:

| Issue | Vercel Serverless | Render Container |
|-------|------------------|------------------|
| **IP Detection** | ❌ Lambda IPs flagged | ✅ Stable container IPs |
| **Session Persistence** | ❌ Cold starts | ✅ Persistent sessions |
| **Browser Resources** | ❌ Limited memory | ✅ Full resources |
| **Execution Time** | ❌ 15min limit | ✅ Unlimited |
| **Authentication** | ❌ No cookie persistence | ✅ Profile storage |

## 🧪 Testing After Deployment

### 1. Health Check
Visit: `https://your-app-name.onrender.com/health`

Expected response:
```json
{"status": "healthy", "message": "YouTube Summarizer API is running"}
```

### 2. Test Chromium Extraction
Visit: `https://your-app-name.onrender.com`

Try extracting a YouTube video like:
`https://www.youtube.com/watch?v=dQw4w9WgXcQ`

### 3. Monitor Logs

In Render dashboard:
1. Go to your service
2. Click **"Logs"** tab
3. Look for successful browser launch:

```
✅ Found Chrome at: /usr/bin/google-chrome-stable
🚀 Starting container-optimized Chromium extraction...
✅ Browser instance created with persistent profile
✅ Successfully navigated to YouTube
🍪 Successfully extracted 45 total cookies
✅ Filtered to 12 YouTube-relevant cookies
```

## 🔧 Troubleshooting

### Build Fails
- Check if all files are committed to GitHub
- Verify Dockerfile syntax
- Check build logs for specific error

### Chrome Installation Issues
```bash
# If you see Chrome installation errors, try the alternative Dockerfile:
# Replace google-chrome-stable installation with chromium-browser
```

### Memory Issues  
- Upgrade to Standard plan ($25/month)
- Monitor memory usage in Render dashboard

### Cookie Extraction Still Fails
- Check container logs for browser launch errors
- Verify YouTube is accessible from your container
- Consider enabling debug mode

## 🎯 Success Indicators

You'll know it's working when:

1. ✅ **Build completes** without Chrome installation errors
2. ✅ **Health endpoint** responds correctly  
3. ✅ **Browser logs** show successful Chrome launch
4. ✅ **Cookie extraction** finds 10+ YouTube cookies
5. ✅ **Video processing** works without "bot detection" errors

## 📈 Performance Optimization

Once working:

1. **Monitor Usage**: Check CPU/memory in Render dashboard
2. **Upgrade Plan**: If needed for better performance
3. **Enable CDN**: For faster global access
4. **Set up Monitoring**: Health checks and alerting

## 🚨 Important Notes

- **First deployment** takes 5-10 minutes due to Chrome installation
- **Subsequent deployments** are faster (2-3 minutes)
- **Cold starts** don't affect containers like serverless functions
- **Browser sessions persist** between requests (major advantage!)

## 💰 Cost Estimate

- **Starter Plan**: $7/month - Good for testing and light usage
- **Standard Plan**: $25/month - Recommended for regular usage  
- **Pro Plan**: $85/month - High volume or performance needs

Much more cost-effective than paying for Browser-as-a-Service alternatives!

---

**🎉 That's it! Your YouTube Summarizer should now work reliably on Render.com with proper YouTube authentication and no more bot detection issues.**