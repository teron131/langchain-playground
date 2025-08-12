# 🎯 Render Dashboard Update Guide

## Step-by-Step Instructions to Deploy Optimized YouTube API

### **Step 1: Access Your Render Dashboard**

1. Go to [https://dashboard.render.com](https://dashboard.render.com)
2. Log in to your account
3. Find your existing service (the one you've been using for testing)

### **Step 2: Update Build & Deploy Settings**

1. Click on your service name to open the service details
2. Click the **"Settings"** tab (usually at the top)
3. Scroll down to the **"Build & Deploy"** section

**Update these fields:**

**Build Command:**
```
pip install -r requirements-youtube.txt
```

**Start Command:**
```
python api/youtube_main.py
```

### **Step 3: Environment Variables**

1. Still in the Settings tab, scroll to **"Environment Variables"** section
2. Keep all your existing variables and ensure these are set:

```
GOOGLE_API_KEY=your_actual_google_api_key
OPENROUTER_API_KEY=your_actual_openrouter_api_key
FAL_KEY=your_actual_fal_api_key
PORT=$PORT
```

*Note: Replace "your_actual_*" with your real API keys*

### **Step 4: Deploy the Changes**

1. Click **"Save Changes"** at the bottom of the settings page
2. Go back to the **"Events"** or **"Logs"** tab
3. Click **"Manual Deploy"** button (usually near the top)
4. Select **"Deploy latest commit"**

### **Step 5: Monitor Deployment**

Watch the build logs. You should see:

```
==> Building service
==> Installing dependencies from requirements-youtube.txt
==> Starting application with: python api/youtube_main.py
==> Service is live at https://your-service.onrender.com
```

### **Step 6: Verify Deployment**

Test the health endpoint:
```
https://your-service.onrender.com/health
```

Should return:
```json
{"status": "healthy", "timestamp": "2025-08-12T03:04:00Z"}
```

### **🚨 Common Issues:**

1. **If build fails**: Check that `requirements-youtube.txt` exists in your repo
2. **If start fails**: Check that `api/youtube_main.py` exists
3. **If environment variables missing**: Double-check all API keys are set

### **✅ Success Indicators:**

- ✅ Build completes without errors
- ✅ Service shows as "Live" 
- ✅ Health endpoint responds with 200 OK
- ✅ Logs show "Uvicorn running on http://0.0.0.0:8000"

**Follow these steps and let me know when the deployment is complete!**