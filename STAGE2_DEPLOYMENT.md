# 🚀 Stage 2 Deployment: YouTube Processing

## Update Render Dashboard Settings

Go to: `https://dashboard.render.com/web/srv-d2a6olndiees738munfg/settings`

**Update these settings:**

### Build Command:
```bash
pip install --upgrade pip && pip install -r requirements-stage2.txt
```

### Start Command:
```bash
python -m uvicorn api.stage2_main:app --host 0.0.0.0 --port $PORT
```

### Environment Variables (add if missing):
```
GEMINI_API_KEY=your_gemini_key_here
FAL_KEY=your_fal_key_here
```

## What Stage 2 Tests:

✅ **YouTube Video Extraction** - Using yt-dlp without cookies
✅ **Subtitle Processing** - Downloads and processes subtitles  
✅ **AI Summarization** - Uses Gemini to generate summaries
✅ **Container Environment** - Tests all dependencies in Render.com

## Expected Results:

- 🟢 **Working videos**: Most public videos should process successfully
- 🟡 **Some failures**: Videos requiring authentication may fail with "Sign in to confirm you're not a bot"
- 📊 **Testing data**: Will show us which videos need browser automation

## Test URLs:

Try these public videos:
- `https://www.youtube.com/watch?v=dQw4w9WgXcQ` (Rick Roll - should work)
- `https://www.youtube.com/watch?v=kJQP7kiw5Fk` (Despacito - should work)
- Any recent video that might be restricted

## Next Steps:

1. **Deploy Stage 2** - Update dashboard settings above
2. **Test various videos** - See which ones fail with bot detection
3. **Stage 3** - Add browser automation for failed videos
4. **Final solution** - Hybrid approach with fallback

---

**After updating settings, click "Manual Deploy" and wait 2-3 minutes for Stage 2 to deploy!**