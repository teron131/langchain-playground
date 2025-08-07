"""
YouTube Video Summarization Example

This example demonstrates how to:
1. Load a YouTube video transcript using the YouTubeLoader
2. Send the transcript to Gemini API for summarization
3. Get a concise summary of the video content

Usage:
    python yt_summary.py
"""

import os

from dotenv import load_dotenv
from google.genai import Client, types

from langchain_playground.Tools.YouTubeLoader.youtube import youtube_loader

load_dotenv()


def summarize_with_gemini(transcript: str, summary_type: str = "detailed") -> str:
    """Summarize transcript using Gemini API.

    Args:
        transcript (str): The full transcript to summarize
        summary_type (str): Type of summary - "brief", "detailed", or "bullet_points"

    Returns:
        str: The generated summary
    """
    client = Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Different prompt templates for different summary types
    prompts = {
        "brief": """Provide a brief 2-3 sentence summary of this video transcript:

{transcript}

Summary:""",
        "detailed": """Provide a detailed summary of this video transcript, including:
1. Main topic and purpose
2. Key points discussed
3. Important conclusions or takeaways
4. Any notable examples or case studies mentioned

Transcript:
{transcript}

Detailed Summary:""",
        "bullet_points": """Create a bullet-point summary of this video transcript:
- Main topic:
- Key points: (list the main points discussed)
- Conclusions: (main takeaways)
- Action items: (if any recommendations are given)

Transcript:
{transcript}

Bullet Summary:""",
    }

    prompt = prompts.get(summary_type, prompts["detailed"]).format(transcript=transcript)

    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1000,
        ),
    )

    return response.text


def main():
    """Main example function."""
    # Example YouTube URL - replace with any video you want to summarize
    url = "https://youtu.be/6Nn4MJYmv4A?si=mA3Q14UpYEzvv0sC"

    print("🎬 YouTube Video Summarization Example")
    print("=" * 50)

    try:
        # Step 1: Load the YouTube video transcript
        print("📥 Loading YouTube video transcript...")
        transcript = youtube_loader(url)
        print("✅ Transcript loaded successfully")

        # Step 2: Generate different types of summaries
        print("\n🤖 Generating summaries with Gemini...")

        # Brief summary
        print("\n📋 BRIEF SUMMARY:")
        brief_summary = summarize_with_gemini(transcript, "brief")
        print(brief_summary)

        # Detailed summary
        print("\n📝 DETAILED SUMMARY:")
        detailed_summary = summarize_with_gemini(transcript, "detailed")
        print(detailed_summary)

        # Bullet points summary
        print("\n📌 BULLET POINTS SUMMARY:")
        bullet_summary = summarize_with_gemini(transcript, "bullet_points")
        print(bullet_summary)

        print("\n✅ All summaries generated successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return


if __name__ == "__main__":
    main()
