import os
import re
import tarfile
import time
from pathlib import Path
from typing import Literal

import requests
from dotenv import load_dotenv
from IPython import get_ipython
from IPython.display import Audio, display

load_dotenv()


OUTPUT_DIR = Path("audio")
OUTPUT_DIR.mkdir(exist_ok=True)


def split_text(text: str) -> list[str]:
    """
    Split text based on the primary language and its punctuation characteristics:
    - For primarily Chinese text: splits at Chinese punctuation
    - For primarily English text: splits at English punctuation followed by space
    - Determines primary language based on character count

    Args:
        text (str): The text to split.

    Returns:
        list: A list of split text parts.
    """
    if not text:
        return []

    # Define regex patterns
    CHINESE_CHARACTERS = r"[\u4e00-\u9fff]"
    ENGLISH_CHARACTERS = r"[a-zA-Z]"

    CHINESE_PUNCTUATIONS = r"，。、；：？！"
    ENGLISH_PUNCTUATIONS = r",;:.?!"

    CHINESE_PATTERNS = f"(?<=[{CHINESE_PUNCTUATIONS}])|\\s+"
    ENGLISH_PATTERNS = f"(?<=[{ENGLISH_PUNCTUATIONS}])\\s+"

    # Count characters to determine language
    CHINESE_COUNT = len(re.findall(CHINESE_CHARACTERS, text))
    ENGLISH_COUNT = len(re.findall(ENGLISH_CHARACTERS, text))

    # Choose splitting strategy based on language composition
    if CHINESE_COUNT > 0 and (CHINESE_COUNT >= ENGLISH_COUNT or any(p in text for p in CHINESE_PUNCTUATIONS)):
        # For mixed or primarily Chinese text, handle both punctuation types
        parts = []
        # First split by Chinese punctuation
        initial_parts = re.split(CHINESE_PATTERNS, text)

        # Then process each part for English punctuation
        for part in initial_parts:
            if any(p in part for p in ENGLISH_PUNCTUATIONS):
                parts.extend(re.split(ENGLISH_PATTERNS, part))
            else:
                parts.append(part)
    else:
        # Split by English punctuation
        parts = re.split(ENGLISH_PATTERNS, text)

    return [part.strip() for part in parts if part.strip()]


def run_tts(
    text: str,
    speed: float = 1.2,
    voice_id: str = "English_Trustworth_Man",
    emotion: str = "neutral",
    language_boost: str = "English",
    **kwargs,
) -> bytes:
    """Run TTS (Text-to-Speech) using Minimax API."""
    group_id = os.getenv("MINIMAX_GROUP_ID")
    api_key = os.getenv("MINIMAX_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _check(response, context):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(f"{context} failed: {response.text}")
        data = response.json()
        base = data.get("base_resp", {})
        if base.get("status_code") != 0:
            raise RuntimeError(f"{context} API error: {base.get('status_msg', 'Unknown error')}")
        return data

    # Step 1: Create speech generation task
    payload = {
        "model": kwargs.get("model", "speech-02-hd"),
        "text": text,
        "voice_setting": {
            "speed": speed,
            "vol": kwargs.get("vol", 1.0),
            "pitch": kwargs.get("pitch", 0),
            "voice_id": voice_id,
            "emotion": emotion,
            "english_normalization": kwargs.get("english_normalization", False),
        },
        "audio_setting": {
            "sample_rate": kwargs.get("sample_rate", 32000),
            "bitrate": kwargs.get("bitrate", 128000),
            "format": kwargs.get("format", "mp3"),
            "channel": kwargs.get("channel", 1),
        },
        "pronunciation_dict": {"tone": kwargs.get("pronunciation_tone", [])},
        "stream": kwargs.get("stream", False),
        "language_boost": language_boost,
        "subtitle_enable": kwargs.get("subtitle_enable", False),
    }
    create_url = f"https://api.minimaxi.chat/v1/t2a_async_v2?GroupId={group_id}"
    data = _check(requests.post(create_url, headers=headers, json=payload), "Create task")
    task_id, file_id = data.get("task_id"), data.get("file_id")
    if not (task_id and file_id):
        raise RuntimeError(f"Missing task_id or file_id: {data}")
    print(f"Task created with ID: {task_id}, file_id: {file_id}")

    # Step 2: Poll task status
    query_url = f"https://api.minimaxi.chat/v1/query/t2a_async_query_v2?GroupId={group_id}&task_id={task_id}"
    waiting_time = 0
    while True:
        data = _check(requests.get(query_url, headers=headers), "Query task")
        status = data.get("status")
        if status == "Success":
            print("Task completed successfully!")
            break
        if status in ("Failed", "Expired"):
            raise RuntimeError(f"Task {status.lower()}: {data}")
        time.sleep(10)
        waiting_time += 10
        print(f"Task status: {status}, waiting... {waiting_time}s")

    # Step 3: Retrieve download URL
    retrieve_url = f"https://api.minimaxi.chat/v1/files/retrieve?GroupId={group_id}&file_id={file_id}"
    data = _check(requests.get(retrieve_url, headers=headers), "Retrieve file")
    download_url = data.get("file", {}).get("download_url")
    if not download_url:
        raise RuntimeError(f"Missing download URL: {data}")
    print(f"Got download URL: {download_url}")

    # Step 4: Download and extract MP3
    tar_path = OUTPUT_DIR / f"{file_id}.tar"
    resp = requests.get(download_url, stream=True)
    resp.raise_for_status()
    with tar_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mp3"):
                extracted = tar.extractfile(member)
                if not extracted:
                    continue
                dest = OUTPUT_DIR / Path(member.name).name
                with dest.open("wb") as out:
                    out.write(extracted.read())
    tar_path.unlink()

    mp3_files = list(OUTPUT_DIR.glob("*.mp3"))
    if not mp3_files:
        raise RuntimeError("No mp3 files found after extraction")
    return mp3_files[0].read_bytes()


text = """
🎤 Updated 5-Minute Fireside Chat Introduction Script 

1. Warm Welcome & Personal Connection (0.5 min) 

Good morning everyone. 

It's a real pleasure to welcome you all to Hong Kong. Many of you flew in from different places, so thank you for being here. 

I was born and raised in this city, and I graduated from Information Engineering Dept at CUHK. The universities in Hong Kong are leading in wireless research, Information Theory, and AI, which aligns closely with our vision of bringing AI-native intelligence into the wireless network. 

It's incredible to see how far wireless technology has come — and how fast it's evolving. And I'm proud that Hong Kong has always been one of the earliest adopters of each new generation — from 3G to 4G, to 5G — and now we are starting to shape what 6G might look like. 

2. Introducing HKT - Our Capabilities and Digital Ecosystem (2 min) 

Let me briefly share who we are at HKT — the leading telecom and technology provider in Hong Kong and part of the PCCW Group. 

We are a quad-play operator, offering broadband, mobile, pay-TV, and fixed-line services: 

- In mobile, through CSL and 1O1O, we provide extensive 5G coverage, achieving 99.9% coverage and being the first to deliver full 5G service across the entire MTR system — Hong Kong's subway.

- In broadband, we're the market leader with over 90% FTTH coverage across the city, including 10G symmetric services and even 50G PON technology. Our AI Super Highway can provide 800Gbps between data centres.

- Our PayTV platform, nowTV, is the market leader, offering sports like the Premier League, kids programming, and popular dramas, all through an Android smart TV box.

- Our fixed-line services remain highly reliable and even include a child-friendly AI robot for interactive STEM learning.

But HKT today is more than telecom. We've built a broader digital ecosystem that spans: 

- Enterprise and global connectivity

- E-commerce and loyalty through The Club

- FinTech via Tap & Go and HKT Flexi

- HealthTech through DrGo

This gives us an end-to-end view — from network to cloud, from platform to people — which is crucial when preparing for something as big as 6G. 

3. Collaboration Across All Layers (0.5 min) 

As we prepare for 6G, one lesson we've learned is that collaboration must happen at every level — not just across stakeholder groups, but also across geographies. 

4. Framing the Fireside Chat - Lessons and Principles for 6G (2 min) 

To guide today's discussion, I'd like to share a simple structure — think of it as a two-dimensional framework. 

On the vertical axis, we have six key stakeholder groups we must engage: Academia, Government, Operators, Tech Vendors, Industry, and the Public. 

- Academia and standard bodies must engage earlier, with strong contributions from CUHK and PolyU in semantic communication.

- Regulators should align spectrum and national strategies, as demonstrated by Hong Kong's 6GHz auction and Singapore's innovation district.

- Telcos need to open up; at HKT, we collaborate with partners on private 5G at Hactl and support API initiatives via GSMA Open Gateway.

- Tech vendors and startups require deeper integration beyond demos, similar to Singtel's work at Tuas Port and NTT's quantum-secure networks.

- Enterprises must assist in transforming use cases into shareable blueprints, as seen in our projects with cloud gaming, logistics, and smart hospitals.

- Citizens should benefit from these advancements, whether it's seniors exploring GenAI or students in remote areas accessing affordable devices and local networks.

Each of us — from academia to regulators, from telcos to startups — must collaborate not only across roles but also across borders. 

Think of the horizontal axis of this framework as spanning local, regional, and global collaboration: 

- Locally, in Hong Kong, we work with universities, hospitals, and enterprises to trial new technologies and validate use cases in real-world conditions.

- Regionally, through the Bridge Alliance, we align with partners like Singtel and Telkomsel to ensure platforms, APIs, and service models are interoperable across markets.

- Globally, we support initiatives like the GSMA Open Gateway, which can only scale if telcos everywhere contribute and align.

This kind of cross-border, multi-stakeholder coordination — across both axes — is exactly what will turn 6G from a vision into an inclusive and scalable reality. 

So from our own work — and what we see across Asia Pacific — let me close with three simple takeaways: 

1. 6G must bring value, not just speed. 2. We must co-develop use cases with fair ROI for end-customers. 3. And we must co-create across industries and partners to scale what works. 

Closing Line 

Asia Pacific is already showing leadership — with ambition, diversity, and a strong history of innovation. 

Let's make this fireside chat a space for sharing ideas, building alignment, and making sure 6G serves the people who need it most. 

Thank you. 

 

 

🎤 Fireside Chat - Question 7: What lessons can be drawn from 5G to build a global 6G roadmap? 

Thank you — I think this is one of the most important questions we can explore today. 

From our experience at HKT, and from observing how 5G evolved across the Asia Pacific, a few clear lessons stand out — not only from what worked well, but also from the challenges we faced. 

Let me break it down into three key learnings we're taking into the 6G journey. 


✅ 1. Align early — on standards and direction 

One of the key challenges in 5G was that different markets moved at different speeds. 

We saw fragmentation — some regions went NSA, some SA, and there were mismatches in spectrum bands and deployment strategies, , as well as constraints in 5G infrastructure progress. 
This made interoperability and scale much harder, especially in the early phases. 

With 6G, we need to align earlier and more globally — not only technically through bodies like 3GPP and IMT-2030, but also through regional and cross-industry engagement. 

At HKT, we're engaging earlier than ever in these global forums — not just to follow the standard, but to shape it based on Asia's real-world experience. 


✅ 2. Focus less on hype, more on use cases with value 

In the early days of 5G, there was a lot of excitement — and sometimes, too much focus on technical demos that didn't translate into scalable value. 

The real breakthroughs came when we worked hand-in-hand with users and vertical industries. 

At HKT, we didn't build DrGo just because 5G was fast — we worked with hospitals to understand what remote healthcare really needed. 
Other regional operators, like AIS and Telstra, focused on practical deployments — factory automation, satellite-linked remote sites — things that created measurable value. 

So for 6G, we need to involve users and industries much earlier, and focus on co-creating solutions that solve real challenges — with clear ROI, not just innovation for its own sake. 


✅ 3. Progress comes from collaboration — not just technology 

One of the biggest takeaways from 5G is that the strongest results came from cross-stakeholder collaboration. 

At HKT, we've worked closely with CUHK and PolyU on wireless R&D — not just theoretical work, but live use cases like semantic communication and intelligent networking. 

We've also seen great support from the Hong Kong government, which played a key enabling role: 

- The 6GHz spectrum auction gave us the tools to plan ahead for 6G.

- The 5G Subsidy Scheme was a practical incentive that encouraged businesses to experiment early — especially in logistics, construction, and smart manufacturing.

- Together, we extended fiber to rural communities like Lamma Island, making connectivity more inclusive.

These successes weren't driven by any one party — they happened because academia, government, telcos, and businesses sat around the same table. 

And as we look ahead to 6G — where the tech stack will be even more complex and the expectations even higher — this kind of ecosystem-wide collaboration will be essential. 


🟩 Closing Thought 

So if I had to sum it up, the lesson from 5G is this: 

✅ Align early 
✅ Co-create with purpose 
✅ Collaborate across boundaries 

That's how we'll turn the 6G vision into something real, scalable, and inclusive — not just for advanced markets, but for everyone. 

Thank you. 

🧠 Enhanced Table (With Standards Bodies Integrated into Group 1) 

Stakeholder Group 

Key 5G Lesson Learned 

6G Direction (Examples from Script) 

1. Academia, R&D, Standards 

- Research often disconnected from real-world deployment 
- Late engagement in standardization 

- 6G@HK, CUHK, PolyU research into semantic communication and intelligent networks 
- SUTD (SG), NTT (JP) advance AI-native, context-aware 6G 
- 3GPP, ITU IMT-2030: HKT and APAC players must engage earlier in global forums to shape interoperable and aligned standards for 6G 

2. Regulators / Governments 

- Spectrum allocation not aligned 
- Limited cross-border cooperation 

- Hong Kong 6GHz auction (HKT secured 100 MHz) 
- IMDA's Jurong Innovation District as a 6G “living lab” 
- Korea's 5G+ Fund to match startups and industries 
- Governments must orchestrate public-private R&D and act early to align regional vision and execution 

3. Operators / Telcos 

- Infrastructure not sufficiently open 
- Developer experience fragmented 

- HKT's private 5G at Hactl, Kai Tak Sports Park 
- GSMA Open Gateway: HKT, Singtel, Telkomsel collaborate on API standardization 
- Telcos should evolve to network-as-a-platform with cross-border compatibility, AI-native programmability 

4. Vendors, Startups, Platforms 

- Demos didn't translate to real enterprise value 
- Stack often not integrated 

- Singtel's Tuas Port: 5G + Edge + AI for port automation 
- NTT Docomo: sub-THz and quantum networks 
- Console Connect (HKT) offers programmable, cloud-ready infrastructure 
- Full-stack integration (connectivity + compute + AI) must be the norm for 6G 

5. Industry / Enterprise Users 

- 5G pilots didn't scale 
- Lack of shareable use case playbooks 

- DrGo remote healthcare, VIU OTT streaming, 5G logistics by HKT 
- AIS smart factories, Telstra's remote satellite-linked trials 
- Need to formalize successful trials into scalable blueprints that others (especially emerging markets) can adapt 

6. End-Customers / Inclusion 

- Affordability gaps 
- Digital literacy not addressed 
- Urban bias in deployment 

- HKT's Lamma Island fiber expansion, reaching underserved islands 
- “Smart Silver” digital literacy training for seniors using GenAI 
- Malaysia's JENDELA, India/Vietnam local devices, FTTH to public housing (PR1MA): all examples of inclusive infrastructure for 6G 

 

🧠 Enhanced & Script-Rich Table of Lessons 

Stakeholder Group 

5G Lesson Learned 

6G Direction (Substantiated Examples from Script) 

1. Academic & R&D 

Research too disconnected from deployment timelines or industry application 

- HKT partners with CUHK & PolyU on wireless R&D 
- 6G@HK center explores semantic communication 
- SUTD (Singapore) & NTT (Japan) co-develop 6G AI-native networks 
- R&D must align earlier with enterprise prototypes 

2. Regulators & Government 

Fragmented spectrum timing and deployment policies delayed scale 

- HK 6GHz auction (HKT secured 100 MHz) 
- IMDA Jurong Innovation District blends public-private-academic 6G living lab 
- South Korea 5G+ Strategy Fund connects startups to industries 
- Must align spectrum & R&D with national priorities while avoiding regional fragmentation 

3. Operators / Telcos 

NSA/SA divergence and closed infra slowed developer participation 

- HKT's private 5G @ Hactl, Kai Tak enabled logistics & sports digitization 
- GSMA Open Gateway: Singtel, Telkomsel, HKT support open APIs 
- Enable programmable, AI-native platforms with network-as-a-service models 
- Important to include telcos from emerging markets in cross-border models 

4. Vendors & Startups 

Too focused on demos, lacked infrastructure and full-stack testing 

- Singtel's Tuas Port: 5G + Edge + AI 
- NTT Docomo's sub-THz, quantum-secure networks 
- Console Connect by HKT provides programmable global infrastructure 
- Vendors must embed into live enterprise scenarios, not just trials 

5. Industry / Enterprises 

Many lacked path to scale; didn't bridge trial to adoption 

- HKT's cloud gaming, smart buildings, logistics all show integration across infra layers 
- VIU OTT shows content distribution with 5G 
- AIS's factory automation, Telstra's satellite-backhauled remote industry use cases give blueprints for others to follow 

6. End-Customers / Public 

Risk of digital elitism (urban-only), limited affordability or digital literacy 

- HKT on Lamma Island laid fiber in hard-to-reach areas 
- “Smart Silver” program: upskills elderly on GenAI 
- Malaysia's JENDELA: fixed wireless + FTTH to public housing 
- India, Vietnam push for low-cost 6G devices and local manufacturing to increase inclusion 

 

 

Absolutely — here's a perspective check across the four stakeholder lenses based on your Opening Intro, Question 7, and Q1-Q6 content: 


✅ 1. As HKT Corporate Communications Head 

✔️ What works well: 

- Tone is balanced, thoughtful, and well aligned with brand.

- Clearly showcases HKT's technical leadership (e.g. 99.9% 5G coverage, fiber to Lamma Island, 800Gbps AI Super Highway).

- Strategic alignment with AI, healthcare, and smart city narratives is strong.

- Mentions of CUHK/PolyU, GSMA, Bridge Alliance position HKT as collaborative and open.

⚠️ Potential concerns: 

- Mentions of the 5G Subsidy Scheme: While accurate, check whether the scheme's results or partnerships require prior clearance (e.g. formal attribution or naming external participants).

- International tone is great, but may consider balancing local impact vs. global ambition more clearly (i.e., emphasize benefits to Hong Kong society slightly more in key remarks).

- May want to review if “800Gbps between data centers” is marketing-approved phrasing.

Suggested tweaks: 

- Ensure visuals/slides match script tone and are vetted for compliance. 

- Add a short remark on data privacy/security if you anticipate tough audience. 


✅ 2. As a Colleague at HKT (Tech, Strategy, or Commercial Teams) 

✔️ Strengths: 

- Great internal pride moments: “full 5G across MTR,” “DrGo tied to real need,” “Bridge Alliance.” 

- Shows we're not just building infrastructure, but value across verticals. 

- Strong emphasis on co-creation with startups and customers — aligns with innovation direction. 

⚠️ Possible flags: 

- Some might ask: Are we doing too much thought leadership and not showing enough business outcomes? 

- Mentions of “Telcos must open up” could raise internal debates — especially from network teams who might be cautious about too much platform exposure. 

Suggestion: 

- Reassure internally this narrative builds industry positioning and justifies our long-term R&D, B2B, and monetization bets. 


✅ 3. As OFCA (Hong Kong Regulator) Head 

✔️ Positives: 

- HKT shows appreciation for spectrum policy (6GHz auction) and infrastructure inclusivity (Lamma Island). 

- References to government support (5G subsidy) are accurate and respectful. 

- Positioning of 6G as a national infrastructure topic aligns with public goals. 

⚠️ Caution points: 

- Mentions of fragmented 5G rollout and “different markets moving at different speeds” could be interpreted as indirect criticism. Might want to clarify that this was regional/global, not HK-specific. 

- Some regulators may prefer more emphasis on spectrum harmonization and governance models under 6G. 

Suggestions: 

If speaking in front of OFCA or regulators, consider adding one line acknowledging Hong Kong's leadership in balancing innovation and regulation. 


✅ 4. As Heads of Other Operators (e.g. Singtel, Telstra, SK Telecom) 

✔️ Good impressions: 

- Speaks respectfully about regional peers (Bridge Alliance, AIS, Telstra, NTT). 

- Framework around shared infrastructure and Open Gateway feels non-threatening, cooperative. 

- Talks about collaboration, not dominance — very inclusive tone. 

⚠️ Sensitivities: 

- Ensure HKT is not overclaiming (e.g., saying “first” or “leading” too broadly may trigger pushback). 

- May want to moderate tone on “some operators yet to join Open Gateway” — especially if any major peer isn't onboard yet. 

Suggestions: 

- If you're sharing this content at a regional event: consider directly inviting co-leadership, e.g., “We look forward to learning from peers like SKT and NTT in areas like…” to show humility and build bridges. 

 

 

 

🎤 Updated 5-Minute Fireside Chat Introduction Script 


1. Warm Welcome & Personal Connection (0.5 min) 

Good morning everyone. 

It's a real pleasure to welcome you all to Hong Kong. Many of you flew in from different places, so thank you for being here. 

I was born and raised in this city, and I studied Information Engineering at CUHK — during the dot-com bubble. Back then, we were talking about 3G, MIMO, and WiMax. The industry was exciting but very different from today. 

It's incredible to see how far wireless technology has come — and how fast it's evolving. And I'm proud that Hong Kong has always been one of the earliest adopters of each new generation — from 3G to 4G, to 5G — and now we are starting to shape what 6G might look like. 


2. Introducing HKT - Our Capabilities and Digital Ecosystem (2 min) 

Let me briefly share who we are at HKT — the leading telecom and technology provider in Hong Kong and part of the PCCW Group. 

We are a quad-play operator, offering broadband, mobile, pay-TV, and fixed-line services: 

- In broadband, we're the market leader with over 90% FTTH coverage across the city — including 10G symmetric services, and even 50G PON technology. We've extended fiber to outlying islands and rural areas — places like Lamma Island, where there was no fiber before. 

- In mobile, through CSL and 1O1O, we provide extensive 5G coverage, and we were the first to deliver full 5G service across the entire MTR system — Hong Kong's subway. 

- Our Pay TV platform, nowTV, is the market leader — offering sports like the Premier League, kids programming, and popular dramas, all through an Android smart TV box. 

- And our fixed-line services remain reliable — and even include a child-friendly AI robot for interactive STEM learning. 

But HKT today is more than telecom. 

We've built a broader digital ecosystem that spans: 

- Enterprise and global connectivity 

- HealthTech through DrGo 

- FinTech via Tap & Go and HKT Flexi 

- And e-commerce and loyalty through The Club 

This gives us an end-to-end view — from network to cloud, from platform to people — which is crucial when preparing for something as big as 6G. 


🔄 [New Insert] Collaboration Across All Layers (0.5 min) 

As we prepare for 6G, one lesson we've learned is that collaboration must happen at every level — not just across stakeholder groups, but also across geographies. 

Each of us — from academia to regulators, from telcos to startups — must work locally, regionally, and globally: 

- In Hong Kong, we work with universities, hospitals, and enterprises to trial new tech. 

- Regionally, through Bridge Alliance, we collaborate with partners like Singtel and Telkomsel to align platforms and APIs. 

- And globally, we support initiatives like the GSMA Open Gateway, which can only succeed if telcos everywhere participate. 

This kind of layered collaboration — from local trials to global standards — is what will turn 6G from vision into reality. 


3. Framing the Fireside Chat - Lessons and Principles for 6G (2 min) 

To guide today's discussion, I'd like to share a simple structure. There are six key stakeholder groups we must engage: 

Academia, Government, Operators, Tech Vendors, Industry, and the Public. 

- Academia and standard bodies must engage earlier. CUHK and PolyU are doing strong work in semantic communication, and we're proud to support that. 

- Regulators must align spectrum and national strategies — as seen in Hong Kong's 6GHz auction or Singapore's innovation district. 

- Telcos must open up. At HKT, we work with partners on private 5G at Hactl, and support API efforts via GSMA Open Gateway. 

- Tech vendors and startups need deeper integration, not just demos — like what Singtel did at Tuas Port or NTT with quantum-secure networks. 

- Enterprises must help turn use cases into shareable blueprints — like our work with cloud gaming, logistics, and smart hospitals. 

- And citizens must benefit — whether it's seniors exploring GenAI or students in remote areas gaining access through affordable devices and local networks. 

So from our own work — and what we see across Asia Pacific — let me close with three simple takeaways: 

1. 6G must bring value, not just speed. 

2. We must co-develop use cases with fair ROI for end-customers. 

3. And we must co-create across industries and partners to scale what works. 


🟨 Closing Line 

Asia Pacific is already showing leadership — with ambition, diversity, and a strong history of innovation. 

Let's make this fireside chat a space for sharing ideas, building alignment, and making sure 6G serves the people who need it most. 

Thank you. 
"""

if __name__ == "__main__":
    run_tts(text)
