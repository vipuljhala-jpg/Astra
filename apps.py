"""
🔵 Astra — Build-Your-Own-Alexa (single-file Streamlit demo)

Flow:  🎙 mic → Sarvam STT → Claude (tool-calling) → Sarvam TTS → 🔊

Run:
    pip install -r requirements.txt
    streamlit run app.py

Paste your Anthropic + Sarvam API keys in the sidebar — nothing else needed.
Weather (Open-Meteo) and music (iTunes previews) are free, no keys required.

Latency notes (already applied):
  - claude-haiku (fastest Claude) with a tight system prompt + low max_tokens
  - one shared requests.Session (connection reuse ≈ 100-300 ms saved per call)
  - Anthropic client created once and cached, not per request
  - single-round tool use in most turns; hard cap of 3 rounds
"""

import base64
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic
import requests
import streamlit as st

# =============================================================================
# Page setup + styling
# =============================================================================
st.set_page_config(page_title="Astra Voice Assistant", page_icon="🔵", layout="centered")

st.markdown(
    """
    <style>
      .astra-ring {
        width: 84px; height: 84px; margin: 0 auto 6px auto; border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, #1b3a6b, #050d1a 70%);
        border: 3px solid #19c2ff; box-shadow: 0 0 24px #19c2ff55;
        display: flex; align-items: center; justify-content: center; font-size: 34px;
      }
      .astra-title { text-align: center; font-size: 1.6rem; font-weight: 700; margin-bottom: 0; }
      .astra-sub { text-align: center; color: #7a8aa0; margin-top: 2px; }
    </style>
    <div class="astra-ring">🎙️</div>
    <p class="astra-title">Astra</p>
    <p class="astra-sub">Your own Alexa — STT → Claude tools → TTS</p>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Shared HTTP session (connection reuse = lower latency)
# =============================================================================
@st.cache_resource
def http() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "astra-demo/1.0"})
    return s


@st.cache_resource
def claude_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


# =============================================================================
# Skills (Alexa's "intents") — weather, time, music
# =============================================================================
WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city. Use for any weather/temperature/rain question.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. 'Bengaluru'"}},
            "required": ["city"],
        },
    },
    {
        "name": "get_time",
        "description": "Get current date and time. Use when asked the time, date, or day.",
        "input_schema": {
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "IANA tz, default Asia/Kolkata"}},
            "required": [],
        },
    },
    {
        "name": "play_music",
        "description": "Search a song and play a preview. Use when asked to play music/a song/an artist.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Song and/or artist name"}},
            "required": ["query"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another. Use when asked to convert money or exchange rates.",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "Source currency code, e.g. USD"},
                "to_currency": {"type": "string", "description": "Target currency code, e.g. INR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
    {
        "name": "get_joke",
        "description": "Fetch a random joke. Use when asked for a joke or to make the user laugh.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_wikipedia",
        "description": "Get a short Wikipedia summary for a person, place, or concept. Use for factual 'who is' or 'what is' questions.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Topic to look up"}},
            "required": ["query"],
        },
    },
    {
        "name": "flip_coin_or_roll_dice",
        "description": "Flip a coin or roll a dice. Use when asked to flip a coin or roll a die/dice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["flip_coin", "roll_dice"], "description": "flip_coin or roll_dice"},
                "sides": {"type": "integer", "description": "Number of sides on the dice, default 6"},
            },
            "required": ["action"],
        },
    },
]


def get_weather(city: str) -> dict:
    try:
        geo = http().get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=8,
        ).json()
        if not geo.get("results"):
            return {"error": f"Could not find city '{city}'."}
        loc = geo["results"][0]
        cur = http().get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"], "longitude": loc["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
            },
            timeout=8,
        ).json()["current"]
        return {
            "city": loc["name"], "country": loc.get("country", ""),
            "temperature_c": cur["temperature_2m"], "feels_like_c": cur["apparent_temperature"],
            "humidity_pct": cur["relative_humidity_2m"], "wind_kmh": cur["wind_speed_10m"],
            "condition": WMO_CODES.get(cur["weather_code"], "unknown"),
        }
    except Exception as exc:
        return {"error": f"Weather lookup failed: {exc}"}


def get_time(timezone: str = "Asia/Kolkata") -> dict:
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        timezone = "Asia/Kolkata"
        now = datetime.now(ZoneInfo(timezone))
    return {"timezone": timezone, "time": now.strftime("%I:%M %p"), "date": now.strftime("%A, %d %B %Y")}


def play_music(query: str) -> dict:
    try:
        res = http().get(
            "https://itunes.apple.com/search",
            params={"term": query, "media": "music", "limit": 1}, timeout=8,
        ).json()
        if not res.get("results"):
            return {"error": f"No song found for '{query}'."}
        t = res["results"][0]
        return {
            "track": t.get("trackName"), "artist": t.get("artistName"),
            "album": t.get("collectionName"), "preview_url": t.get("previewUrl"),
            "artwork_url": t.get("artworkUrl100"),
            "note": "A 30-second preview starts playing for the user now.",
        }
    except Exception as exc:
        return {"error": f"Music search failed: {exc}"}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    try:
        resp = http().get(
            f"https://open.er-api.com/v6/latest/{from_currency.upper()}",
            timeout=8,
        ).json()
        if resp.get("result") != "success":
            return {"error": f"Could not fetch rates for {from_currency}."}
        rate = resp["rates"].get(to_currency.upper())
        if not rate:
            return {"error": f"Unknown currency {to_currency}."}
        converted = round(amount * rate, 2)
        return {"from": f"{amount} {from_currency.upper()}", "to": f"{converted} {to_currency.upper()}", "rate": rate}
    except Exception as exc:
        return {"error": f"Currency conversion failed: {exc}"}


def get_joke() -> dict:
    try:
        data = http().get(
            "https://official-joke-api.appspot.com/random_joke", timeout=8
        ).json()
        return {"setup": data["setup"], "punchline": data["punchline"]}
    except Exception as exc:
        return {"error": f"Joke fetch failed: {exc}"}


def search_wikipedia(query: str) -> dict:
    try:
        resp = http().get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_"),
            timeout=8,
        ).json()
        if resp.get("type") == "disambiguation" or "extract" not in resp:
            return {"error": f"No clear Wikipedia article for '{query}'."}
        return {"title": resp["title"], "summary": resp["extract"][:400], "url": resp.get("content_urls", {}).get("desktop", {}).get("page", "")}
    except Exception as exc:
        return {"error": f"Wikipedia lookup failed: {exc}"}


def flip_coin_or_roll_dice(action: str, sides: int = 6) -> dict:
    import random
    if action == "flip_coin":
        return {"result": random.choice(["Heads", "Tails"])}
    return {"result": random.randint(1, max(2, sides)), "sides": sides}


def execute_tool(name: str, args: dict) -> dict:
    fn = {
        "get_weather": get_weather, "get_time": get_time, "play_music": play_music,
        "convert_currency": convert_currency, "get_joke": get_joke,
        "search_wikipedia": search_wikipedia, "flip_coin_or_roll_dice": flip_coin_or_roll_dice,
    }.get(name)
    return fn(**args) if fn else {"error": f"Unknown tool {name}"}


# =============================================================================
# Brain — Claude tool-calling loop (replaces Alexa's NLU/intent router)
# =============================================================================
SYSTEM_PROMPT = (
    "You are Astra, a friendly voice assistant like Alexa. Your replies are spoken "
    "aloud by TTS: keep them to 1-2 short conversational sentences, no markdown, no "
    "emojis. Use tools for weather, time, music, currency, jokes, Wikipedia, and coin/dice — "
    "never guess live data. For jokes say the setup then pause then the punchline. "
    "After starting music just confirm briefly. If the user speaks Hindi or Kannada, reply "
    "in the same language. Answer general knowledge directly."
)


def run_assistant(messages: list, api_key: str) -> tuple[str, list]:
    client = claude_client(api_key)
    convo = list(messages)
    tool_events = []
    for _ in range(3):  # hard cap on tool rounds — keeps worst-case latency bounded
        resp = client.messages.create(
            model="claude-haiku-4-5", max_tokens=300,
            system=SYSTEM_PROMPT, tools=TOOLS, messages=convo,
        )
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text").strip(), tool_events
        convo.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, dict(block.input))
                tool_events.append({"name": block.name, "input": dict(block.input), "result": result})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        convo.append({"role": "user", "content": results})
    return "Sorry, I got stuck on that one. Try again?", tool_events


# =============================================================================
# Speech — Sarvam STT (Saarika) + TTS (Bulbul)
# =============================================================================
def speech_to_text(audio_bytes: bytes, sarvam_key: str) -> str:
    resp = http().post(
        "https://api.sarvam.ai/speech-to-text",
        headers={"api-subscription-key": sarvam_key},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": "saarika:v2.5", "language_code": "unknown"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("transcript", "").strip()


def detect_tts_language(text: str) -> str:
    for ch in text:
        if "\u0900" <= ch <= "\u097F":
            return "hi-IN"
        if "\u0C80" <= ch <= "\u0CFF":
            return "kn-IN"
    return "en-IN"


def text_to_speech(text: str, sarvam_key: str) -> bytes:
    resp = http().post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": sarvam_key, "Content-Type": "application/json"},
        json={
            "text": text[:1500],
            "target_language_code": detect_tts_language(text),
            "speaker": "anushka",
            "model": "bulbul:v2",
        },
        timeout=20,
    )
    resp.raise_for_status()
    audios = resp.json().get("audios", [])
    if not audios:
        raise RuntimeError("Sarvam TTS returned no audio")
    return base64.b64decode(audios[0])


# =============================================================================
# Secrets — loaded from Streamlit secrets manager
# =============================================================================
anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
sarvam_key = st.secrets["SARVAM_API_KEY"]

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    voice_on = st.toggle("🔊 Voice replies (TTS)", value=True)

    st.divider()
    st.markdown("**🚀 Capabilities**")
    with st.expander("🌦️ Weather & Time"):
        st.markdown(
            "- *Is it going to rain in Mumbai tomorrow?*\n"
            "- *Compare weather: Delhi vs Bengaluru*\n"
            "- *What time is it in New York right now?*"
        )
    with st.expander("💱 Finance"):
        st.markdown(
            "- *Convert 10,000 JPY to EUR*\n"
            "- *What's the GBP to INR exchange rate?*\n"
            "- *How much is 500 AED in USD?*"
        )
    with st.expander("🎵 Music"):
        st.markdown(
            "- *Play Blinding Lights by The Weeknd*\n"
            "- *Play something by AR Rahman*\n"
            "- *Play Kesariya from Brahmastra*"
        )
    with st.expander("🧠 Knowledge"):
        st.markdown(
            "- *Explain quantum entanglement simply*\n"
            "- *What caused the 2008 financial crisis?*\n"
            "- *Who is Sundar Pichai?*\n"
            "- *What is the James Webb Space Telescope?*"
        )
    with st.expander("🎲 Fun & Utilities"):
        st.markdown(
            "- *Roll a 20-sided dice*\n"
            "- *Flip a coin to decide something*\n"
            "- *Tell me a dark humour joke*"
        )

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear"):
        for k in ("history", "display", "last_audio_hash"):
            st.session_state.pop(k, None)
        st.rerun()
    if col2.button("💾 Export"):
        lines = []
        for item in st.session_state.get("display", []):
            ts = item.get("ts", "")
            lines.append(f"[{ts}] {item['role'].upper()}: {item['content']}")
        st.download_button("⬇️ Download", "\n".join(lines), file_name="astra_chat.txt", mime="text/plain")

# =============================================================================
# Session state
# =============================================================================
ss = st.session_state
ss.setdefault("history", [])          # text-only history for Claude
ss.setdefault("display", [])          # rich log for the UI
ss.setdefault("last_audio_hash", "")
ss.setdefault("status", "idle")       # idle | listening | thinking | speaking

# =============================================================================
# Status badge
# =============================================================================
_status_colors = {"idle": "#7a8aa0", "listening": "#19c2ff", "thinking": "#f5a623", "speaking": "#4caf50"}
_status_icons  = {"idle": "💤", "listening": "👂", "thinking": "🧠", "speaking": "🗣️"}
_s = ss.get("status", "idle")
st.markdown(
    f'<p style="text-align:center;color:{_status_colors[_s]};font-size:0.85rem">'
    f'{_status_icons[_s]} {_s.capitalize()}</p>',
    unsafe_allow_html=True,
)

# =============================================================================
# Render conversation
# =============================================================================
for item in ss.display:
    with st.chat_message(item["role"], avatar="🧑" if item["role"] == "user" else "🔵"):
        st.write(item["content"])
        if item.get("ts"):
            st.caption(item["ts"])
        for ev in item.get("tools", []):
            r = ev["result"]
            if ev["name"] == "play_music" and r.get("preview_url"):
                cols = st.columns([1, 5])
                if r.get("artwork_url"):
                    cols[0].image(r["artwork_url"], width=72)
                cols[1].markdown(f"**{r.get('track')}** — {r.get('artist')}")
                st.audio(r["preview_url"], autoplay=item.get("fresh", False))
            elif ev["name"] == "get_weather" and "temperature_c" in r:
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{r['city']} — {r['condition'].title()}", f"{r['temperature_c']} °C", f"feels like {r['feels_like_c']} °C")
                c2.metric("💧 Humidity", f"{r.get('humidity_pct', '—')} %")
                c3.metric("💨 Wind", f"{r.get('wind_kmh', '—')} km/h")
            elif ev["name"] == "convert_currency" and "to" in r:
                st.info(f"💱 {r['from']} = **{r['to']}** (rate: {r['rate']})", icon="💱")
            elif ev["name"] == "search_wikipedia" and "summary" in r:
                with st.expander(f"📖 {r['title']}"):
                    st.write(r["summary"])
                    if r.get("url"):
                        st.markdown(f"[Read more on Wikipedia]({r['url']})")
            elif ev["name"] == "flip_coin_or_roll_dice" and "result" in r:
                label = "🪙 Coin flip" if "sides" not in r else f"🎲 Dice ({r['sides']}-sided)"
                st.success(f"{label}: **{r['result']}**")
        if item.get("tts"):
            st.audio(item["tts"], format="audio/wav", autoplay=item.get("fresh", False))
        item["fresh"] = False  # autoplay only once

# =============================================================================
# Input — push-to-talk mic + text fallback
# =============================================================================
from streamlit_mic_recorder import mic_recorder
rec = mic_recorder(start_prompt="🎙️ Start speaking", stop_prompt="⏹️ Stop", format="wav", key="mic")
audio = None
if rec and rec.get("bytes"):
    import io
    audio = io.BytesIO(rec["bytes"])
typed = st.chat_input("…or type your request")

user_text = None

if audio is not None:
    raw = audio.getvalue()
    h = hashlib.md5(raw).hexdigest()
    if h != ss.last_audio_hash:  # don't reprocess the same clip on rerun
        ss.last_audio_hash = h
        if not sarvam_key:
            st.error("SARVAM_API_KEY missing from Streamlit secrets.")
        else:
            ss.status = "listening"
            with st.spinner("👂 Transcribing…"):
                try:
                    user_text = speech_to_text(raw, sarvam_key)
                except Exception as exc:
                    st.error(f"STT failed: {exc}")
            ss.status = "idle"
            if user_text == "":
                st.warning("Didn't catch that — try again a bit louder.")
                user_text = None

if typed:
    user_text = typed.strip()

# =============================================================================
# One assistant turn
# =============================================================================
if user_text:
    if not anthropic_key:
        st.error("ANTHROPIC_API_KEY missing from Streamlit secrets.")
        st.stop()

    now_ts = datetime.now().strftime("%H:%M")
    ss.history.append({"role": "user", "content": user_text})
    ss.display.append({"role": "user", "content": user_text, "ts": now_ts})

    ss.status = "thinking"
    with st.spinner("🧠 Thinking…"):
        try:
            reply, tool_events = run_assistant(ss.history, anthropic_key)
        except Exception as exc:
            ss.history.pop()
            ss.display.pop()
            ss.status = "idle"
            st.error(f"Claude call failed: {exc}")
            st.stop()

    tts_bytes = None
    if voice_on and sarvam_key and reply:
        ss.status = "speaking"
        with st.spinner("🗣️ Generating voice…"):
            try:
                tts_bytes = text_to_speech(reply, sarvam_key)
            except Exception as exc:
                st.warning(f"TTS failed (showing text only): {exc}")

    ss.status = "idle"
    ss.history.append({"role": "assistant", "content": reply})
    ss.display.append(
        {"role": "assistant", "content": reply, "tools": tool_events, "tts": tts_bytes, "fresh": True, "ts": now_ts}
    )
    st.rerun()
