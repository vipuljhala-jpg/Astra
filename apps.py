"""
🔵 Astra — Build-Your-Own-Alexa (single-file Streamlit demo, WAKE-WORD edition)

Flow:
    (always listening) "Hey Astra / Jarvis / Alexa…"  → wake!
        → Astra greets you: "Yes, how can I help?"
        → your question is auto-recorded (stops when you stop speaking)
        → Sarvam STT → Claude (tool-calling) → Sarvam TTS → 🔊 spoken reply
        → back to listening for the wake word

How the always-on part works (real Alexa architecture, miniaturized):
    A background thread owns the microphone via `sounddevice` and runs
    Picovoice Porcupine (on-device wake-word engine — same tech class as
    Echo's wake chip). Nothing is sent anywhere until the wake word fires.
    After wake, simple energy-based endpointing records until ~1.2s of
    silence, then pushes the WAV into a queue. A Streamlit fragment polls
    the queue and runs the normal STT → Claude → TTS turn.

IMPORTANT — this only works when the app runs on the SAME machine as the
microphone (i.e. `streamlit run app.py` on your laptop, opened at
localhost). Deployed on a server, the server has no mic — use the
push-to-talk button instead (still included as fallback).

Setup:
    pip install -r requirements.txt
    1) Get a FREE Picovoice AccessKey at https://console.picovoice.ai
    2) (Optional) Train the custom wake word "Hey Astra" in that console,
       download the .ppn file for Windows, and put its path in the sidebar.
       Until then, use a built-in keyword like "jarvis" or "alexa".
    3) Keys: either .streamlit/secrets.toml with
           ANTHROPIC_API_KEY = "sk-ant-..."
           SARVAM_API_KEY = "..."
           PICOVOICE_ACCESS_KEY = "..."
       or paste them in the sidebar.

Tip: use headphones — otherwise the mic can hear Astra's own voice/music.
"""

import base64
import hashlib
import io
import json
import queue
import threading
import time
import wave
from datetime import datetime
from zoneinfo import ZoneInfo

import anthropic
import requests
import streamlit as st

# Audio deps are optional — app still runs (push-to-talk + text) without them
try:
    import numpy as np
    import sounddevice as sd
    import pvporcupine
    WAKE_DEPS_OK = True
    WAKE_DEPS_ERR = ""
except Exception as _e:  # ImportError or missing PortAudio
    WAKE_DEPS_OK = False
    WAKE_DEPS_ERR = str(_e)

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
    <p class="astra-sub">Say the wake word — hands free, like Alexa</p>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Shared HTTP session + cached Claude client (latency)
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
# Skills (Alexa's "intents")
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
            f"https://open.er-api.com/v6/latest/{from_currency.upper()}", timeout=8,
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
        data = http().get("https://official-joke-api.appspot.com/random_joke", timeout=8).json()
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
        return {
            "title": resp["title"], "summary": resp["extract"][:400],
            "url": resp.get("content_urls", {}).get("desktop", {}).get("page", ""),
        }
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
# Brain — Claude tool-calling loop
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
    for _ in range(3):
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


@st.cache_data(show_spinner=False)
def cached_tts(text: str, sarvam_key: str) -> bytes:
    """Cache fixed phrases (the wake greeting) so they play instantly."""
    return text_to_speech(text, sarvam_key)


def wav_duration_sec(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 3.0  # safe guess


# =============================================================================
# Wake-word engine — background thread that owns the microphone
# =============================================================================
def _pcm_to_wav(frames: list[bytes], sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"".join(frames))
    return buf.getvalue()


def _record_utterance(stream, sample_rate, frame_len, state,
                      max_sec=10.0, silence_sec=1.3) -> bytes | None:
    """After wake: record until the user stops speaking (energy endpointing)."""
    state["mode"] = "recording"
    frames, started, silent, t0 = [], False, 0.0, time.time()
    threshold = state["threshold"]
    while state["running"] and (time.time() - t0) < max_sec:
        data, _ = stream.read(frame_len)
        frames.append(bytes(data))
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
        if rms > threshold:
            started, silent = True, 0.0
        elif started:
            silent += frame_len / sample_rate
            if silent >= silence_sec:
                break
    state["mode"] = "wake"
    if not started:
        return None  # user woke it but said nothing
    return _pcm_to_wav(frames, sample_rate)


def _wake_loop(state: dict, access_key: str, keyword: str,
               keyword_path: str, sensitivity: float, greet_delay: float):
    """Background thread: Porcupine wake word -> record utterance -> queue."""
    try:
        if keyword_path:
            pp = pvporcupine.create(access_key=access_key,
                                    keyword_paths=[keyword_path],
                                    sensitivities=[sensitivity])
        else:
            pp = pvporcupine.create(access_key=access_key,
                                    keywords=[keyword],
                                    sensitivities=[sensitivity])
    except Exception as exc:
        state["error"] = f"Porcupine init failed: {exc}"
        return
    try:
        with sd.RawInputStream(samplerate=pp.sample_rate, blocksize=pp.frame_length,
                               dtype="int16", channels=1) as stream:
            state["mode"] = "wake"
            while state["running"]:
                data, _ = stream.read(pp.frame_length)
                if time.time() < state["muted_until"]:
                    continue  # Astra is speaking — ignore own voice
                pcm = np.frombuffer(data, dtype=np.int16)
                if pp.process(pcm) >= 0:
                    state["events"].put(("wake", None))
                    time.sleep(greet_delay)  # let the greeting play
                    audio = _record_utterance(stream, pp.sample_rate, pp.frame_length, state)
                    if audio:
                        state["events"].put(("utterance", audio))
                    else:
                        state["events"].put(("no_speech", None))
    except Exception as exc:
        state["error"] = f"Microphone stream failed: {exc}"
    finally:
        pp.delete()


def start_engine(access_key, keyword, keyword_path, sensitivity, threshold, greet_delay):
    state = {
        "events": queue.Queue(), "mode": "starting", "running": True,
        "muted_until": 0.0, "error": None, "threshold": threshold,
        "params": (access_key, keyword, keyword_path, sensitivity, greet_delay),
    }
    t = threading.Thread(target=_wake_loop, daemon=True,
                         args=(state, access_key, keyword, keyword_path, sensitivity, greet_delay))
    t.start()
    return state


# =============================================================================
# Keys — Streamlit secrets first, sidebar fallback (no crash if missing)
# =============================================================================
def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


# =============================================================================
# Sidebar
# =============================================================================
BUILTIN_KEYWORDS = ["jarvis", "alexa", "computer", "hey google", "hey siri", "porcupine", "bumblebee"]

with st.sidebar:
    st.header("🔑 API Keys")
    anthropic_key = secret("ANTHROPIC_API_KEY") or st.text_input("Anthropic API key", type="password")
    sarvam_key = secret("SARVAM_API_KEY") or st.text_input("Sarvam API key", type="password")

    st.divider()
    st.header("🎤 Wake word (hands-free)")
    if not WAKE_DEPS_OK:
        st.info(
            "Hands-free 'Hey Astra' works only when this app runs **locally on "
            "your own machine** (the server needs direct mic access). On this "
            "deployment, use 🎙️ Push-to-talk or typing below — the full voice "
            "pipeline still works!",
            icon="ℹ️",
        )
        pico_key, wake_enabled = "", False
        keyword, keyword_path, sensitivity, mic_threshold = "jarvis", "", 0.6, 350
        greeting_text = "Yes? How can I help you?"
    else:
        pico_key = secret("PICOVOICE_ACCESS_KEY") or st.text_input(
            "Picovoice AccessKey", type="password", help="Free at console.picovoice.ai"
        )
        wake_enabled = st.toggle("Always listening", value=bool(pico_key),
                                 disabled=not pico_key)
        keyword = st.selectbox("Built-in wake word", BUILTIN_KEYWORDS, index=0,
                               help="Instant option. For 'Hey Astra', train it free on console.picovoice.ai and set the .ppn path below.")
        keyword_path = st.text_input("Custom .ppn path (optional)", placeholder=r"C:\path\Hey-Astra_en_windows.ppn")
        sensitivity = st.slider("Wake sensitivity", 0.1, 1.0, 0.6, 0.05)
        mic_threshold = st.slider("Speech threshold (endpointing)", 100, 1500, 350, 50,
                                  help="Raise if it never stops recording (noisy room); lower if it cuts you off.")
        greeting_text = st.text_input("Wake greeting", value="Yes? How can I help you?")

    voice_on = st.toggle("🔊 Voice replies (TTS)", value=True)

    st.divider()
    st.markdown("**🚀 Capabilities**")
    with st.expander("🌦️ Weather & Time"):
        st.markdown("- *Is it going to rain in Mumbai?*\n- *What time is it in New York?*")
    with st.expander("💱 Finance"):
        st.markdown("- *Convert 10,000 JPY to EUR*\n- *GBP to INR rate?*")
    with st.expander("🎵 Music"):
        st.markdown("- *Play Blinding Lights*\n- *Play Kesariya from Brahmastra*")
    with st.expander("🧠 Knowledge"):
        st.markdown("- *Who is Sundar Pichai?*\n- *Explain quantum entanglement simply*")
    with st.expander("🎲 Fun & Utilities"):
        st.markdown("- *Roll a 20-sided dice*\n- *Tell me a joke*")

    st.divider()
    col1, col2 = st.columns(2)
    if col1.button("🗑️ Clear"):
        for k in ("history", "display", "last_audio_hash"):
            st.session_state.pop(k, None)
        st.rerun()
    if col2.button("💾 Export"):
        lines = [f"[{i.get('ts','')}] {i['role'].upper()}: {i['content']}"
                 for i in st.session_state.get("display", [])]
        st.download_button("⬇️ Download", "\n".join(lines), file_name="astra_chat.txt", mime="text/plain")

# =============================================================================
# Session state
# =============================================================================
ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("display", [])
ss.setdefault("last_audio_hash", "")

# =============================================================================
# Start / restart / stop the wake engine to match sidebar settings
# =============================================================================
engine = ss.get("engine")
wanted_params = (pico_key, keyword, keyword_path.strip(), sensitivity, 2.0)

if engine and (not wake_enabled or engine["params"] != wanted_params):
    engine["running"] = False          # stop old thread
    ss.engine = engine = None
if wake_enabled and WAKE_DEPS_OK and pico_key and engine is None:
    ss.engine = engine = start_engine(pico_key, keyword, keyword_path.strip(),
                                      sensitivity, mic_threshold, greet_delay=2.0)
if engine:
    engine["threshold"] = mic_threshold  # live-tunable


# =============================================================================
# Shared turn processing (used by wake word, push-to-talk, and typing)
# =============================================================================
def process_turn(user_text: str, in_fragment: bool):
    if not anthropic_key:
        st.error("Anthropic API key missing (secrets or sidebar).")
        return
    now_ts = datetime.now().strftime("%H:%M")
    ss.history.append({"role": "user", "content": user_text})
    ss.display.append({"role": "user", "content": user_text, "ts": now_ts})

    with st.spinner("🧠 Thinking…"):
        try:
            reply, tool_events = run_assistant(ss.history, anthropic_key)
        except Exception as exc:
            ss.history.pop(); ss.display.pop()
            st.error(f"Claude call failed: {exc}")
            return

    tts_bytes = None
    if voice_on and sarvam_key and reply:
        with st.spinner("🗣️ Generating voice…"):
            try:
                tts_bytes = text_to_speech(reply, sarvam_key)
            except Exception as exc:
                st.warning(f"TTS failed (showing text only): {exc}")

    # Mute the wake mic while Astra speaks, so she doesn't hear herself
    if tts_bytes and engine:
        engine["muted_until"] = time.time() + wav_duration_sec(tts_bytes) + 0.6

    ss.history.append({"role": "assistant", "content": reply})
    ss.display.append({"role": "assistant", "content": reply, "tools": tool_events,
                       "tts": tts_bytes, "fresh": True, "ts": now_ts})
    st.rerun(scope="app") if in_fragment else st.rerun()


# =============================================================================
# Wake-word poller — fragment that checks the mic thread's queue twice a second
# =============================================================================
@st.fragment(run_every=0.5)
def wake_poller():
    if engine is None:
        if wake_enabled:
            st.caption("⏳ starting wake engine…")
        return
    if engine.get("error"):
        st.error(f"Wake engine: {engine['error']}")
        return

    mode = engine.get("mode", "starting")
    badges = {
        "starting": ("#7a8aa0", "⏳ Starting…"),
        "wake": ("#19c2ff", f"👂 Listening for “{keyword if not keyword_path.strip() else 'Hey Astra'}”…"),
        "recording": ("#f5a623", "🔴 Recording your question — speak now!"),
    }
    color, label = badges.get(mode, badges["starting"])
    st.markdown(f'<p style="text-align:center;color:{color};font-size:0.9rem">{label}</p>',
                unsafe_allow_html=True)

    try:
        event, payload = engine["events"].get_nowait()
    except queue.Empty:
        return

    if event == "wake":
        # Greet instantly (cached TTS) and mute mic for the greeting duration
        if voice_on and sarvam_key:
            try:
                g = cached_tts(greeting_text, sarvam_key)
                engine["muted_until"] = time.time() + wav_duration_sec(g) + 0.2
                st.audio(g, format="audio/wav", autoplay=True)
            except Exception:
                pass
        st.toast("👂 Wake word detected — ask your question!")

    elif event == "no_speech":
        st.toast("Didn't hear a question — say the wake word again.")

    elif event == "utterance":
        if not sarvam_key:
            st.error("Sarvam API key missing — can't transcribe.")
            return
        with st.spinner("👂 Transcribing…"):
            try:
                text = speech_to_text(payload, sarvam_key)
            except Exception as exc:
                st.error(f"STT failed: {exc}")
                return
        if not text:
            st.toast("Couldn't understand that — try again.")
            return
        process_turn(text, in_fragment=True)


wake_poller()

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
                c1.metric(f"{r['city']} — {r['condition'].title()}", f"{r['temperature_c']} °C",
                          f"feels like {r['feels_like_c']} °C")
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
        item["fresh"] = False

# =============================================================================
# Fallback inputs — push-to-talk + text (work even without the wake engine)
# =============================================================================
audio = None
try:
    from streamlit_mic_recorder import mic_recorder
    rec = mic_recorder(start_prompt="🎙️ Push to talk", stop_prompt="⏹️ Stop",
                       format="wav", key="mic")
    if rec and rec.get("bytes"):
        audio = io.BytesIO(rec["bytes"])
except Exception:
    st.caption("Push-to-talk unavailable (`pip install streamlit-mic-recorder`) — typing still works.")

typed = st.chat_input("…or type your request")

user_text = None
if audio is not None:
    raw = audio.getvalue()
    h = hashlib.md5(raw).hexdigest()
    if h != ss.last_audio_hash:
        ss.last_audio_hash = h
        if not sarvam_key:
            st.error("Sarvam API key missing (secrets or sidebar).")
        else:
            with st.spinner("👂 Transcribing…"):
                try:
                    user_text = speech_to_text(raw, sarvam_key)
                except Exception as exc:
                    st.error(f"STT failed: {exc}")
            if user_text == "":
                st.warning("Didn't catch that — try again a bit louder.")
                user_text = None

if typed:
    user_text = typed.strip()

if user_text:
    process_turn(user_text, in_fragment=False)
