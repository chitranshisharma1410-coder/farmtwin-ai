import os, json, re, time, pathlib, httpx
from dotenv import load_dotenv
# Always load .env next to this file, even when Uvicorn is launched from another directory.
ENV_FILE = pathlib.Path(__file__).resolve().with_name(".env")
load_dotenv(dotenv_path=ENV_FILE)
SARVAM_KEY = (os.getenv("SARVAM_API_KEY") or "").strip()
SARVAM_BASE = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai")
SARVAM_CHAT_MODEL = os.getenv("SARVAM_CHAT_MODEL", "sarvam-105b")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
client = bool(SARVAM_KEY)
COST_KL = float(os.getenv("WATER_COST_PER_KL", 15))  # Rs per 1000 L
MM_PER_PCT = 3.0  # 1% soil moisture ~ 3 mm water in the root zone (calibrate per soil)

# A plot's coordinates don't change depending on who is looking at the dashboard or
# which network/VPN they're on, so the farm's location is a fixed server-side setting
# rather than something read from the visitor's browser. Set these once per deployment.
FARM_LAT = float(os.getenv("FARM_LAT", 12.69))
FARM_LON = float(os.getenv("FARM_LON", 79.98))
FARM_LOCATION_LABEL = os.getenv("FARM_LOCATION_LABEL", "Default farm location")

# ---------- upag.gov.in mandi prices via Parse.bot ----------
# Parse.bot turns a normal webpage into a typed API. To use it you first create a
# "scraper" for the upag.gov.in prices page in the Parse.bot dashboard (paste the
# page URL, describe the fields you want - commodity, market, modal price, date);
# that gives you a scraper_id and an endpoint_name, which go in .env alongside the
# API key. Until PARSEBOT_SCRAPER_ID/PARSEBOT_ENDPOINT are set this agent stays
# quietly inactive (same "not configured" pattern as the other data sources).
PARSEBOT_KEY = os.getenv("PARSEBOT_API_KEY")
PARSEBOT_SCRAPER_ID = os.getenv("PARSEBOT_SCRAPER_ID")
PARSEBOT_ENDPOINT = os.getenv("PARSEBOT_ENDPOINT", "prices")


def upag_market_agent(crop, state="Tamil Nadu"):
    if not (PARSEBOT_KEY and PARSEBOT_SCRAPER_ID):
        return {"note": "Set PARSEBOT_SCRAPER_ID (and PARSEBOT_ENDPOINT) to see live upag.gov.in prices"}
    try:
        r = httpx.get(
            f"https://api.parse.bot/scraper/{PARSEBOT_SCRAPER_ID}/{PARSEBOT_ENDPOINT}",
            timeout=10, headers={"X-API-Key": PARSEBOT_KEY},
            params={"commodity": CROPS[crop]["agmark"], "state": state}).json()
        return {"records": r if isinstance(r, list) else r.get("records", r)}
    except Exception as e:
        return {"note": f"upag.gov.in unavailable: {e}"}
CROPS = {  # kc = crop coefficient, thr = stress below this %, tgt = refill target %
    "paddy": dict(kc=1.2, thr=55, tgt=75, agmark="Paddy(Dhan)(Common)"),
    "groundnut": dict(kc=0.9, thr=35, tgt=55, agmark="Groundnut"),
    "tomato": dict(kc=1.1, thr=45, tgt=65, agmark="Tomato"),
    "millet": dict(kc=0.7, thr=30, tgt=50, agmark="Bajra(Pearl Millet/Cumbu)"),
}
FB = pathlib.Path("feedback.json")
_last = {}  # last successful weather fetch (cache)

# ---------- Language auto-detection (no dropdown needed) ----------
# Major Indian languages each use a visually/Unicode-distinct script, so once we
# have TEXT (typed, or transcribed from speech on the frontend) we can tell which
# language it's in just by which script's code points show up in it. This lets the
# backend decide the reply language itself instead of trusting a client-side picker.
LANGS = {"en": "English", "ta": "Tamil", "hi": "Hindi", "te": "Telugu", "kn": "Kannada",
         "ml": "Malayalam", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati",
         "pa": "Punjabi", "or": "Odia"}
_SCRIPT_RANGES = [  # (lang code, (Unicode block start, end)) - checked in this order
    ("ta", (0x0B80, 0x0BFF)), ("te", (0x0C00, 0x0C7F)), ("kn", (0x0C80, 0x0CFF)),
    ("ml", (0x0D00, 0x0D7F)), ("bn", (0x0980, 0x09FF)), ("gu", (0x0A80, 0x0AFF)),
    ("pa", (0x0A00, 0x0A7F)), ("or", (0x0B00, 0x0B7F)),
    ("hi", (0x0900, 0x097F)),  # Devanagari: shared by Hindi/Marathi - we default to Hindi,
                                # since script alone can't tell those two apart. If the
                                # farmer's UI language cookie says "mr", pass fallback="mr"
                                # and pure-Devanagari text will keep that instead.
]


def detect_lang(text, fallback="en"):
    """Guess the language of `text` from its script. Returns a key from LANGS.
    Falls back to `fallback` (normally 'the language this user used last time')
    when the text has no recognizable Indic script in it (e.g. numbers only, or
    the recognizer mis-heard and returned nothing usable)."""
    if not text:
        return fallback
    counts = {}
    for ch in text:
        cp = ord(ch)
        for code, (lo, hi) in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if counts:
        return max(counts, key=counts.get)
    return "en" if any(c.isalpha() for c in text) else fallback


def lang_name(code):
    return LANGS.get(code, "English")


def _sarvam_headers():
    return {"api-subscription-key": SARVAM_KEY, "Content-Type": "application/json"}

def llm(system, user, image=None, max_tokens=700):
    """Sarvam chat completion. Returns None on unavailable/error so rule-based paths still work."""
    if not SARVAM_KEY or image:
        return None
    try:
        r = httpx.post(f"{SARVAM_BASE}/v1/chat/completions", headers=_sarvam_headers(),
                       json={"model": SARVAM_CHAT_MODEL,
                             "messages":[{"role":"system","content":system},{"role":"user","content":user}],
                             "max_tokens":max_tokens, "temperature":0.2}, timeout=45)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None

def sarvam_transcribe_audio(audio_bytes, filename="recording.webm", mime="audio/webm", language_code=None):
    if not SARVAM_KEY:
        raise RuntimeError("SARVAM_API_KEY is not configured")
    if not audio_bytes or len(audio_bytes) < 256:
        raise RuntimeError("Recording is empty or too small. Hold Speak, talk, then press Stop & understand.")

    # Sarvam expects multipart/form-data for STT. Do not send a JSON Content-Type
    # here: httpx creates the multipart boundary automatically.
    safe_mime = (mime or "audio/webm").split(";", 1)[0]
    safe_name = filename or "recording.webm"
    files = {"file": (safe_name, audio_bytes, safe_mime)}
    data = {"model": SARVAM_STT_MODEL, "mode": "transcribe",
            "language_code": "unknown" if language_code in (None, "", "auto", "unknown") else language_code}
    try:
        r = httpx.post(f"{SARVAM_BASE}/speech-to-text",
                       headers={"api-subscription-key": SARVAM_KEY},
                       files=files, data=data, timeout=90)
    except Exception as e:
        raise RuntimeError(f"Sarvam STT connection failed: {e}") from e

    if r.status_code >= 400:
        # Preserve Sarvam's actual error message instead of exposing only
        # httpx's generic '400 Bad Request'. This makes audio-format problems
        # immediately diagnosable in the UI.
        try:
            err = r.json().get("error", {})
            msg = err.get("message") if isinstance(err, dict) else str(err)
            code = err.get("code") if isinstance(err, dict) else None
            detail = f"{code}: {msg}" if code and msg else (msg or r.text[:500])
        except Exception:
            detail = r.text[:500] or "Bad request"
        raise RuntimeError(f"Sarvam STT rejected the recording ({r.status_code}): {detail}")

    body = r.json()
    transcript = body.get("transcript") or body.get("text") or ""
    # Saaras normally returns language_code. If it does not, infer the language
    # from the transcript so the next /api/ask call still gets the correct lock.
    returned_code = (body.get("language_code") or "").lower()
    returned_lang = returned_code.split("-", 1)[0] if returned_code else ""
    detected_lang = returned_lang if returned_lang in LANGS else detect_lang(transcript, fallback="en")
    return {
        "transcript": transcript,
        "language_code": body.get("language_code"),
        "language_probability": body.get("language_probability"),
        "detected_lang": detected_lang,
    }

def _tts_speaker(language_code):
    code = (language_code or "en-IN").lower()
    if code.startswith("ta-"): return os.getenv("SARVAM_TAMIL_SPEAKER", "ratan")
    if code.startswith("hi-"): return os.getenv("SARVAM_HINDI_SPEAKER", "shubh")
    return speaker if (speaker := os.getenv("SARVAM_DEFAULT_SPEAKER", "shubh")) else "shubh"

def sarvam_tts(text, language_code="en-IN", speaker="shubh"):
    if not SARVAM_KEY:
        raise RuntimeError("SARVAM_API_KEY is not configured")
    r = httpx.post(f"{SARVAM_BASE}/text-to-speech", headers=_sarvam_headers(),
                   json={"text": text[:2500], "language_code": language_code,
                         "model": SARVAM_TTS_MODEL, "speaker": _tts_speaker(language_code), "output_audio_codec":"wav"}, timeout=90)
    r.raise_for_status()
    body = r.json()
    audios = body.get("audios") or []
    if not audios:
        raise RuntimeError("Sarvam TTS returned no audio")
    return audios[0]


def parse_json(t):
    try:
        return json.loads(re.search(r"\{.*\}", t or "", re.S).group(0))
    except Exception:
        return None


# ---------- User-selectable location ----------
def geocode_location(district, state="", country="India"):
    district = (district or "").strip()
    state = (state or "").strip()
    country = (country or "India").strip() or "India"
    if not district:
        raise RuntimeError("Enter a district name first.")
    query = ", ".join(x for x in (district, state, country) if x)
    try:
        r = httpx.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "jsonv2", "limit": 5, "countrycodes": "in"},
            headers={"User-Agent": "FarmTwin/1.0 (district weather lookup)"},
            timeout=12,
        )
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        raise RuntimeError(f"Location lookup failed: {e}") from e
    if not rows:
        raise RuntimeError(f"Could not find '{query}'. Try District + State, for example 'Salem, Tamil Nadu'.")
    best = rows[0]
    addr = best.get("address", {})
    return {
        "label": best.get("display_name", query),
        "district": addr.get("county") or addr.get("city_district") or addr.get("town") or district,
        "state": addr.get("state") or state,
        "country": addr.get("country") or country,
        "lat": float(best["lat"]),
        "lon": float(best["lon"]),
    }

def reverse_geocode(lat, lon):
    try:
        r = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10},
            headers={"User-Agent": "FarmTwin/1.0 (farm location lookup)"},
            timeout=12,
        )
        r.raise_for_status()
        body = r.json()
        addr = body.get("address", {})
        return {
            "label": body.get("display_name", "My location"),
            "district": addr.get("county") or addr.get("city_district") or addr.get("town") or addr.get("city") or "",
            "state": addr.get("state", ""),
            "country": addr.get("country", "India"),
            "lat": float(lat),
            "lon": float(lon),
        }
    except Exception:
        return {"label": "My location", "district": "", "state": "", "country": "India", "lat": float(lat), "lon": float(lon)}

# ---------- Weather agent ----------
def weather_agent(lat, lon):
    out = {"source": "Open-Meteo", "days": [], "status": "LIVE", "fetched_at": time.strftime("%Y-%m-%d %H:%M")}
    try:
        r = httpx.get("https://api.open-meteo.com/v1/forecast", timeout=10, params=dict(
            latitude=lat, longitude=lon, forecast_days=5, timezone="Asia/Kolkata",
            daily="precipitation_sum,precipitation_probability_max,temperature_2m_max,et0_fao_evapotranspiration",
            hourly="relative_humidity_2m")).json()
        d = r["daily"]
        out["days"] = [dict(date=d["time"][i], rain_mm=d["precipitation_sum"][i] or 0,
                            prob=d["precipitation_probability_max"][i] or 0,
                            tmax=d["temperature_2m_max"][i], et0=d["et0_fao_evapotranspiration"][i] or 4)
                       for i in range(5)]
        h = r["hourly"]["relative_humidity_2m"][:72]
        out["humidity_3d_avg"] = round(sum(h) / len(h))
    except Exception as e:
        if _last:
            return dict(_last, status="CACHED", error=str(e))
        out.update(source="Demo data (not real)", status="SIMULATED", error=str(e), humidity_3d_avg=78, days=[
            dict(date=f"Day {i+1}", rain_mm=m, prob=p, tmax=34, et0=4.5)
            for i, (m, p) in enumerate([(2, 20), (14, 78), (6, 50), (0, 10), (0, 10)])])
    imd = os.getenv("IMD_URL")  # verified IMD endpoint of your choice; kept as context for the advisor
    if imd:
        try:
            out["imd_raw"] = httpx.get(imd, timeout=8).text[:1500]
            out["source"] += " + IMD"
        except Exception:
            pass
    if out["status"] == "LIVE":
        _last.clear(); _last.update(out)
    return out


# ---------- Market agent (Agmarknet via data.gov.in, upag.gov.in as fallback) ----------
def market_agent(crop, state="Tamil Nadu"):
    key = os.getenv("DATA_GOV_IN_KEY")
    if not key:
        return upag_market_agent(crop, state)
    try:
        r = httpx.get("https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070", timeout=10, params={
            "api-key": key, "format": "json", "limit": 3,
            "filters[state.keyword]": state, "filters[commodity]": CROPS[crop]["agmark"]}).json()
        return {"records": [dict(market=x["market"], modal_price=x["modal_price"], date=x["arrival_date"])
                            for x in r.get("records", [])]}
    except Exception as e:
        return upag_market_agent(crop, state)


# ---------- Feedback agent ----------
def _rows():
    try:
        return json.loads(FB.read_text())
    except Exception:
        return []


def rain_trust():
    r = [x["actual_rain_mm"] / x["predicted_rain_mm"] for x in _rows()[-10:] if x["predicted_rain_mm"] > 0]
    return round(min(1.5, max(0.5, sum(r) / len(r))), 2) if r else 1.0


def log_feedback(row):
    FB.write_text(json.dumps(_rows() + [row]))
    p, a = row["predicted_rain_mm"], row["actual_rain_mm"]
    return dict(rain_trust=rain_trust(), difference=round(a - p, 1), accuracy=round(1 - abs(a - p) / max(a, p, 1), 2))


# ---------- Water agent: what-if simulation + constrained optimizer ----------
def run(zones, wx, mode, cap, rain_scale=1.0, no_rain=0):
    """Deterministic zone water-balance simulation.

    Each day first applies ET loss and effective rainfall to soil moisture. If the
    strategy calls for irrigation, litres are calculated from the *predicted*
    pre-irrigation moisture, not yesterday's starting moisture:

        litres = max(target - predicted_moisture, 0) * MM_PER_PCT * area_m2

    This removes the old double-counting/ordering ambiguity that made zone litres
    look inconsistent with the decision reason.
    """
    zs = [dict(z, m=float(z["moisture"])) for z in zones]
    used = stress = 0
    required = 0
    day0 = []
    zone_totals = {z["name"]: 0 for z in zs}
    for d, day in enumerate(wx["days"]):
        rain = 0 if d < no_rain else day["rain_mm"] * rain_scale
        for z in sorted(zs, key=lambda z: z["m"] - CROPS[z["crop"]]["thr"]):
            c = CROPS[z["crop"]]
            loss = day["et0"] * c["kc"] / MM_PER_PCT
            gain = rain * 0.7 / MM_PER_PCT
            predicted = min(100, max(0, z["m"] + gain - loss))
            if mode == "all":
                go = d == 0
            elif mode == "wait2":
                go = d == 2
            else:
                go = predicted < c["thr"] + 5
            irr = 0
            if go:
                needed = max(0, c["tgt"] - predicted) * MM_PER_PCT * z["area"]
                required += needed
                irr = min(needed, max(0, cap - used))
                used += irr
                zone_totals[z["name"]] += irr
            z["m"] = min(100, max(0, predicted + irr / (MM_PER_PCT * z["area"])))
            stress += z["m"] < c["thr"]
            if d == 0:
                day0.append(dict(zone=z["name"], liters=round(irr), moisture_before=round(z["m"] - irr / (MM_PER_PCT * z["area"]), 1), moisture_after=round(z["m"], 1)))
    return dict(water_used=round(used), water_required=round(required), cost=round(used / 1000 * COST_KL), stress_days=int(stress),
                risk="Low" if stress == 0 else "Medium" if stress <= 2 else "High", day0=day0,
                zone_totals={k: round(v) for k,v in zone_totals.items()})


def water_agent(zones, wx, water_l, budget, rain_scale=1.0, no_rain=0, temp_delta=0):
    if temp_delta:  # hotter days raise evapotranspiration (rough +4% per degC)
        wx = dict(wx, days=[dict(d, et0=d["et0"] * (1 + 0.04 * temp_delta)) for d in wx["days"]])
    cap = min(water_l, budget / COST_KL * 1000)
    sc = {k: run(zones, wx, k, cap, rain_scale, no_rain) for k in ("all", "wait2", "smart")}
    names = {"all": "Irrigate everything today", "wait2": "Wait 2 days", "smart": "FarmTwin plan"}
    plan = []
    for a in sorted(sc["smart"]["day0"], key=lambda a: a["zone"]):
        plan.append(f"Irrigate {a['zone']} - {a['liters']} L" if a["liters"] else f"Don't irrigate {a['zone']}")
    if any(x["rain_mm"] * rain_scale > 1 for x in wx["days"]) and no_rain < 5:
        plan.append("Reassess after rainfall")
    a, w, s = sc["all"], sc["wait2"], sc["smart"]
    dry = run(zones, wx, "smart", cap, rain_scale, 5)  # SDG13 stress test: no rain for 5 days
    spared = sum(1 for x in s["day0"] if x["liters"] == 0) if wx["days"][0]["rain_mm"] * rain_scale > 1 else 0
    # "water saved" compares smart scheduling to flooding every zone today, with
    # NEITHER capped by your water/budget inputs. Comparing the two capped scenarios
    # (as before) breaks whenever your cap is tight: both strategies simply spend the
    # whole cap and the difference collapses to 0, which reads as "not working" even
    # though the cap is doing exactly what it should. This version stays meaningful
    # (and roughly constant) regardless of what you put in Water available/Budget;
    # `shortfall_l` below is what actually reacts to those two inputs.
    UNCAPPED = 10 ** 9
    all_needed = run(zones, wx, "all", UNCAPPED, rain_scale, no_rain)["water_used"]
    smart_needed = run(zones, wx, "smart", UNCAPPED, rain_scale, no_rain)["water_used"]
    saved = max(0, all_needed - smart_needed)
    shortfall = max(0, round(smart_needed - cap))  # how far your cap falls short of the smart plan's real need
    sdg = [
        dict(goal="SDG 2", target="Sustainable, climate-resilient food production for small farmers",
             value=f"{max(0, w['stress_days'] - s['stress_days'])} crop-stress zone-days avoided vs waiting 2 days"),
        dict(goal="SDG 6", target="Water-use efficiency",
             value=f"{saved} L saved ({round(100 * saved / all_needed) if all_needed else 0}% less than irrigating everything), by scheduling instead of flooding every zone"
                   + (f". Your {round(cap)} L cap is {shortfall} L short of what the smart plan needs, so some zones will still fall short" if shortfall else "")),
        dict(goal="SDG 12", target="Responsible use of resources",
             value=f"Rs {max(0, a['cost'] - s['cost'])} pumping cost avoided; the plan stays inside your water and budget caps"),
        dict(goal="SDG 13", target="Resilience to climate extremes",
             value=f"Dry-spell stress test (no rain 5 days): crop risk {dry['risk']}, water {dry['water_used']} L"),
        dict(goal="SDG 15", target="Halt land degradation",
             value=f"{spared} zone(s) spared irrigation ahead of rain, so less runoff and waterlogging"),
    ]
    return dict(
        scenarios=[dict(key=k, name=names[k], **sc[k]) for k in ("all", "wait2", "smart")],
        plan=plan, water_saved=saved, water_needed=smart_needed, shortfall_l=shortfall,
        requested_water_l=round(water_l), budget=round(budget), cap_l=round(cap), sdg=sdg)


# ---------- Vision agent (image + weather) ----------
def vision_agent(b64, mime, wx, zones, lang):
    # There's no text to auto-detect a language from here (it's a photo, not a
    # question), so this trusts whatever `lang` the frontend last detected/used -
    # the farmer never has to pick a language, but a photo alone can't confirm one.
    ctx = f"Humidity last 3 days: {wx.get('humidity_3d_avg')}%. Rain forecast mm: {[d['rain_mm'] for d in wx['days']]}. Zones: {zones}"
    t = llm("You are a crop-health agent for Indian smallholders. Reply with JSON only: "
            '{"diagnosis":str,"confidence":0-1,"reasoning":str,"treat_now":bool}. '
            "Reasoning MUST combine the photo with the weather context (e.g. humidity, coming rain). "
            f"Write text fields in {lang_name(lang)}.",
            ctx, image=(mime, b64))
    return parse_json(t) or {"diagnosis": "Vision agent offline (set ANTHROPIC_API_KEY)", "confidence": 0,
                             "reasoning": "", "treat_now": False, "detected_lang": lang}


# ---------- Situation parser fallback ----------
# ---------- Deterministic numeric helpers for farmer what-if inputs ----------
_EN_NUM = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,
    "seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,
    "sixty":60,"seventy":70,"eighty":80,"ninety":90,"hundred":100,"thousand":1000,"lakh":100000,
}
_HI_NUM = {"शून्य":0,"एक":1,"दो":2,"तीन":3,"चार":4,"पांच":5,"पाँच":5,"छह":6,"छः":6,"सात":7,"आठ":8,"नौ":9,
           "दस":10,"ग्यारह":11,"बारह":12,"तेरह":13,"चौदह":14,"पंद्रह":15,"पन्द्रह":15,"सोलह":16,
           "सत्रह":17,"अठारह":18,"उन्नीस":19,"बीस":20,"तीस":30,"चालीस":40,"पचास":50,"साठ":60,
           "सत्तर":70,"अस्सी":80,"नब्बे":90,"सौ":100,"हजार":1000,"हज़ार":1000,"लाख":100000}
_TA_NUM = {"பூஜ்யம்":0,"ஒன்று":1,"இரண்டு":2,"மூன்று":3,"நான்கு":4,"ஐந்து":5,"ஆறு":6,"ஏழு":7,"எட்டு":8,"ஒன்பது":9,
           "பத்து":10,"பதினொன்று":11,"பன்னிரண்டு":12,"பதின்மூன்று":13,"பதினான்கு":14,"பதினைந்து":15,"பதினாறு":16,
           "பதினேழு":17,"பதினெட்டு":18,"பத்தொன்பது":19,"இருபது":20,"முப்பது":30,"நாற்பது":40,"ஐம்பது":50,
           "அறுபது":60,"எழுபது":70,"எண்பது":80,"தொண்ணூறு":90,"நூறு":100,"ஆயிரம்":1000,"ஆயிர":1000,"லட்சம்":100000}

def _word_number_value(text):
    """Parse common whole-number phrases such as 'five thousand', 'पाँच हजार', 'ஐயாயிரம்'."""
    if not text: return None
    t = text.lower().strip()
    # Direct dictionary entries first.
    for k,v in sorted({**_EN_NUM, **_HI_NUM, **_TA_NUM}.items(), key=lambda kv: -len(kv[0])):
        if t == k: return float(v)
    # English/Hindi additive phrases: five thousand, पाँच हजार, etc.
    tokens = re.findall(r"[a-zA-Z]+|[\u0900-\u097F]+|[\u0B80-\u0BFF]+", t)
    maps = ({**_EN_NUM, **_HI_NUM, **_TA_NUM})
    total = 0; current = 0; found = False
    for tok in tokens:
        if tok not in maps: continue
        found = True; v = maps[tok]
        if v >= 100:
            current = max(1,current) * v
            total += current; current = 0
        else:
            current += v
    if found and (total + current) > 0:
        return float(total + current)
    return None


def _find_quantity_near_words(t, quantity_words, unit_words):
    """Find ONE quantity tied to its nearest unit, without swallowing earlier quantities.

    This is deliberately strict: in a sentence such as
    'I have 4000 litres and 8000 rupees', the water matcher must return 4000
    and the budget matcher must return 8000. The old matcher could consume the
    whole phrase before 'rupees' and accidentally add 4000 + 8000 = 12000.
    """
    unit = "|".join(map(re.escape, unit_words))
    qwords = "|".join(map(re.escape, quantity_words))
    m = re.search(r"([0-9][0-9,]*(?:\.[0-9]+)?\s*[kKmM]?)\s*(?:"+unit+r")", t, re.I)
    if m:
        raw=m.group(1).replace(",","").replace(" ","")
        mult=1000 if raw.lower().endswith('k') else (1000000 if raw.lower().endswith('m') else 1)
        raw=raw.rstrip('kKmM')
        return float(raw)*mult

    # Only accept contiguous number-word tokens immediately before the unit.
    # This prevents unrelated earlier quantities from being included.
    all_maps = {**_EN_NUM, **_HI_NUM, **_TA_NUM}
    token_alt = "|".join(sorted((re.escape(k) for k in all_maps), key=len, reverse=True))
    m = re.search(r"((?:(?:"+token_alt+r")[\s-]*){1,6})(?:"+unit+r")", t, re.I)
    if m:
        val=_word_number_value(m.group(1).strip())
        if val is not None: return val

    # Quantity word + numeric value: 'water is 5000', 'पानी 5000'.
    m = re.search(r"(?:"+qwords+r")[^0-9]{0,30}([0-9][0-9,]*(?:\.[0-9]+)?)", t, re.I)
    if m: return float(m.group(1).replace(",",""))
    return None

def _extract_situation(text):
    """Deterministically extract explicit what-if values from farmer text.

    Explicit farmer numbers always win over dashboard defaults and over the LLM.
    Includes English, Hindi/Hinglish and common Tamil/Tanglish wording.
    """
    t = (text or "").strip().lower()
    out = {}
    num = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
    def n(v): return float(v.replace(",", ""))

    water_words = ("water", "paani", "pani", "पानी", "जल", "litre", "liter", "லிட்டர்", "லிட்டர", "லிட்", "தண்ணி", "தண்ணீர்", "நீர்", "నీళ్లు", "నీరు")
    water_units = ("l", "lt", "ltr", "litre", "litres", "liter", "liters", "லிட்டர்", "லிட்டர", "லிட்", "லிட்டர்கள்", "నీళ్లు", "నీరు")
    # First try a strict number + unit match; then support 'water 5000', 5k L,
    # and number words such as 'five thousand litres' / 'पाँच हजार लीटर'.
    val = _find_quantity_near_words(t, water_words, water_units)
    litre_present = bool(re.search(r"(?:[0-9][0-9,]*(?:\.[0-9]+)?\s*[kKmM]?\s*(?:l|lt|ltr|litre|litres|liter|liters)|(?:லிட்டர்|லிட்டர|லிட்|லிட்டர்கள்|लीटर|लीटर्स))", t, re.I))
    if val is not None and (any(k in t for k in water_words) or litre_present):
        out["water_l"] = val

    # Explicit Hindi/Tamil number words immediately before the water unit.
    if "water_l" not in out:
        numword = "|".join(sorted((re.escape(k) for k in {**_HI_NUM, **_TA_NUM}), key=len, reverse=True))
        wm = re.search(r"((?:(?:"+numword+r")[\s-]*){1,6})(?:லிட்டர்|லிட்டர|லிட்|लीटर|लीटर्स|लीटरों)\b", t, re.I)
        if wm:
            vv = _word_number_value(wm.group(1).strip())
            if vv is not None: out["water_l"] = vv

    # Tamil water forms where the number precedes/folllows the noun, e.g. "5000 லிட்டர் தண்ணி"
    if "water_l" not in out:
        m = re.search(r"(?:தண்ணி|தண்ணீர்|நீர்|నీళ్లు|నీరు).*?" + num + r"\s*(?:லிட்டர்|லிட்டர|லிட்|l|litre|liter)?", t, re.I)
        if not m:
            m = re.search(num + r"\s*(?:லிட்டர்|லிட்டர|லிட்)\s*(?:தண்ணி|தண்ணீர்|நீர்)", t, re.I)
        if m:
            # first capture is either the number in the first pattern or the only number in the second
            nums = re.findall(num, m.group(0))
            if nums: out["water_l"] = n(nums[-1])

    budget_words = ("budget", "rs", "rupee", "rupees", "₹", "रुपये", "रुपया", "पैसे", "பட்ஜெட்", "ரூபாய்", "ரூபாய்கள்")
    bm = re.search(r"(?:₹|rs\.?\s*|rupees?\s*|budget\s*(?:is|of|=|:)?\s*|பட்ஜெட்\s*(?:=|:)?\s*|ரூபாய்\s*(?:=|:)?\s*|रुपये?\s*(?:=|:)?\s*)" + num, t, re.I)
    if bm and any(k in t for k in budget_words):
        out["budget"] = n(bm.group(1))
    else:
        # Number words followed by rupee/budget unit.
        units=("rupees", "rupee", "रुपये", "रुपया", "ரூபாய்", "பட்ஜெட்")
        val=_find_quantity_near_words(t, ("budget","रुपये","रुपய","பட்ஜெட்"), units)
        if val is not None and any(k in t for k in budget_words): out["budget"]=val


    # Rain change: English/Hinglish/Hindi/Tamil.
    rain = r"(?:rain|rainfall|baarish|barish|बारिश|वर्षा|மழை|மழைய|மழைநீர்)"
    less = r"(?:less|lower|kam|कम|decrease|reduction|reduced|குறை|குறைவு|குறைந்த|குறையும்)"
    more = r"(?:more|higher|zyada|ज्यादा|increase|increased|அதிக|அதிகம்|அதிகர)"
    m = re.search(num + r"\s*%?\s*" + less + r".*?" + rain, t, re.I) or re.search(rain + r".*?" + num + r"\s*%?\s*" + less, t, re.I)
    if m:
        nums = re.findall(num, m.group(0)); out["rain_scale"] = max(0.0, 1.0 - n(nums[0]) / 100.0)
    else:
        m = re.search(num + r"\s*%?\s*" + more + r".*?" + rain, t, re.I) or re.search(rain + r".*?" + num + r"\s*%?\s*" + more, t, re.I)
        if m:
            nums = re.findall(num, m.group(0)); out["rain_scale"] = 1.0 + n(nums[0]) / 100.0

    # Dry spell, including Tamil "5 நாட்களுக்கு மழை இல்லை".
    days = r"(?:days?|din|दिन|நாள்|நாட்கள்|நாட்களுக்கு)"
    no_rain = r"(?:no|without|nahi|nahin|नहीं|बिना|இல்லை|இல்ல|மழை இல்லை|மழையில்லை)"
    m = re.search(num + r"\s*" + days + r".*?" + no_rain + r".*?" + rain, t, re.I)
    if not m: m = re.search(no_rain + r".*?" + rain + r".*?" + num + r"\s*" + days, t, re.I)
    if not m: m = re.search(num + r"\s*" + days + r".*?" + rain + r".*?" + no_rain, t, re.I)
    if m:
        nums = re.findall(num, m.group(0)); out["no_rain_days"] = max(0, min(5, int(n(nums[0]))))

    # Temperature increase.
    degree = r"(?:°?c|degrees?|degree|டிகிரி|டிகிரிகள்|டிகிரி செல்சியஸ்|डिग्री)"
    hot = r"(?:hotter|higher|increase|more|गरम|बढ़|ज्यादा|சூடு|வெப்பம்|அதிகர)"
    m = re.search(r"(?:\+\s*)?" + num + r"\s*" + degree + r".*?" + hot, t, re.I) or re.search(hot + r".*?(?:\+\s*)?" + num + r"\s*" + degree, t, re.I)
    if m:
        nums = re.findall(num, m.group(0)); out["temp_delta"] = n(nums[0])
    return out


def _script_lang(text, fallback="en"):
    """Detect the language of THIS request, with script first and romanized cues second.

    Important: do not keep the previous UI language when a new request is clearly
    English. The reply language must follow the farmer's current utterance.
    """
    if not text: return fallback
    counts = {}
    for ch in text:
        cp = ord(ch)
        for code, (lo, hi) in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if counts:
        return max(counts, key=counts.get)

    t = " " + re.sub(r"[^a-zA-Z]+", " ", text.lower()) + " "
    tamil_cues = (
        " thanni ", " thanneer ", " tannir ", " mazha ", " mazhai ",
        " irukku ", " iruku ", " venum ", " venumaa ", " venuma ",
        " naal ", " naatkal ", " naatku ", " vidanum ", " vidanuma ",
        " paasanam ", " pasanam ", " vivasayi ", " vivasayigal ",
        " enakku ", " enga ", " engaluku ", " konjam ", " evlo ",
        " neer ", " thani ", " payir ", " nel ", " vivasayi "
    )
    hindi_cues = (
        " paani ", " pani ", " baarish ", " barish ", " din ",
        " hai ", " nahi ", " chahiye ", " mere paas ", " mere pas ",
        " kitna ", " kitne ", " kheti ", " fasal ", " sinchai "
    )
    if sum(c in t for c in tamil_cues) >= 1: return "ta"
    if sum(c in t for c in hindi_cues) >= 1: return "hi"

    # Plain Latin-script text is treated as English for this request. This is
    # what prevents a previous Tamil/Hindi request from forcing the next English
    # answer into the old language.
    return "en" if any(c.isalpha() for c in text) else fallback


def _has_romanized_indic_cues(text):
    """Return a likely Indic language for clearly romanized speech, otherwise None.

    Native-script detection remains the primary method. These cues only help when
    Saaras returns Latin-script text (for example, Tanglish/Hinglish).
    """
    if not text:
        return None
    t = " " + re.sub(r"[^a-zA-Z]+", " ", text.lower()) + " "
    cues = {
        "ta": (" thanni ", " thanneer ", " mazha ", " mazhai ", " irukku ", " venum ", " enakku ", " naal ", " naatkal ", " vidanum ", " paasanam ", " vivasayi ", " epdi ", " enna ", " sollu "),
        "hi": (" paani ", " baarish ", " barish ", " nahi ", " chahiye ", " mere paas ", " kitna ", " fasal ", " kheti ", " kya ", " kaise ", " batao "),
        "te": (" neellu ", " varsham ", " kavali ", " naaku ", " ela ", " enti "),
        "kn": (" neeru ", " male ", " beku ", " nanage ", " hege ", " enu "),
        "ml": (" vellam ", " mazha ", " venam ", " enikku ", " engane ", " entha "),
        "bn": (" jol ", " brishti ", " chai ", " amar ", " keno ", " ki "),
        "mr": (" pani ", " paus ", " pahije ", " mala ", " kasa ", " kay "),
        "gu": (" pani ", " varsad ", " joie ", " mare ", " kem ", " shu "),
        "pa": (" paani ", " meeh ", " chahida ", " mainu ", " kiven ", " ki "),
        "or": (" pani ", " barsa ", " darkar ", " mote ", " kemiti ", " kana "),
    }
    scores = {code: sum(t.count(cue) for cue in words) for code, words in cues.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 1 else None

# ---------- Orchestrator + Advisor ----------
def orchestrate(q, lang, zones, water_l, budget):
    # Sarvam handles natural-language interpretation when available; the deterministic
    # water engine remains the source of truth for all irrigation quantities.
    detector = llm(
        'Extract a farmer situation as JSON only. Keys: rain_scale, water_l, budget, no_rain_days, temp_delta, language. '
        'Use null when absent. rain_scale=1 means normal forecast; 0.7 means 30% less rain. '
        'language must be one of en,hi,ta,te,kn,ml,bn,mr,gu,pa,or.', q, max_tokens=220)
    p = parse_json(detector) or {}
    # Always run the local parser too. Sarvam remains the richer interpreter,
    # but a temporary API/JSON failure must never make a hypothetical silently
    # revert to the initial dashboard values. Explicit values from the farmer
    # win over the dashboard, and Sarvam values win over the local fallback.
    local = _extract_situation(q)
    # Deterministic values explicitly present in the farmer's sentence ALWAYS win.
    # The LLM may fill gaps, but it can never overwrite a value we can parse exactly.
    # The current utterance decides the response language. The LLM's language
    # field is deliberately NOT allowed to override this: it can misclassify
    # short Tamil/Hindi utterances, while the frontend can pass a reliable STT
    # language_code such as ta-IN/hi-IN. For romanized speech, the fallback
    # from Saaras is retained when the text is ambiguous.
    detected = _script_lang(q, fallback=None)
    romanized = _has_romanized_indic_cues(q)
    if detected and detected != "en":
        # Native Indic script in the current utterance always wins.
        lang = detected
    elif romanized:
        # For Latin-script Hinglish/Tanglish/etc., use strong language cues.
        lang = romanized
    elif detected == "en":
        lang = "en"
    else:
        lang = lang if lang in LANGS else "en"
    rs = float(local.get("rain_scale", p.get("rain_scale") if p.get("rain_scale") is not None else rain_trust()))
    wl = float(local.get("water_l", p.get("water_l") if p.get("water_l") is not None else water_l))
    bd = float(local.get("budget", p.get("budget") if p.get("budget") is not None else budget))
    nr = max(0, min(5, int(local.get("no_rain_days", p.get("no_rain_days") if p.get("no_rain_days") is not None else 0))))
    td = float(local.get("temp_delta", p.get("temp_delta") if p.get("temp_delta") is not None else 0))
    trace = ["Orchestrator: understood farmer situation", "Language: " + lang_name(lang), "Weather agent: forecast for the farm"]
    wx = weather_agent(FARM_LAT, FARM_LON)
    trace.append(f"Hypothetical inputs: water {wl:.0f} L, budget Rs {bd:.0f}, effective cap {min(wl, bd / COST_KL * 1000):.0f} L; rain x{rs:.2f}, {nr} dry days, +{td:.1f}C")
    res = water_agent(zones, wx, wl, bd, rs, nr, td)
    smart_zone_text = ", ".join(f"{k}: {v} L" for k,v in res["scenarios"][2].get("zone_totals", {}).items())
    trace.append(f"Zone water (smart plan): {smart_zone_text}")
    trace.append("Advisor agent: explaining the recalculated plan")
    s = res["scenarios"][2]
    prompt = (f"Question: {q}\nRecalculated plan: {res['plan']}\nSmart plan: {s}\n"
              f"Water used: {s['water_used']} L; water saved: {res['water_saved']} L; risk: {s['risk']}\n"
              f"Weather: {wx['days']}\nRespond in {lang_name(lang)}. Use only supplied numbers. Max 4 sentences.")
    advisor_system = ("You are FarmTwin's agricultural advisor. Explain the already-calculated irrigation result clearly; "
                      "never invent quantities or override the water engine. "
                      f"LANGUAGE_LOCK={lang}. The farmer's current message determines the reply language. "
                      "Return ONLY the answer in that language. Do not translate into English, do not add an English heading, "
                      "and do not mix languages. If LANGUAGE_LOCK=ta, use natural Tamil Unicode script only (தமிழ்), not Tanglish. "
                      "If LANGUAGE_LOCK=hi, use natural Hindi in Devanagari only. If LANGUAGE_LOCK=en, use English only. "
                      "Keep all supplied numbers exactly unchanged.")
    ans = llm(advisor_system, prompt, max_tokens=400)

    # The advisor is allowed to make the explanation natural, but it is NOT
    # allowed to leak English action strings such as "Don't irrigate" into a
    # Hindi/Tamil answer. If the model mixes scripts, discard that answer and
    # use the deterministic fully-localized answer below.
    def _mixed_language_answer(text, target):
        if not text:
            return True
        if target == "hi":
            # Allow the FarmTwin brand and Latin digits/zone IDs, but reject
            # ordinary English words in the farmer-facing answer.
            english_words = re.findall(r"\b(?:Irrigate|Don't|Do|not|today|water|used|risk|Low|Medium|High|Reassess|after|rainfall|plan|Zone|Crop|FarmTwin)\b", text, re.I)
            devanagari = len(re.findall(r"[\u0900-\u097F]", text))
            return len(english_words) >= 2 or devanagari < 8
        if target == "ta":
            english_words = re.findall(r"\b(?:Irrigate|Don't|Do|not|today|water|used|risk|Low|Medium|High|Reassess|after|rainfall|plan|Zone|Crop|FarmTwin)\b", text, re.I)
            tamil = len(re.findall(r"[\u0B80-\u0BFF]", text))
            return len(english_words) >= 2 or tamil < 8
        return False

    if _mixed_language_answer(ans, lang):
        ans = None

    # Fallback must obey the same language contract as the LLM path. Never
    # paste the engine's English plan strings into a Hindi/Tamil response.
    irr_items = [a for a in s.get("day0", []) if a.get("liters")]
    if not ans:
        if lang == "ta":
            risk_ta = {"Low":"குறைவு", "Medium":"நடுத்தரம்", "High":"அதிகம்"}.get(s["risk"], s["risk"])
            if irr_items:
                actions = " மற்றும் ".join(f"{a['zone'].replace('Zone','மண்டலம்')} {a['liters']} லிட்டர்" for a in irr_items)
                action = f"இன்று {actions} பாசனம் செய்யவும்."
            else:
                action = "இன்று பாசனம் தேவையில்லை."
            ans = (f"FarmTwin கணக்கீட்டின்படி, இன்று {s['water_used']} லிட்டர் தண்ணீர் பயன்படுத்தப்படும். "
                   f"பயிர் ஆபத்து {risk_ta}. {action}")
        elif lang == "hi":
            risk_hi = {"Low":"कम", "Medium":"मध्यम", "High":"अधिक"}.get(s["risk"], s["risk"])
            if irr_items:
                actions = " और ".join(f"{a['zone'].replace('Zone','ज़ोन')} में {a['liters']} लीटर" for a in irr_items)
                action = f"आज {actions} पानी दें।"
            else:
                action = "आज सिंचाई की आवश्यकता नहीं है।"
            ans = (f"FarmTwin की गणना के अनुसार आज {s['water_used']} लीटर पानी उपयोग होगा। "
                   f"फसल जोखिम {risk_hi} है। {action}")
        else:
            if irr_items:
                actions = " and ".join(f"{a['zone']} ({a['liters']} L)" for a in irr_items)
                action = f"Irrigate {actions} today."
            else:
                action = "No irrigation is needed today."
            ans = (f"FarmTwin calculates {s['water_used']} L of water for today's plan. "
                   f"Crop risk is {s['risk']}. {action}")
    return dict(answer=ans, detected_lang=lang, params=dict(rain_scale=rs, water_l=wl, budget=bd, no_rain_days=nr, temp_delta=td), trace=trace, **res)


# ---------- Decision agent: structured, explainable output (no LLM needed) ----------
def decision_agent(zones, wx, res, trust):
    sc = {x["key"]: x for x in res["scenarios"]}
    s, w = sc["smart"], sc["wait2"]
    low = [z["name"] for z in zones if z["moisture"] < CROPS[z["crop"]]["thr"]]
    d = wx["days"][0]
    irr = [a for a in s["day0"] if a["liters"]]
    dec = "Irrigate " + ", ".join(f"{a['zone']} ({a['liters']} L)" for a in sorted(irr, key=lambda a: a["zone"])) if irr else "Do not irrigate today"
    reasons = ([f"{', '.join(low)} below the crop moisture threshold"] if low else ["All zones above their crop thresholds"]) + [
        f"Rain forecast {d['rain_mm']} mm ({d['prob']}% chance) in the next day",
        f"Water cap {res['cap_l']} L; plan uses {s['water_used']} L"]
    conf = 0.55 + (0.15 if wx["status"] == "LIVE" else -0.1) + (0.1 if d["prob"] < 30 or d["prob"] > 70 else 0) - abs(trust - 1) * 0.2
    return dict(decision=dec, reason=reasons,
                data_used=["Soil moisture (farmer input)", "Crop thresholds", f"Weather [{wx['status']}] {wx['source']}", "Water and budget caps", "Past rain feedback"],
                resource_impact=f"{s['water_used']} L water, Rs {s['cost']}",
                alternative=f"Wait 2 days: {w['water_used']} L, crop risk {w['risk']}",
                trade_off=f"Saves about {res['water_saved']} L vs irrigating everything today",
                risk=s["risk"], confidence=round(max(0.3, min(0.9, conf)), 2),
                label="Estimated: prototype water-balance calculation, not a measured result")


def agents_report(zones, wx, res, dec):
    s = next(x for x in res["scenarios"] if x["key"] == "smart")
    d = wx["days"][0]
    low = [z["name"] for z in zones if z["moisture"] < CROPS[z["crop"]]["thr"]]
    thr = ", ".join(sorted({z["crop"] + " " + str(CROPS[z["crop"]]["thr"]) + "%" for z in zones}))
    wc = {"LIVE": 0.85, "CACHED": 0.6}.get(wx["status"], 0.3)
    rows = [
        ("Weather", wx["status"], f"Rain {d['rain_mm']} mm ({d['prob']}%), humidity {wx.get('humidity_3d_avg')}%", wx["source"], "Rain and ET0 inputs", wc),
        ("Soil", "FARMER INPUT", f"Below threshold: {', '.join(low) or 'none'}", "Entered by farmer (SIMULATED in demo)", "Starting moisture", 0.5),
        ("Crop", "ESTIMATED", "Stress thresholds " + thr, "Built-in crop table", "Stress limits", None),
        ("Water", "ESTIMATED", f"Plan uses {s['water_used']} L of {res['cap_l']} L cap; zone litres = moisture deficit × {MM_PER_PCT:g} mm/% × area", "Water-balance model", "Litres per zone", None),
        ("Risk", "ESTIMATED", f"Crop risk {s['risk']}", "Simulation", "Flags stress", None),
        ("Simulation", "ESTIMATED", "3 strategies + 5-day dry test", "Water-balance model", "Compares actions", None),
        ("Decision", "ESTIMATED", dec["decision"], "Rules + optimizer", "Final recommendation", dec["confidence"]),
        ("Feedback", "LEARNING", f"Rain trust factor {rain_trust()}", "feedback.json", "Recalibrates rain", None),
        ("Disease / image", "ON DEMAND", "Upload a photo in Crop photo check", "Vision provider", "Photo + weather reasoning", None),
        ("Localization", "READY" if client else "FALLBACK", "Indian-language STT/TTS + translated explanation", "Sarvam", "Detects language and speaks the answer", None)]
    return [dict(name=n, status=a, finding=b, source=c, contribution=e, confidence=f) for n, a, b, c, e, f in rows]


def data_sources():
    on = lambda k: "CONFIGURED" if os.getenv(k) else "NOT CONFIGURED"
    return [
        dict(name="IMD (India Meteorological Department)", data="Forecast", official=True, status=on("IMD_URL"), note="No verified public JSON API found; set IMD_URL if you have one"),
        dict(name="Open-Meteo", data="Forecast, ET0, humidity", official=False, status=_last.get("status", "NOT FETCHED YET"), note="Fallback, last fetched " + _last.get("fetched_at", "never")),
        dict(name="data.gov.in Agmarknet", data="Mandi prices", official=True, status=on("DATA_GOV_IN_KEY"), note="Needs free API key"),
        dict(name="upag.gov.in (via Parse.bot)", data="Mandi prices", official=True,
             status="CONFIGURED" if (PARSEBOT_KEY and PARSEBOT_SCRAPER_ID) else "NOT CONFIGURED",
             note="Needs PARSEBOT_SCRAPER_ID from a Parse.bot scraper built against the upag.gov.in prices page"),
        dict(name="Soil Health Card / ICAR", data="Soil", official=True, status="NOT INTEGRATED", note="No public API found; moisture is farmer-entered")]
