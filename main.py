import pathlib, sqlite3, json, time, base64
from typing import List, Optional
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import agents as A

app = FastAPI(title="FarmTwin")
db = sqlite3.connect("farmtwin.db", check_same_thread=False)
db.execute("create table if not exists log(id integer primary key, ts text, kind text, payload text)")


def log(kind, p):
    db.execute("insert into log(ts,kind,payload) values(?,?,?)", (time.strftime("%F %T"), kind, json.dumps(p, default=str)))
    db.commit()


class Zone(BaseModel):
    name: str
    crop: str
    area: float      # m2
    moisture: float  # %


class Farm(BaseModel):
    zones: List[Zone]
    water_l: float = 0
    budget: float = 2000
    rain_scale: Optional[float] = None
    no_rain_days: int = 0
    temp_delta: float = 0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    district: str = ""
    state: str = ""
    location_label: str = ""


class Ask(Farm):
    question: str
    lang: str = "en"


class Photo(Farm):
    image: str
    mime: str = "image/jpeg"
    lang: str = "en"


class Feedback(BaseModel):
    predicted_rain_mm: float
    actual_rain_mm: float
    condition: Optional[str] = None


def zl(f):
    return [z.model_dump() for z in f.zones]


@app.get("/api/config")
def config():
    return {"sarvam_configured": bool(A.SARVAM_KEY), "location": {
        "label": A.FARM_LOCATION_LABEL, "lat": A.FARM_LAT, "lon": A.FARM_LON
    }}


@app.post("/api/location/district")
def location_district(payload: dict):
    return A.geocode_location(payload.get("district", ""), payload.get("state", ""), payload.get("country", "India"))


@app.post("/api/location/coordinates")
def location_coordinates(payload: dict):
    try:
        lat = float(payload.get("latitude"))
        lon = float(payload.get("longitude"))
    except (TypeError, ValueError):
        return {"error": "Invalid coordinates"}
    return A.reverse_geocode(lat, lon)


@app.post("/api/simulate")
def simulate(f: Farm):
    wx = A.weather_agent(f.latitude if f.latitude is not None else A.FARM_LAT, f.longitude if f.longitude is not None else A.FARM_LON)
    trust = A.rain_trust()
    res = A.water_agent(zl(f), wx, f.water_l, f.budget, f.rain_scale or trust, f.no_rain_days, f.temp_delta)
    dec = A.decision_agent(zl(f), wx, res, trust)
    out = dict(weather=wx, rain_trust=trust, market=A.market_agent(f.zones[0].crop), decision=dec,
               agents=A.agents_report(zl(f), wx, res, dec), **res)
    log("simulation", dict(farm=f.model_dump(), decision=dec, scenarios=res["scenarios"]))
    return out


@app.post("/api/ask")
def ask(a: Ask):
    return A.orchestrate(a.question, a.lang, zl(a), a.water_l, a.budget)


@app.post("/api/transcribe")
def transcribe(payload: dict):
    try:
        raw = payload.get("audio", "")
        if not raw:
            return {"error": "No audio supplied"}
        if "," in raw:
            raw = raw.split(",", 1)[1]
        audio = base64.b64decode(raw)
        return A.sarvam_transcribe_audio(
            audio,
            payload.get("filename", "voice.webm"),
            payload.get("mime", "audio/webm"),
            payload.get("language_code", "unknown"),
        )
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/tts")
def tts(payload: dict):
    try:
        raw_lang = (payload.get("language_code") or "en-IN").strip().lower()
        # Accept either "ta", "ta-IN", etc. from the browser. Keep TTS language
        # explicit so the audio engine never silently falls back to Tamil.
        lang_map = {
            "en": "en-IN", "ta": "ta-IN", "hi": "hi-IN", "te": "te-IN",
            "kn": "kn-IN", "ml": "ml-IN", "bn": "bn-IN", "mr": "mr-IN",
            "gu": "gu-IN", "pa": "pa-IN", "or": "od-IN",
        }
        language_code = lang_map.get(raw_lang, raw_lang)
        audio = A.sarvam_tts(
            payload.get("text", ""),
            language_code,
            payload.get("speaker", "shubh"),
        )
        return {"audio": audio, "mime": "audio/wav"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/vision")
def vision(p: Photo):
    wx = A.weather_agent(p.latitude if p.latitude is not None else A.FARM_LAT, p.longitude if p.longitude is not None else A.FARM_LON)
    return A.vision_agent(p.image, p.mime, wx, zl(p), p.lang)


@app.post("/api/feedback")
def feedback(fb: Feedback):
    r = A.log_feedback(fb.model_dump())
    log("feedback", dict(fb.model_dump(), **r))
    return r


@app.get("/api/data-sources")
def sources():
    return A.data_sources()


app.mount("/", StaticFiles(directory=pathlib.Path(__file__).parent, html=True), name="ui")