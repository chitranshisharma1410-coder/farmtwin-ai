FARMTWIN - FULL MULTI-LANGUAGE + LOCATION BUILD

1. Put your Sarvam API key in a file named .env beside main.py:
   SARVAM_API_KEY=YOUR_REAL_SARVAM_KEY

   Do NOT put the key inside index.html or commit/share the .env file.

2. Install dependencies:
   python -m pip install -r requirements.txt

3. Start:
   python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
   OR double-click run_farmtwin.bat

4. Open:
   http://127.0.0.1:8000

VOICE / LANGUAGE
- The microphone sends language_code=unknown so each recording is auto-detected.
- English speech -> English answer + English TTS.
- Hindi speech -> Hindi answer + Hindi TTS.
- Tamil speech -> Tamil answer + Tamil TTS.
- Also supports Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi and Odia.

LOCATION
- The dashboard no longer depends on one fixed farm location.
- Use "Use my location" to let the browser locate the farm.
- Or enter District + State and press "Set district".
- You can change the district at any time; weather, simulation and vision use the selected coordinates.
- Location is saved in this browser using localStorage.
- District lookup uses OpenStreetMap Nominatim.

IF YOU SEE "SARVAM_API_KEY IS NOT CONFIGURED"
- Confirm the file is exactly named .env (not .env.txt).
- Confirm .env is in the same folder as main.py.
- Confirm the line is exactly: SARVAM_API_KEY=your_real_key
- Stop Uvicorn with Ctrl+C and start it again after changing .env.
- The dashboard shows a Sarvam API status message near Farm Location.
