# CinePulse AI - how to get the .exe

This zip is the **source project**. The Windows `.exe` has to be built on a Windows machine, which
the sandbox I work in is not. You have two ways to get it. **Option A needs nothing installed on
your laptop** - GitHub's free Windows computers do the build for you.

What is already in here: all the code, the face model (`models/blaze_face_short_range.tflite`),
`build_exe.py` (the PyInstaller build), a ready-made GitHub build workflow, and ~170 tests.
`ffmpeg.exe` and the Whisper speech model are downloaded automatically during the build.

---------------------------------------------------------------------------
## OPTION A - GitHub builds it (no installs, works from a slow laptop)

Time: ~15 min of your clicking + 30-60 min of waiting. Needs only a web browser.

1. **Create a free GitHub account** at https://github.com/signup (skip if you have one).
2. **Create a private repository**: top-right "+" -> *New repository* -> name `clipstudio`
   -> choose **Private** -> *Create repository*.
   (Private keeps your code hidden and artifacts private. Free accounts get 2,000 Actions minutes a
   month; Windows minutes count double, and one build is ~60-100 of them.)
3. **Upload the project**: unzip this zip on your laptop. In the new repo click
   *uploading an existing file*, then drag in everything from inside the unzipped folder
   (folders `backend`, `frontend`, `models`, `tests`, `bin`, `.github`, and the files `launcher.py`,
   `build.py`, `build_exe.py`, `requirements-build.txt`, ...). Click *Commit changes*.
   - The browser allows ~100 files per upload; if it complains, upload in two batches
     (e.g. first `tests`, then the rest).
   - **Check the workflow arrived**: the repo must show `.github/workflows/build-windows-exe.yml`.
     Folders starting with a dot are sometimes skipped by drag-and-drop. If it is missing:
     *Add file -> Create new file*, type the name `.github/workflows/build-windows-exe.yml`
     (typing the slashes creates the folders), paste in the contents of that file from the zip, commit.
4. **Run the build**: open the **Actions** tab (click "I understand my workflows, go ahead and enable
   them" if shown) -> click **Build Windows exe** on the left -> **Run workflow** -> green **Run workflow**.
5. **Wait 30-60 minutes.** Click the running job to watch. It installs everything, downloads ffmpeg and
   the models, builds with PyInstaller, then **starts the finished exe and checks it answers**
   (the smoke test) before saving anything.
6. **Download**: when the run is green, scroll to **Artifacts** at the bottom of the run page ->
   click **CinePulseAI-Portable**. GitHub wraps it once more, so you unzip twice:
   download -> unzip -> inside is `CinePulseAI-Portable.zip` -> unzip that -> folder `CinePulseAI`.
7. **Run**: open the `CinePulseAI` folder, double-click **CinePulseAI.exe**. Wait 10-40 seconds on
   first start; your browser opens the app. If Windows SmartScreen warns about an unknown publisher:
   *More info -> Run anyway* (it is your own program).
8. **If the run turns red**: open the failed step, copy the last ~40 lines of its log and send them to me.
   Packaging streamlit + mediapipe often needs one or two rounds of fixes - that is normal, and I can't
   run the build myself, so I will need that log.

### Adding Google Drive backup (optional - the app works without it)
Without any Google files the app runs normally and saves clips in the `outputs` folder next to the exe.
To turn Drive backup on, **no Python is needed**:
1. Make a Google Cloud OAuth client (once, ~10 min):
   - https://console.cloud.google.com/ -> new project `CinePulse AI`.
   - *APIs & Services -> Library* -> **Google Drive API** -> Enable.
   - *OAuth consent screen* -> External -> fill app name + your email -> add yourself as a Test user
     -> then click **Publish app** (otherwise Google signs you out every 7 days).
   - *Credentials -> Create credentials -> OAuth client ID -> Desktop app* -> Download JSON.
2. Rename the file to exactly `credentials.json` and put it in the `CinePulseAI` folder next to `CinePulseAI.exe`.
3. In Drive create a folder for clips; open it and copy the ID from the address bar
   (`drive.google.com/drive/folders/`**`THIS_PART`**).
4. Start CinePulseAI.exe, paste the ID in the sidebar, make a clip. The first upload opens a Google
   sign-in page ("unverified app" is normal: *Advanced -> Go to CinePulse AI*). After you approve,
   `token.json` is created in that folder and later uploads need no sign-in.
5. If the upload says the folder was "not found": your Google permission is too narrow to see a folder
   you made by hand. Delete `token.json`, and ask me for the `--full-access` option for the packaged app
   (or run `python backend\uploader.py --auth --full-access` on any PC that has Python and copy the new token.json over).
6. Optional - bake your token into the build so the exe works with zero sign-in: in the repo,
   *Settings -> Secrets and variables -> Actions -> New repository secret*, name `GOOGLE_TOKEN_JSON`,
   value = the full text of a `token.json` you created. **Anyone holding the built folder can then use
   your Drive access** - never share such a build. Revoke any time at
   https://myaccount.google.com/permissions

---------------------------------------------------------------------------
## OPTION B - build on your own Windows PC

Needs a PC that can handle ~5 GB of downloads and a long build.

1. Install **Python 3.12 (64-bit)** from python.org; tick *Add python.exe to PATH*.
2. Unzip to a short path such as `C:\CinePulseAI`, then in PowerShell:
   ```
   cd C:\CinePulseAI
   python -m venv .venv
   .venv\Scripts\Activate.ps1          (if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
   python -m pip install --upgrade pip
   pip install -r requirements-build.txt
   ```
3. Preview the build (changes nothing):  `python build_exe.py --dry-run`
4. Build:  `python build_exe.py`
   It downloads `ffmpeg.exe`, the face model and the Whisper model automatically if missing, then writes
   `dist\CinePulseAI\` and `dist\CinePulseAI-Portable.zip`.
   - Use `--no-token` to leave `token.json` out. `--no-download` stops all downloads (you must then
     supply `bin\ffmpeg.exe` yourself).
5. Run `dist\CinePulseAI\CinePulseAI.exe` (or send the zip to any Windows laptop).

To run the test-suite too: `pip install pytest playwright`, `playwright install chromium`, then
`python -m unittest discover -s tests -t . -p "test_*.py"`. (Some tests are written for Linux/macOS and
may fail on Windows - they do not affect the build.)

---------------------------------------------------------------------------
## Things to know
- **Offline vs online:** transcription, highlight scoring, face-tracked cropping and merging work with no
  internet. YouTube downloads, edge-tts voices and Google Drive upload need internet.
- **Without Google files** the app still launches and works; clips just stay local and the app tells you
  Drive backup is off.
- **If face tracking can't run**, the app falls back to a centre crop and says so.
- **Face tracking** works best with the speaker's face clearly visible (talking-head framing).
- **No console window:** stop the app with *Quit CinePulse AI* in the sidebar; logs are in `cinepulseai.log`.
- **YouTube downloads** depend on yt-dlp staying current; rebuild occasionally to get the newest one.
- The two ffmpeg *source code* archives you sent earlier cannot be used; the build downloads a ready-made
  Windows ffmpeg instead.


## AI Avatar lip-sync (AI Generator Lab)

In the **AI Generator Lab** tab, tick **Enable AI Avatar Lip-Sync Commentary** and upload a `.jpg`/`.png`
picture of your avatar. The generated video then shows that picture talking in time with the voiceover.

- It runs on the CPU only (no GPU, no API key). A 30-second video takes roughly 30-60 seconds on a laptop.
- A clear, front-facing face (photo or flat illustration) gives real mouth movement: the jaw opens and closes
  with the voice and shuts between words (word timing comes from the bundled Whisper model).
- If no face is found, the whole picture does a gentle talking pulse instead, and you are told so.
- If you also add a background video, the avatar appears as a rounded card over it.
- This is 2-D image warping, not a neural talking-head model, so the motion is stylised and subtle.
