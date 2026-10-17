# Essay Helper

## Local setup

## 1. Download a free local AI model

Open a terminal and run this once:

```powershell
ollama pull llama3.2:3b
```

Ollama must be open while the website is being used. This model runs on your computer and does not use API credits.

## 2. Install the Python packages

Open a terminal in this folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. Start the website

```powershell
python server.py
```

Open http://127.0.0.1:5000 in a browser. Do not open `index.html` directly, because the AI API needs the local Python server.

## What the local AI flow does

1. Reads the handwriting with the local Ollama vision model.
2. Uses `best.pt` when it finds one of the classes your YOLO model was trained to recognize.
3. Sends the extracted essay text to your local Ollama model for Grade 7-friendly suggestions.

## Deploying on Render with hosted AI

Render cannot access Ollama running on your laptop. For hosted essay reading and feedback:

1. Create a Hugging Face access token that has Inference Providers permission.
2. Add it in Render under **Environment** as `HF_TOKEN`. Do not put the token in GitHub.
3. Use `pip install -r requirements-render.txt` as the Render build command. This keeps local-only OCR and YOLO packages off the small hosted service.
4. Use `gunicorn server:app --bind 0.0.0.0:$PORT` as the start command.
5. Commit and push changes to GitHub so Render can deploy them.

When `HF_TOKEN` is set, the app sends the essay page photos and feedback request to Hugging Face's hosted Qwen vision model. Do not upload essays containing names, school names, or other private details. Free inference usage is limited and can change; the app shows an error if the provider limit is reached.

You can upload up to five page photos (10 MB each). The app reads them in the order shown in the preview; use the up/down buttons to fix the order before improving the essay.

The converted text can be edited on the results page. Select **Use corrected text for feedback** to ask the AI for new suggestions based on those corrections.

## Run history

The server saves one history entry for each essay request, including its time, status, page count, detected category (when available), and duration. It does not save the essay text, photo, or filename.

- On Windows, find the log at `%LOCALAPPDATA%\EssayHelper\run-history.jsonl`.
- The terminal running `server.py` also prints each new entry.

Each line is one run. The file stays on this PC between app restarts. It contains run details only, not student writing or uploaded photos.
