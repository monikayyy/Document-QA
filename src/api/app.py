from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from src.service import DocumentQAService

app = FastAPI(title="Document Q&A API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
service: DocumentQAService | None = None


def _get_service() -> DocumentQAService:
    global service
    if service is None:
        service = DocumentQAService()
    return service


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Document Q&A</title>
    <style>
      :root {
        --bg: #f4f6f8;
        --panel: #ffffff;
        --ink: #12212f;
        --muted: #5f6f7d;
        --accent: #007a6e;
      }
      body {
        margin: 0;
        font-family: "Avenir Next", "Segoe UI", sans-serif;
        color: var(--ink);
        background: radial-gradient(circle at top, #d9f1ec 0%, var(--bg) 55%);
      }
      main {
        max-width: 880px;
        margin: 30px auto;
        padding: 0 16px;
      }
      .card {
        background: var(--panel);
        border-radius: 14px;
        box-shadow: 0 8px 20px rgba(18, 33, 47, 0.08);
        padding: 18px;
        margin-bottom: 16px;
      }
      h1 { margin: 0 0 8px; }
      p { margin: 0 0 12px; color: var(--muted); }
      label { display: block; margin: 10px 0 6px; font-weight: 600; }
      input, textarea, button {
        width: 100%;
        box-sizing: border-box;
        border-radius: 8px;
        border: 1px solid #d0d8de;
        padding: 10px 12px;
        font-size: 14px;
      }
      textarea { min-height: 88px; resize: vertical; }
      button {
        margin-top: 12px;
        background: var(--accent);
        color: white;
        border: none;
        font-weight: 700;
        cursor: pointer;
      }
      pre {
        background: #0f1720;
        color: #d9e2ec;
        padding: 12px;
        border-radius: 8px;
        overflow: auto;
      }
    </style>
  </head>
  <body>
    <main>
      <div class="card">
        <h1>Document Q&A</h1>
        <p>Ingest documents, then ask grounded questions with citations.</p>
      </div>

      <div class="card">
        <h2>Ingest</h2>
        <label>Paths (comma separated)</label>
        <input id="paths" value="data/raw" />
        <label>Corpus ID</label>
        <input id="ingestCorpus" value="default" />
        <button onclick="runIngest()">Run Ingest</button>
      </div>

      <div class="card">
        <h2>Ask</h2>
        <label>Corpus ID</label>
        <input id="askCorpus" value="default" />
        <label>Question</label>
        <textarea id="question">What is this corpus about?</textarea>
        <button onclick="runAsk()">Ask</button>
      </div>

      <div class="card">
        <h2>Output</h2>
        <pre id="out">Ready.</pre>
      </div>
    </main>

    <script>
      const out = document.getElementById("out");
      function show(data) {
        out.textContent = typeof data === "string" ? data : JSON.stringify(data, null, 2);
      }
      async function runIngest() {
        try {
          const paths = document.getElementById("paths").value
            .split(",")
            .map(v => v.trim())
            .filter(Boolean);
          const corpus_id = document.getElementById("ingestCorpus").value.trim() || "default";
          const resp = await fetch("/ingest", {
            method: "POST",
            headers: {"content-type": "application/json"},
            body: JSON.stringify({paths, corpus_id, rebuild: false})
          });
          show(await resp.json());
        } catch (e) {
          show(String(e));
        }
      }
      async function runAsk() {
        try {
          const corpus_id = document.getElementById("askCorpus").value.trim() || "default";
          const question = document.getElementById("question").value.trim();
          const resp = await fetch("/ask", {
            method: "POST",
            headers: {"content-type": "application/json"},
            body: JSON.stringify({question, corpus_id, top_k: 4})
          });
          show(await resp.json());
        } catch (e) {
          show(String(e));
        }
      }
    </script>
  </body>
</html>
"""


@app.get("/health")
def health() -> dict:
    stats = _get_service().stats("default")
    return {"status": "ok", "default_index_exists": stats["index_exists"]}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if not req.paths:
        raise HTTPException(status_code=400, detail="At least one path is required")
    try:
        result = _get_service().ingest(paths=req.paths, corpus_id=req.corpus_id, rebuild=req.rebuild)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ingest failed: {type(exc).__name__}") from exc
    return IngestResponse(**result.__dict__)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = _get_service().ask(question=req.question, corpus_id=req.corpus_id, top_k=req.top_k)
        return AskResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ask failed: {type(exc).__name__}") from exc
