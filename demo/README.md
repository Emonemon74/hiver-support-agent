# Demo app

A Streamlit front end over the agent (`src/agent/pipeline.py`) — type a customer
tweet, see intent, drafted reply, and the auto/escalate decision with a reason.

**This is a portfolio demo, not a production tool.** See the in-app disclaimer.

## Run it locally

```bash
cd hiver-support-agent
source .venv/bin/activate        # or: uv venv --python 3.11 && uv pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_..." >> .env   # free key: console.groq.com
streamlit run demo/app.py
```
Opens at `http://localhost:8501`. First run rebuilds the retrieval index from
`data/corpus.parquet` (~10s); after that it's cached.

## Deploy for free (Hugging Face Spaces)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space) (free
   account, no card). Name it, pick **SDK: Streamlit**, visibility: Public.
2. It gives you a git remote like
   `https://huggingface.co/spaces/<you>/<space-name>`. Add it and push this repo:
   ```bash
   cd hiver-support-agent
   git remote add space https://huggingface.co/spaces/<you>/<space-name>
   git push space main
   ```
3. Open the Space → **Settings → Variables and secrets** → add secret
   `GROQ_API_KEY` (free, from console.groq.com). Never put it in code or README.
4. **Settings → App file** → set it to `demo/app.py` (Spaces defaults to
   `app.py` at the repo root; this repo's app lives in `demo/`).
5. The Space rebuilds automatically (installs `requirements.txt`, ~2–3 min).
   Once it's live, the URL is `https://huggingface.co/spaces/<you>/<space-name>`
   — that's the link to put on a resume/LinkedIn.

## Notes

- `MAX_REQUESTS_PER_SESSION` in `app.py` caps runs per browser session (default
  12) so one visitor can't exhaust the shared free Groq quota (200k tokens/day).
  Raise it if you add a paid key later.
- The retrieval index rebuilds from `data/corpus.parquet` (committed, ~1.8 MB) on
  first load; `data/corpus_vecs.npy` is NOT committed (it's a cache) so it
  regenerates once per Space restart.
- Anything the visitor types is sent to Groq's API (per Groq's own privacy
  terms) — don't advertise this as a place to paste real PII.
