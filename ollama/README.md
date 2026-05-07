# Phishing Detection Benchmark — Ollama + Promptfoo

Evaluate how well local Ollama models detect phishing emails using a labeled dataset of 10,000 samples.

---

## 1. Install Ollama

Download and install Ollama from **https://ollama.com/download** for your OS (Windows, macOS, or Linux).

After installing, verify it works:
```bash
ollama --version
```

Ollama runs a local server at `http://localhost:11434` by default. On most platforms it starts automatically; if not, run:
```bash
ollama serve
```

---

## 2. Pull Models

Download each model you want to test. Each model is several GB, so pull only what you need:

```bash
ollama pull llama3.2           # Meta Llama 3.2 (~2GB)
ollama pull mistral            # Mistral 7B (~4GB)
ollama pull phi3               # Microsoft Phi-3 (~2GB)
ollama pull gemma2             # Google Gemma 2 (~5GB)
```

To see all downloaded models:
```bash
ollama list
```

The name shown in `ollama list` is the exact name you need to use in the config script.

---

## 3. Register Models in the Script

Open [generate_config.py](generate_config.py) and edit the `OLLAMA_MODELS` list at the top:

```python
OLLAMA_MODELS = [
    "ollama:chat:llama3.2",   # matches `ollama list` name
    "ollama:chat:mistral",
    "ollama:chat:phi3",
    "ollama:chat:gemma2",
]
```

**Format:** `ollama:chat:<modelname>` where `<modelname>` must exactly match what `ollama list` shows.

To test with only one model (e.g., while others are still downloading), just keep one entry in the list.

---

## 4. Generate the Promptfoo Config

Install the Python dependency, then run the script to produce `promptfoo.yaml`:

```bash
pip install -r requirements.txt
python generate_config.py
```

This samples 5 phishing + 5 legitimate emails from the dataset, builds 10 test cases with correct-answer assertions, and writes `promptfoo.yaml`.

---

## 5. Run the Benchmark

Install promptfoo (requires Node.js 18+):
```bash
npm install -g promptfoo
```

Or run without installing globally:

```bash
npx promptfoo@latest eval
```

From this folder:
```bash
promptfoo eval
```

Open the results dashboard:
```bash
promptfoo view
```

---

## What the Benchmark Tests

Each model is prompted with 10 emails and must:
1. Start its response with `phishing` or `legitimate`
2. If phishing, name the correct category from:
   `authority_scam`, `credential_harvesting`, `financial_scam`, `generic_phishing`,
   `romance_dating`, `social_engineering`, `social_engineering_advanced`,
   `tech_support`, `threats`, `urgency`

Promptfoo checks each response against these expectations and reports a pass/fail table per model.

**How grading works:**
- **Classification** — the response (after stripping leading punctuation/quotes) must *start with* `phishing` or `legitimate`. This avoids false positives like "not phishing, this is legitimate".
- **Category** — checked with a word-boundary regex, so `social_engineering` does **not** match `social_engineering_advanced`.

> **Note:** Models that use spaces instead of underscores in category names (e.g., "social engineering" vs. `social_engineering`) will fail the category check. If you see unexpected failures, open `promptfoo view` and inspect the raw model output.
