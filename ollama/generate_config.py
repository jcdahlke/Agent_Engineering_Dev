"""Generate a promptfoo config that benchmarks Ollama models on phishing detection.

Reads the labeled email dataset, samples a balanced set of phishing + legitimate
emails, and writes `promptfoo.yaml` containing:
  - One prompt template that prepends classification instructions to each email
  - One test case per sampled email, with an assertion that grades the response

Run:
    python generate_config.py
    cd ollama && promptfoo eval
"""

import csv
import random
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Configuration — edit these to change models or sample counts
# ---------------------------------------------------------------------------

# Promptfoo provider IDs. Format: "ollama:chat:<modelname>" where <modelname>
# is exactly what `ollama list` shows on your machine. Add or remove entries
# freely — the generated config will run every model against every test.
OLLAMA_MODELS = [
    "ollama:chat:llama3.2",
    "ollama:chat:phi3.5",
    "ollama:chat:gemma2:2b",
    "ollama:chat:Matheswaran/email-classifier",
]

NUM_PHISHING = 5       # number of phishing emails to sample
NUM_LEGITIMATE = 5     # number of legitimate emails to sample
RANDOM_SEED = 42       # set to None for non-deterministic sampling

DATASET_PATH = Path(__file__).parent / "phishing_legit_dataset_KD_10000.csv"
OUTPUT_PATH = Path(__file__).parent / "promptfooconfig.yaml"

# ---------------------------------------------------------------------------

# Promptfoo uses Nunjucks templating, so {{email_text}} is filled in per test.
PROMPT_TEMPLATE = (
    "You are an email classification agent. You will be given an email, and you must "
    "determine if the email is phishing or legitimate. Start the response by saying either "
    "'phishing' or 'legitimate'. Here is the email:\n\n{{email_text}}"
)


def load_and_sample() -> list[dict]:
    """Load the CSV and return a balanced, shuffled sample of email rows."""
    if not DATASET_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {DATASET_PATH}")

    phishing_rows: list[dict] = []
    legit_rows: list[dict] = []

    with open(DATASET_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] == "1":
                phishing_rows.append(row)
            else:
                legit_rows.append(row)

    if NUM_PHISHING > len(phishing_rows):
        sys.exit(f"ERROR: NUM_PHISHING={NUM_PHISHING} but only {len(phishing_rows)} phishing rows available")
    if NUM_LEGITIMATE > len(legit_rows):
        sys.exit(f"ERROR: NUM_LEGITIMATE={NUM_LEGITIMATE} but only {len(legit_rows)} legitimate rows available")

    rng = random.Random(RANDOM_SEED)
    phishing_samples = rng.sample(phishing_rows, NUM_PHISHING)
    legit_samples = rng.sample(legit_rows, NUM_LEGITIMATE)

    all_samples = phishing_samples + legit_samples
    rng.shuffle(all_samples)
    return all_samples


def build_test(row: dict) -> dict:
    """Build one promptfoo test case (vars + assertions) from a dataset row.

    Uses a JavaScript assertion instead of icontains because the check must
    verify the response *starts with* the correct label — plain icontains gives
    false positives when the model says e.g. "not phishing, this is legitimate".
    """
    is_phishing = row["label"] == "1"
    expected_label = "phishing" if is_phishing else "legitimate"

    return {
        "description": f"{'phishing' if is_phishing else 'legitimate'} | severity={row['severity']} confidence={row['confidence']}",
        "vars": {"email_text": row["text"].strip()},
        "assert": [
            {
                "type": "javascript",
                # Strip leading punctuation/quotes/whitespace, then check the
                # first word matches the expected classification.
                "value": (
                    f'output.trim().toLowerCase().replace(/^[^a-z]+/, "")'
                    f'.startsWith("{expected_label}")'
                ),
            }
        ],
    }


def main() -> None:
    samples = load_and_sample()
    tests = [build_test(row) for row in samples]

    config = {
        "description": "Phishing email detection benchmark across Ollama models",
        "providers": OLLAMA_MODELS,
        "prompts": [PROMPT_TEMPLATE],
        "tests": tests,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False, width=120)

    phishing_count = sum(1 for t in tests if t["description"].startswith("phishing"))
    legit_count = len(tests) - phishing_count

    print(f"Generated: {OUTPUT_PATH}")
    print(f"  {phishing_count} phishing samples")
    print(f"  {legit_count} legitimate samples")
    print(f"  {len(OLLAMA_MODELS)} models: {', '.join(OLLAMA_MODELS)}")
    print()
    print("Next steps:")
    print("  cd ollama")
    print("  promptfoo eval")
    print("  promptfoo view")


if __name__ == "__main__":
    main()
