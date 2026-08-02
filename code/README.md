# Camera-ready reproducibility code

This folder replaces the exploratory notebooks with a headless, ordered, and
reviewer-facing workflow for Code Ocean.

## Files

- `run`: Code Ocean entry point.
- `main.py`: offline VAD mapping, EBS calculations, statistical tests, emotion
  vocabulary coverage, error analysis, and figure generation.
- `model_inference.py`: optional API inference with credentials read only from
  environment variables. It is deliberately not part of the default run.
- `model_manifest.json`: exact historical provider/model identifiers used to
  generate the archived outputs.
- `requirements.txt`: exact environment for the default reproducible run.
- `requirements-inference.txt`: extra packages needed only for optional API inference.

## Canonical data inputs

Place these three study data files in Code Ocean `/data`:

- `/data/emotion-lexicons.pkl`
- `/data/express-single-mask.csv`
- `/data/NRC-VAD-Lexicon.txt`

The five archived model outputs are computational inputs to the offline
analysis. Attach them under `/data/raw-results`:

- `deepseek-results.csv`
- `gemini-results.csv`
- `gpt-results.csv`
- `llama-results.csv`
- `qwen-results.csv`

Keeping the archived outputs makes the paper reproducible without paid API
credentials or dependence on changing hosted model versions. They remain
clearly separate from the three canonical study data files.

## Code Ocean setup

1. Upload this folder's contents to `/code`.
2. Add the exact packages in `requirements.txt` to a Python 3.11 environment.
3. Upload or attach the three canonical data files and archived model outputs.
4. Set `run` as the file to run.
5. Start a Reproducible Run and inspect the generated `/results` files.

## Generated results

- model-specific and combined processed CSV files;
- overall and topic-level VAD t-tests;
- model accuracy and VAD coverage summary;
- emotion-term coverage and unmatched-term tables;
- thresholded error-pair analysis; and
- camera-ready PDF and PNG figures.

## Optional inference

`model_inference.py` requires an exact provider model ID and reads credentials
from `OPENAI_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, or
`TOGETHER_API_KEY`. Add credentials through Code Ocean secrets. Never paste a
credential into a source file.

Example:

```bash
python model_inference.py \
  --provider openai \
  --model-id YOUR_EXACT_MODEL_ID \
  --output /results/raw-results/gpt-results.csv
```

Live inference is not guaranteed to reproduce historical results because APIs
and hosted model versions can change.
