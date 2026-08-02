#!/usr/bin/env python3
"""Optional live inference. This file is intentionally excluded from the default run."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROMPT = """The following text describes an emotional experience shared on social media.
The emotion word has been replaced with a <mask> token. Based on the context,
predict the most suitable emotion word to replace <mask>. Provide only the
predicted emotion word, with no additional text or reasoning.

Text:
{segment}

Answer:
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional API-based emotion inference")
    parser.add_argument("--provider", required=True, choices=("openai", "gemini", "together"))
    parser.add_argument("--model-id", required=True, help="Exact provider model identifier")
    parser.add_argument("--input", type=Path, default=Path("/data/express-single-mask.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=25)
    return parser.parse_args()


def require_environment_key(provider: str) -> None:
    choices = {
        "openai": ("OPENAI_API_KEY",),
        "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "together": ("TOGETHER_API_KEY",),
    }[provider]
    if not any(os.getenv(name) for name in choices):
        raise RuntimeError(
            f"Missing credential. Attach a Code Ocean secret as one of: {', '.join(choices)}"
        )


def make_client(provider: str):
    if provider == "openai":
        from openai import OpenAI

        return OpenAI()
    if provider == "gemini":
        from google import genai

        return genai.Client()
    from together import Together

    return Together()


def predict(client, provider: str, model_id: str, prompt: str) -> tuple[str, float | None]:
    if provider == "gemini":
        response = client.models.generate_content(model=model_id, contents=prompt)
        candidate = response.candidates[0] if response.candidates else None
        avg_logprob = getattr(candidate, "avg_logprobs", None)
        token_count = getattr(getattr(response, "usage_metadata", None), "candidates_token_count", None)
        probability = math.exp(avg_logprob * token_count) if avg_logprob is not None and token_count else None
        return response.text.strip(), probability

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        logprobs=True if provider == "openai" else 1,
    )
    choice = response.choices[0]
    content = getattr(choice.logprobs, "content", None)
    if content is not None:
        token_logprobs = [token.logprob for token in content if token.logprob is not None]
    else:
        token_logprobs = [value for value in (choice.logprobs.token_logprobs or []) if value is not None]
    probability = math.exp(sum(token_logprobs)) if token_logprobs else None
    return choice.message.content.strip(), probability


def main() -> None:
    args = parse_args()
    require_environment_key(args.provider)
    client = make_client(args.provider)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.output) if args.output.exists() else pd.read_csv(args.input)
    if "output" not in frame:
        frame["output"] = None
    if "probability" not in frame:
        frame["probability"] = None
    pending = frame.index[frame["output"].isna()].tolist()
    if args.max_rows is not None:
        pending = pending[: args.max_rows]

    for count, index in enumerate(tqdm(pending, desc="Inference"), start=1):
        try:
            output, probability = predict(
                client, args.provider, args.model_id, PROMPT.format(segment=frame.at[index, "segment"])
            )
            frame.at[index, "output"] = output
            frame.at[index, "probability"] = probability
        except Exception as error:
            print(f"Row {index} failed: {type(error).__name__}: {error}")
        if count % args.save_interval == 0:
            frame.to_csv(args.output, index=False)

    frame.to_csv(args.output, index=False)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

