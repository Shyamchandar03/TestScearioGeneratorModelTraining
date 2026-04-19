"""
============================================================
STEP 4: INFERENCE - Run on New User Stories
============================================================
Usage:
  python 5_inference.py --story "As a user, I want to..."
  python 5_inference.py --file user_stories.txt
  python 5_inference.py --serve    # Start FastAPI server
============================================================
"""

import os
import sys
import json
import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Optional

# Avoid Windows console UnicodeEncodeError when printing emojis/box-drawing chars.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────
MODEL_PATH = "models/flan_t5_test_gen"    # Path to fine-tuned model
FALLBACK_MODEL = "google/flan-t5-base"    # Fallback if fine-tuned not available
APP_VERSION = "2026-04-18"


# ──────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ──────────────────────────────────────────────────────────
class TestScenarioGenerator:
    def __init__(self, model_path: str = MODEL_PATH):
        model_to_load = model_path if os.path.exists(model_path) else FALLBACK_MODEL
        if model_to_load == FALLBACK_MODEL:
            print(f"⚠️  Using fallback model: {FALLBACK_MODEL}")
            print("   Run training first for best results.\n")

        print(f"📥 Loading model: {model_to_load}")
        self.model_name = model_to_load
        self.tokenizer = T5Tokenizer.from_pretrained(model_to_load)
        self.model = T5ForConditionalGeneration.from_pretrained(model_to_load)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✅ Model ready on {self.device.upper()}\n")

        self._bad_words_ids = self._build_bad_words_ids()

    def _build_bad_words_ids(self) -> list:
        blocked_phrases = ["User:", "Assistant:", "System:"]
        bad_words_ids = []
        for phrase in blocked_phrases:
            ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            if ids:
                bad_words_ids.append(ids)
        return bad_words_ids

    def _looks_like_garbage_output(self, text: str) -> bool:
        import re

        t = (text or "").strip()
        if not t:
            return True
        if any(x in t for x in ["User:", "Assistant:", "System:"]):
            return True
        if re.fullmatch(r"(\\d+[\\.\\)]\\s*)+", t):
            return True
        return False

    def _normalize_for_dedup(self, s: str) -> str:
        import re

        s = (s or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-z0-9\s]", "", s)
        return s

    def _template_candidates(self, user_story: str) -> list:
        story = (user_story or "").lower()
        candidates = []

        if any(k in story for k in ["login", "log in", "sign in", "signin"]):
            candidates += [
                "Valid username and password → user is logged in and redirected to the home/dashboard page",
                "Invalid password → show a generic 'Invalid credentials' message (no user enumeration)",
                "Unregistered username/email → show a generic 'Invalid credentials' message",
                "Empty username or password → inline validation error and login is blocked",
                "Account locked after N failed attempts → show lockout message and enforce cooldown/rate limit",
                "Password masked while typing → characters are hidden and can be toggled to show/hide",
                "Remember me enabled → session persists across browser restart (as per requirements)",
                "Session timeout → user is logged out after inactivity and prompted to log in again",
                "SQL injection payload in username field → input is rejected/sanitized; no server error/leakage",
                "Login completes within 2 seconds under normal load → meets performance SLA",
            ]
        elif any(k in story for k in ["reset password", "forgot password", "password reset"]):
            candidates += [
                "Registered email submitted → reset email sent; success message shown",
                "Unregistered email submitted → generic success message (no user enumeration)",
                "Reset link clicked → opens reset form and allows setting a new password",
                "Expired reset link → show 'Link expired' and allow requesting a new link",
                "Weak password rejected → show complexity requirements",
                "Reset link used twice → second attempt is rejected",
                "Rate limit reset requests → too many requests triggers cooldown",
                "Reset completes within 2 seconds → meets performance SLA",
            ]
        else:
            # Generic fallbacks when we don't recognize the feature.
            candidates += [
                "Happy path → valid input completes successfully and confirmation is shown",
                "Invalid input → clear validation error shown and no state is changed",
                "Boundary values → min/max/empty inputs handled correctly",
                "Unauthorized user → access is denied (403/redirect to login)",
                "Security → injection/XSS payload is handled safely with no leakage",
                "Concurrency → repeated/duplicate submissions do not create duplicate records",
                "Resilience → network timeout/retry does not corrupt state",
                "Performance → request completes within target SLA under normal load",
            ]

        return candidates

    def _ensure_scenario_count(self, user_story: str, scenarios: list, desired: int) -> list:
        desired = int(desired) if desired else 0
        if desired <= 0:
            return scenarios

        dedup = set(self._normalize_for_dedup(s) for s in scenarios)
        filled = list(scenarios)
        for cand in self._template_candidates(user_story):
            if len(filled) >= desired:
                break
            norm = self._normalize_for_dedup(cand)
            if not norm or norm in dedup:
                continue
            filled.append(cand)
            dedup.add(norm)

        return filled[:desired]

    def generate(
        self,
        user_story: str,
        num_scenarios: int = 10,
        creativity: float = 0.3,   # (0=deterministic, 1=creative)
    ) -> dict:
        """
        Generate test scenarios for a given user story.

        Returns dict with:
          - scenarios: list of individual test scenario strings
          - raw_output: full generated text
          - metadata: stats about the generation
        """
        # Build prompt
        input_text = (
            "Generate comprehensive test scenarios for the following user story:\n\n"
            f"{user_story.strip()}\n\n"
            "Test Scenarios:"
        )

        max_new_tokens = max(128, min(800, int(num_scenarios) * 80)) if num_scenarios else 600

        def generate_once(prompt: str, *, force_sample: bool) -> str:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                do_sample = force_sample or (creativity >= 0.7)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    do_sample=do_sample,
                    temperature=max(0.8, creativity) if do_sample else 1.0,
                    top_p=0.9 if do_sample else 1.0,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.15,
                    bad_words_ids=self._bad_words_ids or None,
                )

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        raw_output = generate_once(input_text, force_sample=False)

        # Parse into individual scenarios
        scenarios = self._parse_scenarios(raw_output)

        desired = int(num_scenarios) if num_scenarios else 10
        min_accept = 1 if desired <= 1 else min(3, desired)
        if self._looks_like_garbage_output(raw_output) or len(scenarios) < min_accept:
            retry_prompt = (
                "Generate comprehensive test scenarios for the following user story.\n"
                "Return ONLY a numbered list (1., 2., 3., ...) with one scenario per line.\n\n"
                f"{user_story.strip()}\n\n"
                "Test Scenarios:\n1."
            )
            raw_output = generate_once(retry_prompt, force_sample=True)
            scenarios = self._parse_scenarios(raw_output)

        if num_scenarios and len(scenarios) > num_scenarios:
            scenarios = scenarios[:num_scenarios]
        if num_scenarios and len(scenarios) < num_scenarios:
            scenarios = self._ensure_scenario_count(user_story, scenarios, num_scenarios)

        scenarios_numbered = self._to_numbered_list(scenarios)

        return {
            "user_story": user_story,
            "scenarios": scenarios,
            "raw_output": raw_output,
            "scenarios_numbered": scenarios_numbered,
            "metadata": {
                "scenario_count": len(scenarios),
                "coverage": self._analyze_coverage(scenarios),
                "format": "numbered_list",
            }
        }

    def _to_numbered_list(self, scenarios: list) -> str:
        return "\n".join([f"{i}. {s}" for i, s in enumerate(scenarios, 1)])

    def _parse_scenarios(self, text: str) -> list:
        """Extract individual scenarios from numbered list output."""
        import re
        text = (text or "").strip()
        if not text:
            return []

        # Handle cases where the model outputs "1. ... 2. ... 3. ..." on a single line.
        matches = list(re.finditer(r"(\d+)[\.\)]\s+", text))
        if len(matches) >= 2:
            scenarios = []
            for i, m in enumerate(matches):
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                item = text[start:end].strip()
                if item:
                    scenarios.append(item)
            if scenarios:
                return scenarios

        lines = text.split("\n")
        scenarios = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Match lines starting with number
            match = re.match(r"^(\d+)[\.\)]\s+(.+)", line)
            if match:
                scenarios.append(match.group(2).strip())
            elif re.match(r"^[-*•]\s+(.+)", line):
                scenarios.append(re.sub(r"^[-*•]\s+", "", line).strip())
            elif scenarios:
                # Continuation of previous scenario
                scenarios[-1] += " " + line

        # Fallback: return non-empty lines
        if not scenarios:
            scenarios = [l.strip() for l in lines if len(l.strip()) > 15]

        # If we still only have one long blob, try splitting into sentences.
        if len(scenarios) == 1:
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", scenarios[0]) if p.strip()]
            parts = [p for p in parts if len(p) > 15]
            if len(parts) > 1:
                scenarios = parts

        return scenarios

    def _analyze_coverage(self, scenarios: list) -> dict:
        """Quick coverage analysis of generated scenarios."""
        all_text = " ".join(scenarios).lower()

        coverage = {
            "happy_path": any(w in all_text for w in ["valid", "success", "correct", "redirects"]),
            "negative_test": any(w in all_text for w in ["invalid", "error", "incorrect", "fail"]),
            "edge_case": any(w in all_text for w in ["empty", "maximum", "minimum", "limit"]),
            "security": any(w in all_text for w in ["injection", "https", "password", "unauthorized"]),
            "performance": any(w in all_text for w in ["time", "seconds", "load", "response"]),
        }
        covered = sum(coverage.values())
        coverage["coverage_percent"] = round(covered / len(coverage) * 100)
        return coverage

    def format_output(self, result: dict, style: str = "pretty") -> str:
        """Format output for display."""
        if style == "json":
            return json.dumps(result, indent=2)

        if style == "plain":
            return result["raw_output"]

        # Pretty format
        lines = [
            "=" * 65,
            "🧪 GENERATED TEST SCENARIOS",
            "=" * 65,
            f"📖 User Story:",
            f"   {result['user_story']}",
            "",
            f"✅ {result['metadata']['scenario_count']} Test Scenarios Generated:",
            "-" * 65,
        ]
        for i, scenario in enumerate(result["scenarios"], 1):
            lines.append(f"  {i:>2}. {scenario}")

        lines += [
            "",
            "📊 Coverage Analysis:",
            "-" * 65,
        ]
        for category, covered in result["metadata"]["coverage"].items():
            if category == "coverage_percent":
                lines.append(f"     Overall Coverage: {covered}%")
            else:
                icon = "✅" if covered else "❌"
                lines.append(f"     {icon} {category.replace('_', ' ').title()}")

        lines.append("=" * 65)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────
# BATCH PROCESSING
# ──────────────────────────────────────────────────────────
def process_file(generator: TestScenarioGenerator, filepath: str):
    """Process a text file with one user story per line."""
    with open(filepath) as f:
        stories = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    print(f"📄 Processing {len(stories)} user stories from {filepath}\n")
    results = []

    for i, story in enumerate(stories, 1):
        print(f"[{i}/{len(stories)}] Generating for: {story[:60]}...")
        result = generator.generate(story)
        results.append(result)
        print(generator.format_output(result))
        print()

    # Save to JSON
    output_path = filepath.replace(".txt", "_test_scenarios.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {output_path}")


# ──────────────────────────────────────────────────────────
# FASTAPI SERVER (optional)
# ──────────────────────────────────────────────────────────
def start_api_server():
    """Start a FastAPI endpoint for serving predictions."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import PlainTextResponse
        from pydantic import BaseModel
        import uvicorn
    except ImportError:
        print("Install FastAPI: pip install fastapi uvicorn")
        return

    app = FastAPI(title="Test Scenario Generator API", version=APP_VERSION)
    generator = TestScenarioGenerator()

    class StoryRequest(BaseModel):
        user_story: str
        creativity: float = 0.3
        num_scenarios: int = 10

    class ScenarioResponse(BaseModel):
        user_story: str
        scenarios: list
        scenarios_numbered: str
        raw_output: str
        metadata: dict

    @app.post("/generate", response_model=ScenarioResponse)
    async def generate_scenarios(req: StoryRequest):
        result = generator.generate(
            req.user_story,
            num_scenarios=req.num_scenarios,
            creativity=req.creativity,
        )
        return result

    @app.post("/generate/plain", response_class=PlainTextResponse)
    async def generate_scenarios_plain(req: StoryRequest):
        result = generator.generate(
            req.user_story,
            num_scenarios=req.num_scenarios,
            creativity=req.creativity,
        )
        return result["scenarios_numbered"] or result["raw_output"]

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": generator.model_name, "app_version": APP_VERSION}

    print("🚀 Starting API server at http://localhost:8000")
    print("   POST /generate        → {user_story: '...', creativity: 0.7, num_scenarios: 10}")
    print("   POST /generate/plain  → same payload, returns plain text")
    print("   GET  /health    → health check")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ──────────────────────────────────────────────────────────
# MAIN DEMO
# ──────────────────────────────────────────────────────────
DEMO_STORIES = [
    (
        "As a registered user, I want to update my email address in profile settings "
        "so that I can receive communications at my current email."
    ),
    (
        "As a mobile app user, I want to enable biometric authentication "
        "so that I can log in quickly without typing my password."
    ),
    (
        "As an e-commerce customer, I want to apply a discount coupon at checkout "
        "so that I can save money on my purchase."
    ),
]


def main():
    parser = argparse.ArgumentParser(description="Generate test scenarios from user stories")
    parser.add_argument("--story", type=str, help="Single user story string")
    parser.add_argument("--file", type=str, help="Text file with user stories (one per line)")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--output", choices=["pretty", "json", "plain"], default="pretty")
    args = parser.parse_args()

    if args.serve:
        start_api_server()
        return

    generator = TestScenarioGenerator()

    if args.story:
        result = generator.generate(args.story)
        print(generator.format_output(result, style=args.output))

    elif args.file:
        process_file(generator, args.file)

    else:
        # Run demo
        print("🎯 Running demo on sample user stories...\n")
        for story in DEMO_STORIES:
            result = generator.generate(story)
            print(generator.format_output(result))
            print()


if __name__ == "__main__":
    main()
