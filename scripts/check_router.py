"""Quick sanity check: which LLM slots are actually live given current env keys.

Run in the same environment where the API keys live (locally with .env exported,
or as a step inside the GitHub Actions workflow). Exits non-zero if GROQ_API_KEY_B
is missing so CI can fail loudly instead of silently degrading.
"""

import os
import sys

from src.core.llm_router import build_bulk_router, build_llm_router


def main() -> int:
    full = build_llm_router()
    bulk = build_bulk_router()

    print("=== Main cascade (build_llm_router) — live slots ===")
    for a in full.accounts:
        print(f"  {a.display_name}  (rpd={a.rpd})")
    print(f"  total_daily_quota = {full.total_daily_quota} RPD\n")

    print("=== Bulk router (build_bulk_router) — live slots ===")
    for a in bulk.accounts:
        print(f"  {a.display_name}  (rpd={a.rpd})")
    print(f"  total_daily_quota = {bulk.total_daily_quota} RPD\n")

    key_b = bool(os.environ.get("GROQ_API_KEY_B"))
    print(f"GROQ_API_KEY_B set? {'YES' if key_b else 'NO'}")
    if not key_b:
        print("WARNING: slots ③/④ and the 2nd bulk slot are silently disabled.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
