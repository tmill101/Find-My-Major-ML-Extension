"""
File: fetch_onet_questions.py
Path: src/scripts/fetch_onet_questions.py

Purpose:
  CLI script to fetch O*NET Interest Profiler questions and save them locally.
  Loads credentials from a .env file, calls the API, and persists the result.

Original Author(s):
  - Claude Code
  - Nathan Meiners

AI Tools Used:
  - Claude Code - Initial implementation
  - GitHub Copilot - Documentation

Editors:
  - Claude Code (2026-02-16) — Initial implementation
  - AI Assistant (2026-04-20) — Added file header and relocated from scripts/ to src/scripts/

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-20

Assumptions & Constraints:
  - Requires a .env file with ONET_API_KEY
  - Must be run from the project root directory
  - Network access to O*NET API required

Related Docs:
  - docs/src/data.md
  - docs/data/data_governance.md
"""

from src.data.fetch_questions import fetch_questions
from src.data.store import save_questions


def fetch_and_save_questions(start: int = 1, end: int = 60) -> None:
    """
    Name: fetch_and_save_questions

    Purpose:
      Fetches O*NET Interest Profiler questions and saves them to disk.

    Inputs:
      - start: int — first question number (default 1)
      - end: int — last question number (default 60)

    Outputs:
      - Saves questions to data/raw/interest_profiler_questions.json

    Raises / Errors:
      - ValueError: if API key not found in environment
      - requests.RequestException: if API call fails
      - FileNotFoundError: if output directory cannot be created

    Notes:
      - Questions are persisted as immutable JSON after successful fetch
    """
    print(f"Fetching O*NET Interest Profiler questions {start}-{end}...")
    question_set = fetch_questions(api_key=None, start=start, end=end)
    save_questions(question_set)
    print(f"Successfully saved {len(question_set.questions)} questions")


if __name__ == "__main__":
    fetch_and_save_questions()
