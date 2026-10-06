"""
agent/llm.py  —  Part 2

Thin wrapper around the Google Gemini API.

Why a wrapper?
  Every other module (qa.py, generators.py, agent.py) calls ask_llm() instead of
  talking to Gemini directly. This means:
    - Only ONE place needs to change if we switch models or API versions.
    - Easy to mock in tests (just replace ask_llm with a fake function).
    - Retries, timeouts and error handling live here — callers don't worry about it.

CONTRACT:
    ask_llm(prompt: str, system: str | None = None) -> str

IMPORTANT: This module MUST be importable even if GEMINI_API_KEY is not set.
           The key is only read inside ask_llm() when it is actually called.

This file is complete (Part 2). Do not modify the public function signature.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os       # os.environ lets us read environment variables
import time     # time.sleep() pauses execution for the retry backoff
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

# ── Third-party: python-dotenv ────────────────────────────────────────────────
# load_dotenv() reads the .env file and injects its contents into os.environ.
# This must be called at module-load time so the key is available immediately.
from dotenv import load_dotenv

# ── Third-party: google-genai ─────────────────────────────────────────────────
# We import the google.genai module itself — the actual Client is created
# inside ask_llm() so the import never crashes even without a key.
import google.genai as genai
import google.genai.types as genai_types

# ── Load .env file ────────────────────────────────────────────────────────────
# This reads GEMINI_API_KEY (and any other vars) from `.env` into os.environ.
# If `.env` doesn't exist, this is a silent no-op — no error.
load_dotenv()


# ── Custom exception ──────────────────────────────────────────────────────────
class MissingApiKeyError(RuntimeError):
    """
    Raised when ask_llm() is called but GEMINI_API_KEY is not set.

    We use a custom exception (instead of a bare RuntimeError) so callers
    can catch specifically this case and show a friendly UI message.
    """
    pass


# ── Constants ─────────────────────────────────────────────────────────────────
# Use the requested Gemini 3.5 Flash-Lite model for Q&A and generators.
# Change this one constant if you ever want to use a different model.
_DEFAULT_MODEL: str = "gemini-3.5-flash-lite"

# Retry settings — how many times to retry on transient errors, and how long
# to wait between retries (in seconds). We double the wait each time (backoff).
_MAX_RETRIES: int = 2
_INITIAL_BACKOFF_SECONDS: float = 1.0

# A browser-entered key belongs to one request, not the whole Python process.
# ContextVar keeps concurrent Streamlit sessions isolated without changing APIs.
_SESSION_API_KEY: ContextVar[str] = ContextVar("gemini_session_api_key", default="")


def get_api_key() -> str:
    """Prefer the configured environment key, then this request's fallback."""
    return os.environ.get("GEMINI_API_KEY", "").strip() or _SESSION_API_KEY.get()


@contextmanager
def use_api_key(api_key: str) -> Iterator[None]:
    """Temporarily provide a session key to both routing and text generation."""
    token = _SESSION_API_KEY.set(api_key.strip())
    try:
        yield
    finally:
        # Reset even after an exception so a later request cannot reuse this key.
        _SESSION_API_KEY.reset(token)


def ask_llm(prompt: str, system: str | None = None) -> str:
    """
    Send a prompt to Google Gemini and return its text response.

    This function:
      1. Reads GEMINI_API_KEY from the environment (loaded from .env).
      2. Creates a Gemini client and sends the prompt.
      3. Retries up to 2 times on transient network/API errors.
      4. Returns the model's reply as a plain string.

    Parameters
    ----------
    prompt : str
        The main instruction / question to send to the model.
        Example: "Summarise the following notes in 3 bullet points: ..."
    system : str | None, optional
        An optional system instruction that sets the model's overall behaviour.
        Example: "You are a helpful study assistant. Always cite page numbers."
        If None, no system instruction is sent.

    Returns
    -------
    str
        The model's text response, stripped of leading/trailing whitespace.

    Raises
    ------
    MissingApiKeyError
        If GEMINI_API_KEY is not set in the environment or .env file.
    RuntimeError
        If the API call fails after all retries are exhausted.

    Examples
    --------
    >>> answer = ask_llm("What is 2 + 2?")
    >>> isinstance(answer, str)
    True
    """

    # ── Step 1: Read and validate the API key ─────────────────────────────────
    # We read from os.environ (which includes anything loaded by load_dotenv above).
    api_key: str = get_api_key()

    if not api_key:
        # The key is missing. We raise our custom exception so the caller
        # (e.g. Streamlit UI) can catch it and show a nice error message.
        raise MissingApiKeyError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key. "
            "Get one at: https://aistudio.google.com/app/apikey"
        )

    # ── Step 2: Build the Gemini client ───────────────────────────────────────
    # We create the client inside this function (not at module level) so that
    # importing llm.py never crashes when the key is missing.
    client = genai.Client(api_key=api_key)

    # ── Step 3: Build the generation config (contents + system instruction) ───
    # The google-genai SDK uses a structured "contents" list.
    # Each item in contents is a message, tagged as "user" or "model".
    # Here we have a single user turn.

    # Build the config object that carries the system instruction (if any)
    config_kwargs: dict = {}
    if system:
        # system_instruction tells the model how to behave across the whole conversation.
        config_kwargs["system_instruction"] = system

    generation_config = genai_types.GenerateContentConfig(**config_kwargs)

    # ── Step 4: Call the API with retry logic ─────────────────────────────────
    # Transient errors (network timeouts, 503s) are common with cloud APIs.
    # We retry up to _MAX_RETRIES times, waiting longer each time (exponential backoff).
    last_exception: Exception | None = None
    wait: float = _INITIAL_BACKOFF_SECONDS

    for attempt in range(_MAX_RETRIES + 1):   # +1 because attempt 0 is the first try
        try:
            # Send the request to Gemini.
            # generate_content() is synchronous — it blocks until the model responds.
            response = client.models.generate_content(
                model=_DEFAULT_MODEL,
                contents=prompt,            # The user's prompt string
                config=generation_config,   # Optional system instruction + settings
            )

            # response.text is the model's reply as a plain string.
            # .strip() removes any leading/trailing whitespace/newlines.
            return response.text.strip()

        except Exception as exc:
            # Something went wrong. Log the attempt and decide whether to retry.
            last_exception = exc

            if attempt < _MAX_RETRIES:
                # Not our last attempt — wait, then try again.
                # We print a warning so developers can see what's happening.
                print(
                    f"[llm] API call failed (attempt {attempt + 1}/{_MAX_RETRIES + 1}): "
                    f"{exc}. Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                wait *= 2   # Exponential backoff: 1s → 2s → 4s ...
            # If this was the last attempt, the loop ends and we raise below.

    # ── Step 5: All retries exhausted — raise a descriptive error ─────────────
    raise RuntimeError(
        f"Gemini API call failed after {_MAX_RETRIES + 1} attempts. "
        f"Last error: {last_exception}"
    ) from last_exception
