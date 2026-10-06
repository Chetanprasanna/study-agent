"""
tests/test_part2.py  —  Part 2

Pytest tests for:
  - agent/retriever.py  (NotesIndex)
  - agent/llm.py        (ask_llm)   — mocked, no real API calls
  - agent/qa.py         (answer_question) — mocked, no real API calls

All tests here must pass WITHOUT a GEMINI_API_KEY set.
Real API integration is verified manually via scripts/try_qa.py.

Run with:
    pytest tests/test_part2.py -v
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
from unittest.mock import patch, MagicMock   # Tools for replacing real functions with fakes

# ── Third-party ───────────────────────────────────────────────────────────────
import pytest

# ── Modules under test ────────────────────────────────────────────────────────
from agent.retriever import NotesIndex
from agent.llm import ask_llm, MissingApiKeyError
from agent.qa import answer_question


# ═══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_chunks() -> list[dict]:
    """
    A realistic set of chunk dicts (same format as chunk_pages() output).
    Covers multiple pages and different topics so we can test retrieval ranking.
    """
    return [
        {
            "id": 0,
            "page": 1,
            "text": (
                "Photosynthesis is the process by which green plants convert sunlight "
                "into chemical energy. Chlorophyll in the chloroplasts absorbs light "
                "and uses it to convert carbon dioxide and water into glucose and oxygen."
            ),
        },
        {
            "id": 1,
            "page": 1,
            "text": (
                "The light-dependent reactions occur in the thylakoid membranes. "
                "ATP and NADPH are produced here, and oxygen is released as a byproduct "
                "of splitting water molecules."
            ),
        },
        {
            "id": 2,
            "page": 2,
            "text": (
                "Cellular respiration is the process cells use to break down glucose "
                "and produce ATP. The three stages are glycolysis, the Krebs cycle, "
                "and oxidative phosphorylation."
            ),
        },
        {
            "id": 3,
            "page": 2,
            "text": (
                "The mitochondria is the site of cellular respiration. "
                "The inner mitochondrial membrane contains the electron transport chain "
                "which drives ATP synthesis via ATP synthase."
            ),
        },
        {
            "id": 4,
            "page": 3,
            "text": (
                "DNA replication occurs before cell division. The double helix is "
                "unwound by helicase, and new strands are synthesized by DNA polymerase "
                "in the 5' to 3' direction."
            ),
        },
    ]


@pytest.fixture
def index(sample_chunks) -> NotesIndex:
    """Build a NotesIndex from the sample chunks. Reused by many tests."""
    return NotesIndex(sample_chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for NotesIndex (retriever.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNotesIndex:
    """Tests for agent.retriever.NotesIndex"""

    # ── Construction tests ────────────────────────────────────────────────────

    def test_builds_without_error(self, sample_chunks):
        """NotesIndex should construct without raising any exception."""
        idx = NotesIndex(sample_chunks)   # Should not raise
        assert idx is not None

    def test_builds_with_empty_chunks(self):
        """NotesIndex([]) should not crash — empty index is valid."""
        idx = NotesIndex([])
        assert idx is not None

    def test_single_chunk(self):
        """NotesIndex with a single chunk should work fine."""
        chunks = [{"id": 0, "page": 1, "text": "Only one chunk here."}]
        idx = NotesIndex(chunks)
        results = idx.search("chunk")
        assert len(results) <= 1

    # ── search() return type tests ────────────────────────────────────────────

    def test_search_returns_list(self, index):
        """search() must always return a list."""
        result = index.search("photosynthesis")
        assert isinstance(result, list)

    def test_search_returns_dicts(self, index):
        """Every item in search results must be a dict."""
        results = index.search("photosynthesis")
        for item in results:
            assert isinstance(item, dict)

    def test_search_result_has_required_keys(self, index):
        """Each result dict must have 'id', 'page', 'text', and 'score'."""
        results = index.search("photosynthesis")
        assert len(results) > 0, "Expected at least one result for 'photosynthesis'"
        for item in results:
            assert "id" in item,    "Missing 'id'"
            assert "page" in item,  "Missing 'page'"
            assert "text" in item,  "Missing 'text'"
            assert "score" in item, "Missing 'score' — retriever must add this"

    def test_score_is_float(self, index):
        """'score' must be a float (not int or string)."""
        results = index.search("photosynthesis")
        for item in results:
            assert isinstance(item["score"], float), f"score should be float, got {type(item['score'])}"

    def test_score_range(self, index):
        """TF-IDF cosine similarity scores must be in [0.0, 1.0]."""
        results = index.search("photosynthesis")
        for item in results:
            assert 0.0 <= item["score"] <= 1.0, f"Score out of range: {item['score']}"

    # ── Ranking tests ─────────────────────────────────────────────────────────

    def test_results_sorted_descending(self, index):
        """Results must be sorted by score from highest to lowest."""
        results = index.search("photosynthesis")
        scores = [item["score"] for item in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted descending by score"

    def test_best_match_is_relevant(self, index):
        """
        The top result for 'photosynthesis' should be from chunk 0 or 1
        (both mention photosynthesis), NOT from the DNA or respiration chunks.
        """
        results = index.search("photosynthesis")
        assert len(results) > 0
        top_result = results[0]
        # The top result's text should mention photosynthesis
        assert "photosynthesis" in top_result["text"].lower() or \
               "chlorophyll" in top_result["text"].lower(), \
               "Top result for 'photosynthesis' query should be about photosynthesis"

    def test_respiration_query_matches_respiration_chunk(self, index):
        """Querying 'cellular respiration ATP' should surface the respiration chunks."""
        results = index.search("cellular respiration ATP")
        assert len(results) > 0
        top_text = results[0]["text"].lower()
        assert "respiration" in top_text or "atp" in top_text or "mitochondria" in top_text

    def test_dna_query_matches_dna_chunk(self, index):
        """Querying 'DNA replication polymerase' should surface the DNA chunk."""
        results = index.search("DNA replication polymerase")
        assert len(results) > 0
        top_text = results[0]["text"].lower()
        assert "dna" in top_text or "replication" in top_text or "polymerase" in top_text

    # ── top_k tests ───────────────────────────────────────────────────────────

    def test_top_k_respected(self, index):
        """search() should return at most top_k results."""
        results = index.search("biology", top_k=2)
        assert len(results) <= 2

    def test_top_k_1(self, index):
        """top_k=1 should return at most 1 result."""
        results = index.search("photosynthesis", top_k=1)
        assert len(results) <= 1

    def test_top_k_larger_than_chunks(self, index):
        """If top_k > number of chunks, return at most len(chunks) results."""
        results = index.search("biology", top_k=1000)
        assert len(results) <= 5  # We only have 5 sample chunks

    def test_default_top_k_is_4(self, index):
        """Default top_k should be 4 (per the contract)."""
        results = index.search("biology")   # no top_k argument
        assert len(results) <= 4

    # ── Edge case tests ───────────────────────────────────────────────────────

    def test_empty_index_returns_empty(self):
        """An index built from [] should return [] for any query."""
        idx = NotesIndex([])
        assert idx.search("anything") == []

    def test_empty_query_returns_empty(self, index):
        """An empty query string should return []."""
        assert index.search("") == []
        assert index.search("   ") == []

    def test_search_does_not_mutate_original_chunks(self, sample_chunks, index):
        """
        Calling search() should NOT add a 'score' field to the original chunk dicts.
        The retriever must return COPIES, not references to the originals.
        """
        # Run a search
        index.search("photosynthesis")

        # Original chunks should not have a 'score' key
        for chunk in sample_chunks:
            assert "score" not in chunk, \
                "search() must not mutate the original chunk dicts — return copies!"

    def test_unknown_query_returns_empty_or_low_score(self, index):
        """
        A query with words completely absent from the notes should return
        either an empty list or results with very low scores.
        """
        results = index.search("xyzzy quux frobnicate")
        # Either no results (all scores 0.0 were filtered out) or scores are very low
        for item in results:
            assert item["score"] < 0.5, "Unknown query should have low score"

    def test_id_and_page_preserved_correctly(self, sample_chunks):
        """The id and page from the original chunk must appear in search results."""
        idx = NotesIndex(sample_chunks)
        results = idx.search("photosynthesis", top_k=1)
        if results:
            top = results[0]
            # Find the original chunk with this id
            original = next(c for c in sample_chunks if c["id"] == top["id"])
            assert top["page"] == original["page"], "page must match original chunk"
            assert top["text"] == original["text"], "text must match original chunk"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for ask_llm (llm.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAskLlm:
    """
    Tests for agent.llm.ask_llm

    We mock the Gemini client so no real API calls are made.
    These tests validate the logic of ask_llm(), not the Gemini model itself.
    """

    def test_raises_missing_key_error_when_no_key(self):
        """
        ask_llm() must raise MissingApiKeyError when GEMINI_API_KEY is not set,
        NOT a generic error or a crash on import.
        """
        # Temporarily remove GEMINI_API_KEY from the environment
        with patch.dict(os.environ, {}, clear=True):
            # Make sure the key really is gone
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(MissingApiKeyError):
                ask_llm("Hello?")

    def test_returns_string_on_success(self):
        """ask_llm() must return a plain string (the model's response text)."""
        # Build a fake response object that looks like what google-genai returns
        fake_response = MagicMock()
        fake_response.text = "  This is the model's answer.  "

        # Patch genai.Client so no real HTTP call is made
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key-for-testing"}):
            with patch("agent.llm.genai.Client") as mock_client_class:
                # Set up the mock: client.models.generate_content() → fake_response
                mock_client = MagicMock()
                mock_client.models.generate_content.return_value = fake_response
                mock_client_class.return_value = mock_client

                result = ask_llm("What is 2+2?")

        # The result should be a string, stripped of whitespace
        assert isinstance(result, str)
        assert result == "This is the model's answer."

    def test_passes_prompt_to_api(self):
        """ask_llm() must pass the prompt to the Gemini API as 'contents'."""
        fake_response = MagicMock()
        fake_response.text = "Answer"

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            with patch("agent.llm.genai.Client") as mock_client_class:
                mock_client = MagicMock()
                mock_client.models.generate_content.return_value = fake_response
                mock_client_class.return_value = mock_client

                test_prompt = "Explain mitosis in simple terms."
                ask_llm(test_prompt)

                # Check that generate_content was called with our prompt
                call_kwargs = mock_client.models.generate_content.call_args
                # The prompt must appear in the call arguments
                assert test_prompt in str(call_kwargs), \
                    "The prompt must be passed to generate_content()"

    def test_retries_on_transient_error(self):
        """
        ask_llm() should retry up to _MAX_RETRIES times when the API raises an error.
        On the third attempt (after 2 retries), it should succeed.
        """
        fake_response = MagicMock()
        fake_response.text = "Success after retry"

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            with patch("agent.llm.genai.Client") as mock_client_class:
                mock_client = MagicMock()
                # First two calls raise an error, third succeeds
                mock_client.models.generate_content.side_effect = [
                    ConnectionError("timeout"),
                    ConnectionError("timeout"),
                    fake_response,
                ]
                mock_client_class.return_value = mock_client

                # Patch sleep so tests don't actually wait
                with patch("agent.llm.time.sleep"):
                    result = ask_llm("test prompt")

                assert result == "Success after retry"
                # generate_content should have been called 3 times total
                assert mock_client.models.generate_content.call_count == 3

    def test_raises_after_all_retries_exhausted(self):
        """
        If all retries fail, ask_llm() must raise RuntimeError (not silently return None).
        """
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
            with patch("agent.llm.genai.Client") as mock_client_class:
                mock_client = MagicMock()
                # Every call fails
                mock_client.models.generate_content.side_effect = ConnectionError("always fails")
                mock_client_class.return_value = mock_client

                with patch("agent.llm.time.sleep"):
                    with pytest.raises(RuntimeError, match="failed after"):
                        ask_llm("test prompt")

    def test_module_importable_without_key(self):
        """
        Importing agent.llm must NOT raise even if GEMINI_API_KEY is absent.
        This is a hard contract requirement.
        """
        # If we reach this point without ImportError, the test passes.
        # The import already happened at the top of this file.
        import agent.llm  # noqa: F401
        assert True


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for answer_question (qa.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnswerQuestion:
    """
    Tests for agent.qa.answer_question

    We mock ask_llm so no real Gemini calls happen.
    We use real NotesIndex so retrieval is tested end-to-end.
    """

    @pytest.fixture
    def mock_ask_llm(self):
        """
        Pytest fixture that replaces ask_llm in the qa module with a fake
        that always returns a predetermined string.
        """
        with patch("agent.qa.ask_llm", return_value="Photosynthesis converts sunlight to glucose. (Page 1)") as mock:
            yield mock

    # ── Return structure tests ────────────────────────────────────────────────

    def test_returns_dict(self, index, mock_ask_llm):
        """answer_question() must return a dict."""
        result = answer_question("What is photosynthesis?", index)
        assert isinstance(result, dict)

    def test_has_answer_key(self, index, mock_ask_llm):
        """Result must have an 'answer' key."""
        result = answer_question("What is photosynthesis?", index)
        assert "answer" in result

    def test_has_sources_key(self, index, mock_ask_llm):
        """Result must have a 'sources' key."""
        result = answer_question("What is photosynthesis?", index)
        assert "sources" in result

    def test_answer_is_string(self, index, mock_ask_llm):
        """'answer' must be a string."""
        result = answer_question("What is photosynthesis?", index)
        assert isinstance(result["answer"], str)

    def test_sources_is_list(self, index, mock_ask_llm):
        """'sources' must be a list."""
        result = answer_question("What is photosynthesis?", index)
        assert isinstance(result["sources"], list)

    def test_sources_contain_score(self, index, mock_ask_llm):
        """Each source dict must have a 'score' key (from the retriever)."""
        result = answer_question("What is photosynthesis?", index)
        for source in result["sources"]:
            assert "score" in source, "Source must include 'score' from retriever"

    # ── Retrieval integration tests ───────────────────────────────────────────

    def test_sources_are_relevant(self, index, mock_ask_llm):
        """
        The sources for a photosynthesis question should come from the
        photosynthesis chunks (page 1), not from the DNA or respiration chunks.
        """
        result = answer_question("What is photosynthesis?", index)
        assert len(result["sources"]) > 0
        # Top source should mention photosynthesis
        top_source_text = result["sources"][0]["text"].lower()
        assert "photosynthesis" in top_source_text or "chlorophyll" in top_source_text

    def test_ask_llm_is_called(self, index, mock_ask_llm):
        """ask_llm must be called once per answer_question() invocation."""
        answer_question("What is photosynthesis?", index)
        mock_ask_llm.assert_called_once()

    def test_prompt_contains_question(self, index):
        """The prompt sent to ask_llm must include the student's question."""
        captured_prompt = {}

        def fake_llm(prompt, system=None):
            captured_prompt["value"] = prompt
            return "Fake answer"

        with patch("agent.qa.ask_llm", side_effect=fake_llm):
            answer_question("What is DNA replication?", index)

        assert "What is DNA replication?" in captured_prompt["value"], \
            "The question must appear in the prompt sent to Gemini"

    def test_prompt_contains_context(self, index):
        """The prompt must include the retrieved chunk text as context."""
        captured_prompt = {}

        def fake_llm(prompt, system=None):
            captured_prompt["value"] = prompt
            return "Fake answer"

        with patch("agent.qa.ask_llm", side_effect=fake_llm):
            answer_question("What is photosynthesis?", index)

        # The prompt must contain text from the notes (not just the question)
        assert "Page" in captured_prompt["value"], \
            "Prompt should contain page references from retrieved chunks"

    # ── Empty index edge case ─────────────────────────────────────────────────

    def test_empty_index_returns_not_found(self):
        """
        If the index has no chunks, answer_question should return the
        'not found' answer without calling ask_llm at all.
        """
        empty_index = NotesIndex([])

        with patch("agent.qa.ask_llm") as mock_llm:
            result = answer_question("What is anything?", empty_index)

        # Should NOT have called the LLM (no point asking with no context)
        mock_llm.assert_not_called()
        assert "answer" in result
        assert "couldn't find" in result["answer"].lower() or result["answer"] == ""
        assert result["sources"] == []

    # ── Answer content test ───────────────────────────────────────────────────

    def test_answer_comes_from_llm(self, index):
        """The 'answer' field must be exactly what ask_llm returns."""
        expected_answer = "The unique answer from the LLM goes here."
        with patch("agent.qa.ask_llm", return_value=expected_answer):
            result = answer_question("What is photosynthesis?", index)
        assert result["answer"] == expected_answer
