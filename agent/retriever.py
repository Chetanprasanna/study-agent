"""
agent/retriever.py  —  Part 2

Builds an in-memory search index over the text chunks produced by chunker.py,
then lets you query it with plain English to find the most relevant chunks.

How does TF-IDF work (beginner explanation)?
  TF  = Term Frequency    — how often a word appears in a chunk
  IDF = Inverse Document Frequency — how rare the word is across ALL chunks
  TF-IDF score is high when a word appears a lot in ONE chunk but rarely elsewhere.
  This makes common words like "the", "is", "and" score near-zero, while
  subject-specific words like "mitochondria" score high in the right chunk.

  cosine_similarity then measures the "angle" between the query vector and each
  chunk vector. An angle of 0° → score 1.0 (identical). 90° → score 0.0 (unrelated).

CONTRACT:
    class NotesIndex:
        __init__(self, chunks: list[dict])
        search(self, query: str, top_k: int = 4) -> list[dict]

This file is complete (Part 2). Do not modify the public class/method signatures.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import copy     # To safely copy chunk dicts before adding the "score" field

# ── Third-party: scikit-learn ─────────────────────────────────────────────────
# TfidfVectorizer converts text into numerical vectors using TF-IDF weights.
from sklearn.feature_extraction.text import TfidfVectorizer

# cosine_similarity computes how similar two vectors are (range 0.0 → 1.0).
from sklearn.metrics.pairwise import cosine_similarity

# numpy lets us work with arrays of numbers efficiently.
import numpy as np


class NotesIndex:
    """
    An in-memory search index built from text chunks.

    Usage
    -----
    >>> from agent.chunker import chunk_pages
    >>> from agent.pdf_loader import load_pdf
    >>> pages = load_pdf("notes.pdf")
    >>> chunks = chunk_pages(pages)
    >>> index = NotesIndex(chunks)
    >>> results = index.search("What is photosynthesis?", top_k=3)
    >>> results[0]["score"]   # float between 0.0 and 1.0
    0.74
    """

    def __init__(self, chunks: list[dict]) -> None:
        """
        Build the TF-IDF index from a list of chunk dicts.

        This is called once when the user uploads a PDF. After this, search()
        can be called many times quickly because the heavy work is done here.

        Parameters
        ----------
        chunks : list[dict]
            Output of chunk_pages(). Each dict has keys: "id", "page", "text".
            Can be an empty list — search() will return [] gracefully.
        """

        # Store the original chunks so search() can return them with metadata.
        # We keep a copy so external code can't accidentally modify our data.
        self._chunks: list[dict] = chunks

        # If there are no chunks (e.g. empty PDF), we can't build an index.
        # We flag this so search() can return [] without crashing.
        if not chunks:
            self._vectorizer = None      # No vectorizer needed
            self._tfidf_matrix = None    # No matrix needed
            return

        # ── Step 1: Extract the raw text from each chunk ──────────────────────
        # TfidfVectorizer needs a list of strings — one per "document" (chunk).
        texts: list[str] = [chunk["text"] for chunk in chunks]

        # ── Step 2: Compute safe max_df for this corpus size ─────────────────
        # TfidfVectorizer's max_df parameter ignores words that appear in more
        # than max_df fraction of documents. The problem: with very small corpora
        # (e.g. 1 chunk), max_df=0.95 → 0.95 * 1 = 0.95 documents, which is
        # LESS than min_df=1 document, and sklearn raises a ValueError.
        # Fix: if the corpus has fewer than 20 chunks, set max_df=1.0 (keep all
        # terms). This is fine for tiny inputs — no term will be truly ubiquitous.
        n_docs: int = len(texts)
        safe_max_df: float = 0.95 if n_docs >= 20 else 1.0

        # ── Step 3: Fit the TF-IDF vectorizer ────────────────────────────────
        # TfidfVectorizer learns the vocabulary (all unique words across all chunks)
        # and their IDF weights. Then it transforms each text into a sparse vector.
        #
        # Parameters explained:
        #   sublinear_tf=True  → applies log(1+tf) instead of raw tf, which
        #                        dampens the effect of very frequent words.
        #   min_df=1           → include a word even if it appears in only 1 chunk
        #                        (important for small documents).
        #   max_df              → ignore words that appear in max_df fraction of
        #                        chunks (too common to be useful for retrieval).
        #                        Computed above to handle small corpora safely.
        #   ngram_range=(1,2)  → include single words AND two-word phrases.
        #                        "cell division" gets its own feature, not just
        #                        "cell" + "division" separately. Improves recall.
        self._vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            min_df=1,
            max_df=safe_max_df,
            ngram_range=(1, 2),   # unigrams + bigrams
        )

        # fit_transform(texts) does two things in one call:
        #   fit()      → learn the vocabulary and IDF weights from `texts`
        #   transform()→ convert each text into a TF-IDF vector
        # Result is a sparse matrix: shape (num_chunks, num_vocabulary_terms)
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        """
        Find the top-k chunks most relevant to a natural-language query.

        The query is converted into the same TF-IDF vector space as the chunks,
        then we compute cosine similarity between the query vector and every
        chunk vector. The top-k highest similarities are returned.

        Parameters
        ----------
        query : str
            A natural-language question or topic, e.g. "What is mitosis?".
        top_k : int, optional
            How many results to return. Default 4.
            If the index has fewer chunks than top_k, all chunks are returned.

        Returns
        -------
        list[dict]
            A list of chunk dicts (copies), each with an extra "score" field:
                [{"id": 2, "page": 3, "text": "...", "score": 0.87}, ...]
            Sorted by score DESCENDING (best match first).
            Returns [] if the index is empty or no matches are found.

        Examples
        --------
        >>> results = index.search("photosynthesis", top_k=2)
        >>> results[0]["score"] >= results[1]["score"]
        True
        """

        # ── Guard: handle empty index gracefully ──────────────────────────────
        if self._vectorizer is None or self._tfidf_matrix is None:
            # The index was built with an empty chunk list — nothing to search.
            return []

        # ── Guard: handle empty query ─────────────────────────────────────────
        if not query or not query.strip():
            return []

        # ── Step 1: Vectorise the query ───────────────────────────────────────
        # transform() converts the query string into a TF-IDF vector using the
        # vocabulary and IDF weights that were ALREADY learned in __init__.
        # Important: we use transform(), NOT fit_transform(), because we don't
        # want to re-learn the vocabulary from the query — only apply the existing one.
        try:
            query_vector = self._vectorizer.transform([query])
        except Exception:
            # If the query contains only words not in our vocabulary,
            # transform may behave oddly — return empty to be safe.
            return []

        # ── Step 2: Compute cosine similarity ─────────────────────────────────
        # cosine_similarity(query_vector, tfidf_matrix) returns a 2-D array:
        #   shape: (1, num_chunks)  ← one row (the query), many columns (chunks)
        # We flatten it to a 1-D array with [0] so we have one score per chunk.
        similarity_scores: np.ndarray = cosine_similarity(
            query_vector, self._tfidf_matrix
        )[0]

        # ── Step 3: Pick the top-k highest-scoring indices ────────────────────
        # np.argsort() returns indices that would sort the array in ASCENDING order.
        # [::-1] reverses it → DESCENDING order (highest similarity first).
        sorted_indices: np.ndarray = np.argsort(similarity_scores)[::-1]

        # Cap at top_k — don't return more results than requested.
        # Also cap at the total number of chunks (can't return more than we have).
        k: int = min(top_k, len(self._chunks))
        top_indices: np.ndarray = sorted_indices[:k]

        # ── Step 4: Build result list ─────────────────────────────────────────
        results: list[dict] = []

        for idx in top_indices:
            score: float = float(similarity_scores[idx])

            # Skip chunks with zero similarity — they are completely unrelated.
            # This keeps the result list clean when the query matches nothing.
            if score == 0.0:
                continue

            # Make a shallow copy of the chunk dict so we don't mutate the original.
            # Then add the "score" field as required by the contract.
            chunk_with_score: dict = copy.copy(self._chunks[idx])
            chunk_with_score["score"] = round(score, 4)   # 4 decimal places is plenty

            results.append(chunk_with_score)

        # Results are already sorted descending (argsort + reverse gives us that).
        return results
