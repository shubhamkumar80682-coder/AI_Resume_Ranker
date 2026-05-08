import logging
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Structural words that appear in every JD/resume but carry no matching signal.
JD_HEADER_NOISE = frozenset([
    "job", "description", "responsibilities", "qualifications",
    "looking", "need", "needs", "needed", "seeking", "seek", "wanted",
    "ideal", "candidate", "applicant", "requirements", "required",
])

RESUME_HEADER_NOISE = frozenset([
    "resume", "curriculum", "vitae", "references",
])


def _clean_text(text: str) -> str:
    """Remove PDF artifacts and normalize text for TF-IDF."""
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"\b(?:page|pg)\.?\s*\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d+\s*of\s*\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)\d{1,2}(?!\w)", "", text)

    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"[\-_]", " ", text)
    text = text.lower()
    text = re.sub(r"\b(?![cr]\b)[a-z]\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _build_stop_words() -> list[str]:
    """Combine sklearn stopwords with domain-specific header noise words."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stops
    return list(sklearn_stops | JD_HEADER_NOISE | RESUME_HEADER_NOISE)


def _match_score(jd_tfidf_row, resume_tfidf_row, vectorizer: TfidfVectorizer) -> float:
    """Compute a match score based on IDF-weighted term overlap (unigrams only).

    For each JD unigram present in the resume, add its TF-IDF weight.
    Normalize by the total JD unigram TF-IDF weight so the score is in [0, 1].
    """
    feature_names = vectorizer.get_feature_names_out()
    jd_scores = jd_tfidf_row.toarray()[0]
    resume_scores = resume_tfidf_row.toarray()[0]

    total_jd = 0.0
    matched = 0.0
    for i, name in enumerate(feature_names):
        if " " in name:
            continue
        total_jd += jd_scores[i]
        if resume_scores[i] > 0:
            matched += jd_scores[i]

    return matched / total_jd if total_jd > 0 else 0.0


def rank_resumes(job_desc: str, resumes: list[str]) -> list[float]:
    """Return match scores between a job description and each resume.

    Combines TF-IDF cosine similarity with term coverage (recall) for
    more intuitive scores. Returns values in [0, 1] in the same order
    as *resumes*.
    """
    if not job_desc or not job_desc.strip():
        raise ValueError("Job description must not be empty")

    valid_resumes = [r for r in resumes if r and r.strip()]
    if not valid_resumes:
        raise ValueError("At least one non-empty resume is required")

    logger.info("Processing %d valid resumes", len(valid_resumes))

    cleaned_job = _clean_text(job_desc)
    cleaned_resumes = [_clean_text(r) for r in valid_resumes]
    documents = [cleaned_job] + cleaned_resumes

    n_docs = len(documents)
    max_df_count = max(int(0.95 * n_docs), 2)

    vectorizer = TfidfVectorizer(
        stop_words=_build_stop_words(),
        min_df=1,
        max_df=max_df_count,
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b[a-z][a-z\-]+\b",
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    # --- Component 1: Cosine similarity ---
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # --- Component 2: IDF-weighted term coverage (unigrams only) ---
    jd_tfidf = tfidf_matrix[0]

    coverages = []
    for i in range(1, tfidf_matrix.shape[0]):
        cov = _match_score(jd_tfidf, tfidf_matrix[i], vectorizer)
        coverages.append(cov)

    coverages = np.array(coverages)

    # --- Blend cosine similarity with term coverage ---
    # Cosine similarity on sparse TF-IDF vectors is naturally small (0-0.3).
    # Term coverage (recall) captures "how many JD requirements does this resume hit?"
    # and produces more intuitive percentages.
    #
    # We weight coverage heavily since it directly answers "does this resume
    # have what the job asks for?" and produces scores users expect (20-80%).
    scores = 0.3 * cos_sim + 0.7 * coverages

    # Round to 4 decimal places
    scores = [round(float(s), 4) for s in scores]
    logger.info("Scores: %s", scores)
    return scores
