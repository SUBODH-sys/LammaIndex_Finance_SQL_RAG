#Functions written here imported in Streamlit UI/UX file (app.py)
"""
main.py -> core logic for (LlamaIndex NLSQLTableQueryEngine + Groq).

Everything in here is UI-free, so it can be used from Streamlit (app.py), a notebook, or tests.

Pipeline
    uploaded CSV/Excel files -> SQLite (one table per file/sheet, SQL-friendly names)
    -> auto-generated data dictionary (schema profile + optional user notes)
    -> question -> [SQL model: openai/gpt-oss-120b] -> SQL -> run on a READ-ONLY db
    -> [text model: openai/gpt-oss-20b] -> written answer   (+ optional Plotly chart)

Only two LLMs are used:
    SQL_MODEL  "openai/gpt-oss-120b"  -> text-to-SQL only
    TEXT_MODEL "openai/gpt-oss-20b"   -> follow-up rewriting, written answers, chart spec, question ideas
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
import urllib.parse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import pandas as pd
import numpy as np
import plotly.express as px
from pydantic import PrivateAttr
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from llama_index.core import PromptTemplate, SQLDatabase
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse, MessageRole
from llama_index.core.prompts import PromptType
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.groq import Groq
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)
if os.getenv("DEBUG_CACHE") == "1":
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
SQL_MODEL = "openai/gpt-oss-120b"    # writes SQL
TEXT_MODEL = "openai/gpt-oss-20b"    # writes the English answer (+ rewrites follow-ups, chart spec)

WORK_DIR = Path(os.getenv("WORK_DIR", "."))
DEFAULT_CACHE_PATH = WORK_DIR / "llm_cache.db"

QUERY_TIMEOUT_S = 20          # kill runaway SQL (e.g. accidental cross joins)
MAX_RESULT_ROWS = 5000
MAX_CONTEXT_CHARS = 9000      # cap on the data dictionary so prompts stay inside Groq token limits

SUPPORTED_EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls")

EXAMPLE_NOTES = (
    "Money columns (total_revenue, net_income, market_cap_usd_m ...) are in USD millions.\n"
    "Columns ending in _pct are percentages (12.5 means 12.5%).\n"
    "p_e_ratio is NULL when earnings are negative.\n"
    "Join financials and stock_market on ticker AND year."
)

# One shared, harmless embed model: the SQL engine needs no embeddings, and this stops
# LlamaIndex from asking for an OpenAI key.
_MOCK_EMBED = MockEmbedding(embed_dim=8)


# --------------------------------------------------------------------------------------
# 1. Files -> DataFrames -> SQLite
# --------------------------------------------------------------------------------------
SQLITE_RESERVED = {
    "abort", "all", "alter", "and", "as", "between", "by", "case", "check", "column", "create",
    "default", "delete", "desc", "distinct", "drop", "else", "end", "except", "exists", "from",
    "group", "having", "in", "index", "insert", "into", "is", "join", "key", "like", "limit",
    "not", "null", "on", "or", "order", "primary", "references", "select", "set", "table", "then",
    "to", "union", "unique", "update", "values", "when", "where", "with",
}


def clean_identifier(name: Any, fallback: str = "col") -> str:
    """'Revenue Growth (%)' -> 'revenue_growth_pct', 'P/E Ratio' -> 'p_e_ratio'. Always a safe SQL name."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.strip().lower().replace("%", "pct")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_") or fallback
    if s[0].isdigit():
        s = f"c_{s}"
    if s in SQLITE_RESERVED:
        s = f"{s}_col"
    return s


def clean_table_name(name: Any) -> str:
    s = clean_identifier(name, "table")
    return f"{s[:-4]}_tbl" if s.endswith("_col") and s[:-4] in SQLITE_RESERVED else s


def clean_column(name: Any) -> str:
    return clean_identifier(name, "col")


def unique_names(names: List[str]) -> List[str]:
    """Make names unique ('x', 'x' -> 'x', 'x_2'), e.g. when two columns clean to the same name."""
    seen, out = set(), []
    for n in names:
        cand, i = n, 2
        while cand in seen:
            cand, i = f"{n}_{i}", i + 1
        seen.add(cand)
        out.append(cand)
    return out


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            if df.shape[1] == 1:  # maybe ';' or tab separated
                df2 = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
                if df2.shape[1] > 1:
                    df = df2
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"Could not read CSV: {last_err}")


def read_uploaded_file(filename: str, data: bytes) -> Dict[str, pd.DataFrame]:
    """Return {table_hint: DataFrame}. CSV -> 1 table; Excel -> 1 table per sheet."""
    path = Path(filename)
    suffix, stem = path.suffix.lower(), path.stem
    if suffix in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        if len(sheets) == 1:
            return {stem: next(iter(sheets.values()))}
        return {f"{stem}_{sheet}": df for sheet, df in sheets.items()}
    if suffix in (".csv", ".tsv", ".txt"):
        return {stem: _read_csv_bytes(data)}
    raise ValueError(f"Unsupported file type '{suffix}'. Use: {', '.join(SUPPORTED_EXTENSIONS)}")


def load_uploads(files: Dict[str, bytes]) -> Dict[str, pd.DataFrame]:
    """{filename: bytes} -> {table_name: DataFrame} with unique, SQL-safe table names."""
    raw: List[tuple] = []
    for fname, data in files.items():
        raw.extend(read_uploaded_file(fname, data).items())
    names = unique_names([clean_table_name(hint) for hint, _ in raw])
    return {n: df for n, (_, df) in zip(names, raw)}


_DATE_NAME = re.compile(r"(date|time|timestamp|day|month|period)", re.I)
_ISO_LIKE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")


def _to_num(s: pd.Series) -> pd.Series:
    t = (s.astype(str).str.strip()
         .str.replace(r"^\((.*)\)$", r"-\1", regex=True)         # (123) -> -123
         .str.replace(r"[,$€£₹%\s]", "", regex=True))
    return pd.to_numeric(t, errors="coerce")


def _is_text(s: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s)
                or pd.api.types.is_bool_dtype(s))


def _coerce_column(s: pd.Series, colname: str) -> pd.Series:
    """Fix common CSV problems: '$1,234' / '12%' stored as text, and dates stored as text/datetime."""
    if pd.api.types.is_datetime64_any_dtype(s):
        has_time = (s.dropna().dt.normalize() != s.dropna()).any()
        return s.dt.strftime("%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d")
    if not _is_text(s):
        return s
    filled = s.dropna().astype(str).str.strip()
    filled = filled[filled != ""]
    if filled.empty:
        return s
    # numbers written with currency symbols / thousands separators / percent signs
    if filled.str.contains(r"[$€£₹,%]").any():
        num = _to_num(s)
        if num[filled.index].notna().mean() >= 0.95:
            return num
    # dates stored as text
    sample = filled.head(50)
    looks_like_date = sample.str.match(_ISO_LIKE).mean() > 0.9 or (
        _DATE_NAME.search(colname) and sample.str.contains(r"\d").mean() > 0.9)
    if looks_like_date:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().sum() >= 0.95 * len(filled):
            return _coerce_column(parsed, colname)
    return s


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names (unique, SQL-safe) and fix text-typed numbers/dates."""
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").copy()
    df.columns = unique_names([clean_column(c) if str(c).strip() else f"col_{i}"
                               for i, c in enumerate(df.columns)])
    for c in df.columns:
        df[c] = _coerce_column(df[c], c)
    return df


def build_database(frames: Dict[str, pd.DataFrame], db_path: Path) -> Dict[str, List[str]]:
    """
    Write every DataFrame to a fresh SQLite file (one table each). Also indexes columns that appear in
    more than one table (probable join keys). Returns {table: [columns]}.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    if not frames:
        raise ValueError("No tables to load.")
    prepared = {t: prepare_dataframe(df) for t, df in frames.items()}
    for t, df in prepared.items():
        if df.empty or df.shape[1] == 0:
            raise ValueError(f"Table '{t}' has no usable data.")
    con = sqlite3.connect(db_path)
    try:
        for t, df in prepared.items():
            df.to_sql(t, con, index=False, if_exists="replace")
        owners: Dict[str, List[str]] = {}
        for t, df in prepared.items():
            for c in df.columns:
                owners.setdefault(c, []).append(t)
        for c, tabs in owners.items():
            if len(tabs) > 1:
                for t in tabs:
                    con.execute(f'CREATE INDEX IF NOT EXISTS "ix_{t}_{c}" ON "{t}"("{c}")')
        con.commit()
    finally:
        con.close()
    return {t: list(df.columns) for t, df in prepared.items()}


# --------------------------------------------------------------------------------------
# 2. Read-only database access + schema context (auto data dictionary)
# --------------------------------------------------------------------------------------
def _ro_uri(db_path: Path) -> str:
    return "file:" + urllib.parse.quote(Path(db_path).resolve().as_posix(), safe="/:") + "?mode=ro"


def _ro_connect(db_path: Path, timeout_s: float = QUERY_TIMEOUT_S) -> sqlite3.Connection:
    """Read-only SQLite connection with a time limit: the LLM can never modify data or hang the app."""
    con = sqlite3.connect(_ro_uri(db_path), uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    deadline = time.time() + timeout_s
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 100_000)
    return con


def get_readonly_engine(db_path: Path):
    return create_engine("sqlite://", creator=lambda: _ro_connect(db_path), poolclass=NullPool)


def list_tables(db_path: Path) -> List[str]:
    con = _ro_connect(db_path)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


def get_sql_database(db_path: Path) -> SQLDatabase:
    eng = get_readonly_engine(db_path)
    return SQLDatabase(eng, include_tables=list_tables(db_path))


def table_preview(db_path: Path, table: str, n: int = 10) -> pd.DataFrame:
    con = _ro_connect(db_path)
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {int(n)}', con)
    finally:
        con.close()


def profile_database(db_path: Path, max_distinct: int = 51) -> Dict[str, Dict[str, Any]]:
    """Per table: row count and per column: type, nulls, distinct count, min/max, distinct values/examples."""
    con = _ro_connect(db_path, timeout_s=60)
    prof: Dict[str, Dict[str, Any]] = {}
    try:
        for t in list_tables(db_path):
            n_rows = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            cols = []
            for r in con.execute(f'PRAGMA table_info("{t}")').fetchall():
                name, ctype = r[1], (r[2] or "").upper()
                info = {"name": name, "type": ctype or "TEXT", "nulls": 0, "distinct": 0,
                        "min": None, "max": None, "values": [], "examples": []}
                try:
                    nn, nd, lo, hi = con.execute(
                        f'SELECT SUM("{name}" IS NULL), COUNT(DISTINCT "{name}"), MIN("{name}"), MAX("{name}") '
                        f'FROM "{t}"').fetchone()
                    info.update(nulls=nn or 0, distinct=nd, min=lo, max=hi)
                    lim = max_distinct if nd <= max_distinct else 5
                    vals = [v[0] for v in con.execute(
                        f'SELECT DISTINCT "{name}" FROM "{t}" WHERE "{name}" IS NOT NULL ORDER BY 1 LIMIT {lim}')]
                    info["values" if nd <= max_distinct else "examples"] = vals
                except sqlite3.Error:
                    pass
                cols.append(info)
            prof[t] = {"rows": n_rows, "columns": cols}
    finally:
        con.close()
    return prof


def _fmt(v: Any, maxlen: int = 40) -> str:
    s = f"{v:.6g}" if isinstance(v, float) else str(v)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def _render_context(prof: Dict[str, Dict[str, Any]], list_limit: int, notes: str) -> str:
    lines = ["DATA DICTIONARY (auto-generated from the uploaded data)"]
    owners: Dict[str, List[str]] = {}
    for t, p in prof.items():
        lines.append(f"Table `{t}` ({p['rows']} rows)")
        for c in p["columns"]:
            owners.setdefault(c["name"], []).append(t)
            numeric = c["type"] in ("INTEGER", "REAL", "NUMERIC", "FLOAT", "DOUBLE", "INT")
            if c["distinct"] == 0:
                desc = "all values empty"
            elif numeric:
                desc = f"range {_fmt(c['min'])} to {_fmt(c['max'])}"
                if c["distinct"] <= 10 and c["type"] in ("INTEGER", "INT"):
                    desc += "; values: " + ", ".join(_fmt(v) for v in c["values"])
            elif (isinstance(c["min"], str) and _ISO_LIKE.match(c["min"])
                  and isinstance(c["max"], str) and _ISO_LIKE.match(c["max"])):
                desc = f"dates from {c['min']} to {c['max']}"
            elif c["values"] and c["distinct"] <= list_limit:
                desc = f"{c['distinct']} distinct: " + "; ".join(f"'{_fmt(v)}'" for v in c["values"])
            else:
                ex = c["values"][:5] or c["examples"]
                desc = f"{c['distinct']} distinct, e.g. " + "; ".join(f"'{_fmt(v)}'" for v in ex)
            if c["nulls"]:
                desc += f"; {c['nulls']} NULLs"
            lines.append(f"  - {c['name']} ({c['type']}): {desc}")
    shared = {c: tabs for c, tabs in owners.items() if len(tabs) > 1}
    if shared:
        lines.append(
            "Columns present in several tables (likely join keys): "
            + "; ".join(f"{c} ({', '.join(tabs)})" for c, tabs in shared.items())
            + ". When joining, join on ALL key columns that identify a row (e.g. an id AND a year/date), "
              "and always qualify shared column names with the table name."
        )
    if notes.strip():
        lines.append("NOTES FROM THE USER (trust these):\n" + notes.strip()[:2500])
    return "\n".join(lines) + "\n"


def build_schema_context(db_path: Path, notes: str = "", max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Data dictionary added to the text-to-SQL prompt. Fully automatic, so it works for any upload.
    If it gets too long (many columns), value lists are shortened step by step to protect token limits.
    """
    prof = profile_database(db_path)
    text = ""
    for limit in (50, 20, 10, 5, 0):
        text = _render_context(prof, limit, notes)
        if len(text) <= max_chars:
            return text
    return text[:max_chars]


# --------------------------------------------------------------------------------------
# 3. LLM cache + CachedGroq
# --------------------------------------------------------------------------------------
_QUESTION_LINE_RE = re.compile(r"^Question:\s*(.+)$", re.M)


def _extract_semantic_text(payload: Any) -> str:
    """Extract the user's question from prompt templates so schema/table text doesn't dilute embeddings."""
    raw = json.dumps(payload, default=str) if not isinstance(payload, str) else payload
    matches = _QUESTION_LINE_RE.findall(raw)
    if matches:
        for candidate in reversed(matches):
            cleaned = candidate.strip()
            if cleaned and cleaned.lower() != "question here":
                return cleaned
    return raw.strip()


def _schema_fingerprint(payload: Any) -> str:
    """Hash static prompt/schema context separately from the variable question."""
    raw = json.dumps(payload, sort_keys=True, default=str)
    without_question = _QUESTION_LINE_RE.sub("Question: <VAR>", raw)
    return hashlib.sha256(without_question.encode()).hexdigest()


class LLMCache:
    """
    Two-tier persistent cache for LLM responses (SQLite):
      1. Exact SHA-256 lookup.
      2. Semantic cosine-similarity lookup over extracted questions within the same scope.
    """

    _embedder: Any = None
    _embedder_failed: bool = False
    _embedder_lock = threading.Lock()

    def __init__(
        self,
        path: Path = DEFAULT_CACHE_PATH,
        ttl_seconds: Optional[int] = None,
        similarity_threshold: float = 0.85,
        embed_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.path = str(path)
        self.ttl = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.embed_model_name = embed_model_name
        self.hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._stat_lock = threading.Lock()

        self._exec("PRAGMA journal_mode=WAL")
        self._exec(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                scope_hash TEXT,
                semantic_text TEXT,
                embedding BLOB,
                response TEXT,
                created REAL
            )
            """
        )
        existing_cols = {r[1] for r in self._exec("PRAGMA table_info(llm_cache)")}
        for col, col_type in (("scope_hash", "TEXT"), ("semantic_text", "TEXT"), ("embedding", "BLOB")):
            if col not in existing_cols:
                self._exec(f"ALTER TABLE llm_cache ADD COLUMN {col} {col_type}")
        self._exec("CREATE INDEX IF NOT EXISTS ix_llm_cache_scope ON llm_cache(scope_hash)")

    def _get_embedder(self):
        if SentenceTransformer is None or LLMCache._embedder_failed:
            return None
        if LLMCache._embedder is None:
            with LLMCache._embedder_lock:
                if LLMCache._embedder is None and not LLMCache._embedder_failed:
                    try:
                        LLMCache._embedder = SentenceTransformer(self.embed_model_name)
                        logger.info("Loaded semantic cache embedder: %s", self.embed_model_name)
                    except Exception as exc:  # noqa: BLE001
                        LLMCache._embedder_failed = True
                        logger.warning("Semantic embedder unavailable (%s); using exact cache only.", exc)
        return LLMCache._embedder

    def _encode(self, text: str) -> Optional[np.ndarray]:
        embedder = self._get_embedder()
        if embedder is None or not text:
            return None
        vec = embedder.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            try:
                rows = conn.execute(sql, params).fetchall()
                conn.commit()
                return rows
            finally:
                conn.close()

    @staticmethod
    def make_key(model: str, temperature: float, payload: Any) -> str:
        raw = json.dumps({"m": model, "t": temperature, "p": payload}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def make_scope(model: str, temperature: float, settings: Any, payload: Any) -> str:
        raw = json.dumps(
            {"m": model, "t": temperature, "s": settings, "ctx": _schema_fingerprint(payload)},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        key: str,
        scope_hash: Optional[str] = None,
        semantic_text: Optional[str] = None,
    ) -> Optional[tuple[str, str]]:
        now = time.time()

        # 1. Exact match check
        rows = self._exec("SELECT response, created FROM llm_cache WHERE key = ?", (key,))
        if rows and (self.ttl is None or now - rows[0][1] < self.ttl):
            with self._stat_lock:
                self.hits += 1
            logger.info("CACHE EXACT HIT | q=%r", (semantic_text or "")[:80])
            return rows[0][0], key

        # 2. Semantic match check within same scope
        if scope_hash and semantic_text:
            query_vec = self._encode(semantic_text)
            if query_vec is not None:
                candidates = self._exec(
                    "SELECT key, semantic_text, embedding, response, created FROM llm_cache "
                    "WHERE scope_hash = ? AND embedding IS NOT NULL",
                    (scope_hash,),
                )
                best_score = -1.0
                best_cached_q = ""
                best_row: Optional[tuple[str, str]] = None
                for cand_key, cand_q, blob, resp, created in candidates:
                    if self.ttl is not None and now - created >= self.ttl:
                        continue
                    cand_vec = np.frombuffer(blob, dtype=np.float32)
                    if cand_vec.shape != query_vec.shape:
                        continue
                    score = float(np.dot(query_vec, cand_vec))
                    if score > best_score:
                        best_score = score
                        best_cached_q = cand_q or ""
                        best_row = (resp, cand_key)

                logger.info(
                    "CACHE SEMANTIC CHECK | best=%.4f (thresh=%.2f) | cur=%r vs cached=%r | pool=%d",
                    best_score,
                    self.similarity_threshold,
                    semantic_text[:70],
                    best_cached_q[:70],
                    len(candidates),
                )
                if best_row is not None and best_score >= self.similarity_threshold:
                    with self._stat_lock:
                        self.hits += 1
                        self.semantic_hits += 1
                    return best_row

        with self._stat_lock:
            self.misses += 1
        return None

    def delete(self, key: str) -> None:
        self._exec("DELETE FROM llm_cache WHERE key = ?", (key,))

    def set(
        self,
        key: str,
        response: str,
        scope_hash: Optional[str] = None,
        semantic_text: Optional[str] = None,
    ) -> None:
        vec = self._encode(semantic_text) if semantic_text else None
        blob = vec.tobytes() if vec is not None else None
        self._exec(
            "INSERT OR REPLACE INTO llm_cache (key, scope_hash, semantic_text, embedding, response, created) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, scope_hash, semantic_text, blob, response, time.time()),
        )

    def size(self) -> int:
        return self._exec("SELECT COUNT(*) FROM llm_cache")[0][0]

    def clear(self) -> None:
        self._exec("DELETE FROM llm_cache")
        self.hits = self.semantic_hits = self.misses = 0

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": self.size(),
            "hits": self.hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 2) if total else 0.0,
        }


THINK_REGEX = re.compile(r"<think>.*?</think>", re.S | re.I)


def clean_llm_text(text: Optional[str]) -> str:
    """Remove <think>...</think> blocks some reasoning models put in the answer."""
    return THINK_REGEX.sub("", text or "").strip()


def _require_answer(text: Optional[str]) -> str:
    out = clean_llm_text(text)
    if not out or out.lstrip().lower().startswith("<think>"):
        raise RuntimeError(
            "The model returned no final answer (a reasoning model probably used all max_tokens for "
            "thinking). Increase max_tokens in make_llm() and try again."
        )
    return out


class CachedGroq(Groq):
    """Groq LLM whose chat()/complete() use exact + semantic LLMCache."""

    _llm_cache: Optional[LLMCache] = PrivateAttr(default=None)
    _local: Any = PrivateAttr(default_factory=threading.local)

    def __init__(self, *args, llm_cache: Optional[LLMCache] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm_cache = llm_cache

    def _keys(self, payload) -> tuple[str, str, str]:
        settings = [self.max_tokens, self.additional_kwargs]
        exact_key = LLMCache.make_key(self.model, self.temperature, [settings, payload])
        scope_hash = LLMCache.make_scope(self.model, self.temperature, settings, payload)
        semantic_text = _extract_semantic_text(payload)
        return exact_key, scope_hash, semantic_text

    def evict_last(self) -> None:
        key = getattr(self._local, "last_key", None)
        if self._llm_cache is not None and key:
            self._llm_cache.delete(key)
            self._local.last_key = None

    def chat(self, messages, **kwargs) -> ChatResponse:
        cache = self._llm_cache
        if cache is None:
            return super().chat(messages, **kwargs)
        payload = [[m.role.value, str(m.content)] for m in messages]
        key, scope_hash, semantic_text = self._keys(payload)
        hit = cache.get(key, scope_hash=scope_hash, semantic_text=semantic_text)
        if hit is not None:
            text, matched_key = hit
            self._local.last_key = matched_key
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
                raw={"cached": True, "semantic": matched_key != key},
            )
        self._local.last_key = key
        resp = super().chat(messages, **kwargs)
        text = _require_answer(resp.message.content)
        cache.set(key, text, scope_hash=scope_hash, semantic_text=semantic_text)
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=text), raw=resp.raw)

    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        cache = self._llm_cache
        if cache is None:
            return super().complete(prompt, formatted=formatted, **kwargs)
        payload = ["complete", prompt]
        key, scope_hash, semantic_text = self._keys(payload)
        hit = cache.get(key, scope_hash=scope_hash, semantic_text=semantic_text)
        if hit is not None:
            text, matched_key = hit
            self._local.last_key = matched_key
            return CompletionResponse(text=text, raw={"cached": True, "semantic": matched_key != key})
        self._local.last_key = key
        resp = super().complete(prompt, formatted=formatted, **kwargs)
        text = _require_answer(resp.text)
        cache.set(key, text, scope_hash=scope_hash, semantic_text=semantic_text)
        return CompletionResponse(text=text, raw=resp.raw)

def make_llm(api_key: str, model: str, cache: Optional[LLMCache] = None,
             temperature: float = 0.0, max_tokens: int = 2000) -> CachedGroq:
    """
    Create one cached Groq LLM. Nothing global is touched (no Settings.llm, no os.environ), so several
    Streamlit sessions with different API keys can run side by side safely.
    """
    if not api_key:
        raise ValueError("Groq API key missing.")
    extra: Dict[str, Any] = {}
    if model.startswith("openai/gpt-oss"):
        extra["additional_kwargs"] = {"reasoning_effort": "low"}   # faster, far fewer hidden-thinking tokens
    return CachedGroq(model=model, api_key=api_key, temperature=temperature,
                      max_tokens=max_tokens, llm_cache=cache, **extra)


def make_llms(api_key: str, cache: Optional[LLMCache] = None):
    """Return (sql_llm, text_llm): gpt-oss-120b for SQL, gpt-oss-20b for English. Both share one cache."""
    return make_llm(api_key, SQL_MODEL, cache), make_llm(api_key, TEXT_MODEL, cache)


# --------------------------------------------------------------------------------------
# 4. NLSQLTableQueryEngine (SQL only) + answer writer
# --------------------------------------------------------------------------------------
TEXT_TO_SQL_TMPL = (
    "Given an input question, create a syntactically correct {dialect} query to run, "
    "then look at the results and answer.\n"
    "Rules:\n"
    "- Only use the tables and columns listed below. Never invent columns or tables.\n"
    "- Write ONE read-only SELECT statement (a WITH ... SELECT is fine).\n"
    "- When the user names an entity (a company, product, person, city ...), find the matching column in the "
    "data dictionary and match case-insensitively with LOWER(column) LIKE '%name%'.\n"
    "- Always include identifying columns (name/id and any year/date/period column) next to the metrics you "
    "return, so the result is self-explanatory and can be charted.\n"
    "- For 'top N' / 'highest' / 'lowest' questions use ORDER BY with LIMIT. For other plain lists use LIMIT 50.\n"
    "- Ratios, rates and totals that already exist as columns are precomputed; do not recompute them unless asked.\n"
    "- If the question asks for a chart, plot or graph, do not mention charts: just return the rows needed "
    "(for example a period column plus the metric columns, or a category plus an aggregated value).\n"
    "- Output the raw SQL after 'SQLQuery:' with no markdown fences and no explanation.\n\n"
    "Use this exact format, each item on its own line:\n"
    "Question: question here\n"
    "SQLQuery: SQL query to run\n"
    "SQLResult: result of the SQLQuery\n"
    "Answer: final answer here\n\n"
    "Only use the following tables:\n{schema}\n\n"
    "Question: {query_str}\n"
    "SQLQuery: "
)
TEXT_TO_SQL_PROMPT = PromptTemplate(TEXT_TO_SQL_TMPL, prompt_type=PromptType.TEXT_TO_SQL)

SYNTHESIS_TMPL = (
    "You are a data analyst. Answer the question using ONLY the SQL result below.\n"
    "Be concise (2-5 sentences). Quote numbers with their units and the year/period when present. "
    "Infer units only from column names (e.g. *_pct = percent, *_usd = US dollars, *_m = millions) or the notes; "
    "never invent units. If the result is empty, say no matching data was found. "
    "If the question asks for a chart, do not say you cannot draw one: the chart is rendered separately, "
    "so just summarise the key takeaway from the data.\n"
    "{notes_block}\n"
    "Question: {query_str}\n"
    "SQL: {sql_query}\n"
    "SQL result: {context_str}\n"
    "Answer: "
)
SYNTHESIS_PROMPT = PromptTemplate(SYNTHESIS_TMPL, prompt_type=PromptType.SQL_RESPONSE_SYNTHESIS_V2)


def build_query_engine(llm, db_path: Path, schema_context: str) -> NLSQLTableQueryEngine:
    """SQL-only engine (1 SQL-model call per question). Rebuild it whenever the tables change."""
    return NLSQLTableQueryEngine(
        sql_database=get_sql_database(db_path),
        tables=list_tables(db_path),
        llm=llm,
        embed_model=_MOCK_EMBED,
        text_to_sql_prompt=TEXT_TO_SQL_PROMPT,
        context_str_prefix=schema_context,
        synthesize_response=False,
    )


def write_answer(text_llm, question: str, sql: str, df: Optional[pd.DataFrame], notes: str = "") -> str:
    """English answer from the SQL result, produced by the small text model (no schema in the prompt)."""
    if df is None or df.empty:
        return "No matching data was found."
    shown = df.head(50)
    rows = shown.to_string(index=False)[:6000]
    if len(df) > len(shown):
        rows += f"\n(showing the first {len(shown)} of {len(df)} rows)"
    notes_block = f"Notes about the data: {notes.strip()[:1500]}\n" if notes.strip() else ""
    prompt = SYNTHESIS_PROMPT.format(query_str=question, sql_query=sql, context_str=rows, notes_block=notes_block)
    return text_llm.complete(prompt).text.strip()


def clean_sql(sql: Optional[str]) -> Optional[str]:
    if not sql:
        return None
    return re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.I | re.M).strip().rstrip(";")


def run_sql_df(sql: str, db_path: Path, max_rows: int = MAX_RESULT_ROWS) -> pd.DataFrame:
    """Run the generated SQL on the read-only DB (with time + row limits) and return a DataFrame."""
    sql = clean_sql(sql)
    if not sql or not re.match(r"^\s*(select|with)\b", sql, re.I):
        raise ValueError("Only SELECT queries are allowed.")
    con = _ro_connect(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM (\n{sql}\n) LIMIT {int(max_rows)}", con)
    except Exception as e:  # noqa: BLE001  (pandas wraps sqlite errors in DatabaseError)
        if "interrupted" in str(e).lower():
            raise TimeoutError(f"The query took longer than {QUERY_TIMEOUT_S}s and was stopped.") from e
        raise
    finally:
        con.close()


# --------------------------------------------------------------------------------------
# 5. Charts on demand (works for any columns)
# --------------------------------------------------------------------------------------
CHART_REGEX = re.compile(
    r"\b(chart|plot|graph|visuali[sz]e|visuali[sz]ation|histogram|pie|scatter)\b"
    r"|\b(line|bar|area|box)\s*(chart|graph|plot)\b", re.I)
CHART_TYPES = ["line", "bar", "scatter", "pie", "area", "histogram", "box"]
TIME_NAME = re.compile(r"(^|_)(year|yr|fy|date|month|quarter|period|time|day|week)($|_)", re.I)


def wants_chart(text: str) -> bool:
    return bool(CHART_REGEX.search(text or ""))


CHART_SPEC_INSTRUCTIONS = (
    "You are a data-visualisation assistant. Choose the best chart for the data below.\n"
    "Respond with ONLY one JSON object, no markdown, in this shape:\n"
    '{"chart_type": "line|bar|scatter|pie|area|histogram|box", "x": "<column>", '
    '"y": "<column>" or ["<column>", "<column>"] or null, "color": "<column>" or null, "title": "<short title>"}\n'
    "Rules: use exact column names. line/area for trends over time; bar for comparing categories; "
    "pie only for shares with at most 10 slices; scatter for the relationship between two numeric columns; "
    "histogram for the distribution of one numeric column (y = null). "
    "If the user named a chart type, use it. If several entities appear in one column, use it as color.\n"
)


def _time_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]) or TIME_NAME.search(str(c)):
            return c
    return None


def _title(hint: str) -> str:
    t = re.sub(r"^\s*(please\s+)?(plot|show( me)?|chart|graph|draw|visuali[sz]e|give me|create)\s+", "", hint, flags=re.I)
    return (t[:1].upper() + t[1:])[:90]


def _fallback_spec(df: pd.DataFrame, hint: str = "") -> Dict[str, Any]:
    """Rule-based chart choice (0 LLM calls). Also the fallback when an LLM spec is invalid."""
    tcol = _time_col(df)
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != tcol]
    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c != tcol]
    ctype = next((t for t in CHART_TYPES if re.search(rf"\b{t}\b", hint, re.I)), None)
    title = _title(hint)
    if ctype == "pie" and cat and num:
        return {"chart_type": "pie", "x": cat[0], "y": num[0], "color": None, "title": title}
    if ctype == "histogram" and (num or tcol):
        return {"chart_type": "histogram", "x": (num or [tcol])[0], "y": None, "color": None, "title": title}
    if tcol and num:
        color = cat[0] if cat and df[cat[0]].nunique() > 1 else None
        return {"chart_type": ctype or "line", "x": tcol, "y": num if not color else num[0],
                "color": color, "title": title}
    if cat and num:
        return {"chart_type": ctype or "bar", "x": cat[0], "y": num[0], "color": None, "title": title}
    if len(num) >= 2:
        return {"chart_type": ctype or "scatter", "x": num[0], "y": num[1], "color": None, "title": title}
    return {"chart_type": "histogram", "x": df.columns[0], "y": None, "color": None, "title": title}


def _valid_spec(spec: Any, df: pd.DataFrame) -> bool:
    if not isinstance(spec, dict) or spec.get("chart_type") not in CHART_TYPES or spec.get("x") is None:
        return False
    ys = spec.get("y")
    ys = [] if ys is None else (ys if isinstance(ys, list) else [ys])
    cols = [spec.get("x")] + ys + ([spec["color"]] if spec.get("color") else [])
    if not all(c in df.columns for c in cols):
        return False
    if spec["chart_type"] != "histogram" and not ys:
        return False
    return all(pd.api.types.is_numeric_dtype(df[y]) for y in ys)


def get_chart_spec(llm, df: pd.DataFrame, question: str, use_llm: bool = False) -> Dict[str, Any]:
    """Default: rule-based (free). use_llm=True asks the LLM to choose (+1 call), validated, with fallback."""
    if not use_llm or llm is None:
        return _fallback_spec(df, question)
    prompt = (f"{CHART_SPEC_INSTRUCTIONS}\nUser request: {question}\n"
              f"Columns: {list(df.columns)}\nRows: {len(df)}\nSample:\n{df.head(8).to_string(index=False)}\n\nJSON:")
    try:
        text = llm.complete(prompt).text
        spec = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        if _valid_spec(spec, df):
            return spec
    except Exception:  # noqa: BLE001
        pass
    return _fallback_spec(df, question)


def build_figure(df: pd.DataFrame, spec: Dict[str, Any]):
    """Turn a DataFrame + chart spec into a Plotly figure."""
    df = df.copy()
    ctype, x, y, color = spec["chart_type"], spec["x"], spec.get("y"), spec.get("color")
    title = spec.get("title") or ""
    if isinstance(y, list) and len(y) == 1:
        y = y[0]
    if isinstance(y, list):
        color = None  # px cannot combine a list of y columns with a color column
    if x in df.columns and (pd.api.types.is_datetime64_any_dtype(df[x]) or TIME_NAME.search(str(x))):
        df = df.sort_values([c for c in [x, color] if c in df.columns])
        if pd.api.types.is_numeric_dtype(df[x]):
            df[x] = df[x].astype(str)          # years as categories (no 2021.5 ticks)
    common = dict(title=title, template="plotly_white")
    if ctype == "line":
        fig = px.line(df, x=x, y=y, color=color, markers=True, **common)
    elif ctype == "area":
        fig = px.area(df, x=x, y=y, color=color, **common)
    elif ctype == "bar":
        fig = px.bar(df, x=x, y=y, color=color, barmode="group", **common)
    elif ctype == "scatter":
        fig = px.scatter(df, x=x, y=y, color=color, hover_data=list(df.columns), **common)
    elif ctype == "box":
        fig = px.box(df, x=x, y=y, color=color, **common)
    elif ctype == "pie":
        fig = px.pie(df, names=x, values=y, **common)
    else:
        fig = px.histogram(df, x=x, color=color, **common)
    fig.update_layout(legend_title_text="", margin=dict(l=20, r=20, t=60, b=20))
    return fig


# --------------------------------------------------------------------------------------
# 6. Conversational assistant
# --------------------------------------------------------------------------------------
CONDENSE_TMPL = (
    "Given the conversation so far and a follow-up message, rewrite the follow-up as ONE standalone question "
    "about the data tables. Resolve pronouns and references such as 'it', 'that one', "
    "'the same year', 'those', 'now show'. Keep any chart request (plot, bar, line, pie...). "
    "If the follow-up is already standalone, return it unchanged. Output only the question.\n\n"
    "Conversation:\n{history}\n\n"
    "Follow-up: {question}\n"
    "Standalone question: "
)

REFERENCE_REGEX = re.compile(
    r"\b(it|its|that|this|those|these|them|they|their|same|both|also|again|too|instead|previous|above|earlier)\b"
    r"|^\s*(and|but|what about|how about)\b", re.I)


def needs_condense(question: str) -> bool:
    """Only spend an LLM call rewriting the question if it looks like a follow-up."""
    return len(question.split()) <= 6 or bool(REFERENCE_REGEX.search(question))


def auto_answer(df: Optional[pd.DataFrame], chart: bool) -> str:
    """Zero-LLM answer text used in fast mode / for charts."""
    if df is None or df.empty:
        return "No matching data was found."
    if chart:
        return f"Here is your chart ({len(df)} data points). The data table and SQL are below."
    if len(df) == 1:
        return ", ".join(f"{c}: {v}" for c, v in df.iloc[0].items())
    return f"Found {len(df)} rows. See the data table below."


@dataclass
class Reply:
    question: str
    standalone_question: str
    answer: str
    sql: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    fig: Any = None
    chart_spec: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cache_hits: int = 0
    llm_calls: int = 0
    seconds: float = 0.0


@dataclass
class TableAssistant:
    sql_llm: Any                      # gpt-oss-120b: used only inside the SQL engine
    text_llm: Any                     # gpt-oss-20b: rewrite follow-ups, written answers, optional chart spec
    sql_engine: Any                   # NLSQLTableQueryEngine(synthesize=False)
    db_path: Path
    notes: str = ""
    llm_answers: bool = True          # False -> never write an LLM answer (fast mode, 1 call per question)
    llm_chart_spec: bool = False      # True -> text model picks chart type/axes (+1 call), else rules
    max_turns: int = 4
    history: List[Dict[str, str]] = field(default_factory=list)

    def reset(self) -> None:
        self.history.clear()

    def _history_text(self) -> str:
        lines = []
        for turn in self.history[-self.max_turns:]:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer'][:300]}")
        return "\n".join(lines)

    def condense(self, question: str) -> str:
        if not self.history or not needs_condense(question):
            return question
        # Same question asked before and it was already standalone -> no rewrite needed. This keeps the
        # prompt (and therefore the SQL cache key) identical instead of changing with the chat history.
        norm = " ".join(question.lower().split())
        if any(" ".join(t["question"].lower().split()) == norm and t.get("standalone") == t["question"]
               for t in self.history):
            return question
        prompt = CONDENSE_TMPL.format(history=self._history_text(), question=question)
        out = self.text_llm.complete(prompt).text.strip().strip('"')
        first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        return first or question

    def _evict_sql(self) -> None:
        evict = getattr(self.sql_llm, "evict_last", None)
        if evict:
            evict()

    def ask(self, question: str) -> Reply:
        cache = getattr(self.sql_llm, "_llm_cache", None)
        h0, m0, t0 = (cache.hits, cache.misses, time.time()) if cache else (0, 0, time.time())
        standalone = question
        try:
            standalone = self.condense(question)                       # text model (only if follow-up)
            chart = wants_chart(question) or wants_chart(standalone)
            resp = self.sql_engine.query(standalone)                   # SQL model (1 call)
            sql = clean_sql((resp.metadata or {}).get("sql_query"))
            reply = Reply(question, standalone, "", sql=sql)
            if sql:
                try:
                    reply.df = run_sql_df(sql, self.db_path)
                except Exception as e:  # noqa: BLE001
                    reply.error = f"The generated SQL failed to run: {e}"
                    self._evict_sql()
            else:
                reply.error = "The model did not produce a SQL query. Try rephrasing the question."
                self._evict_sql()
            if chart and reply.df is not None and not reply.df.empty:
                reply.chart_spec = get_chart_spec(self.text_llm, reply.df, standalone, use_llm=self.llm_chart_spec)
                reply.fig = build_figure(reply.df, reply.chart_spec)
            if reply.error:
                reply.answer = f"Sorry, I could not answer that. {reply.error}"
            elif chart or not self.llm_answers:
                reply.answer = auto_answer(reply.df, chart)            # the chart/table is the answer: 0 calls
            else:
                reply.answer = write_answer(self.text_llm, standalone, sql, reply.df, self.notes)  # text model
        except Exception as e:  # noqa: BLE001  (rate limits, bad key, malformed SQL, ...)
            reply = Reply(question, standalone, f"Sorry, something went wrong: {e}", error=str(e))
        if cache:
            reply.cache_hits, reply.llm_calls = cache.hits - h0, cache.misses - m0
        reply.seconds = round(time.time() - t0, 2)
        self.history.append({"question": question, "standalone": standalone, "answer": reply.answer})
        return reply


def suggest_questions(text_llm, schema_context: str, n: int = 4) -> List[str]:
    """Ask the small text model for example questions that fit the uploaded tables (+1 cached call)."""
    prompt = (
        f"Here is a description of some database tables:\n{schema_context[:6000]}\n\n"
        f"Write {n} short, varied questions a business user might ask about this data. Each must be answerable "
        "with a single SQL query. Make at least one of them ask for a chart. "
        "Respond with ONLY a JSON array of strings."
    )
    try:
        text = text_llm.complete(prompt).text
        arr = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        return [str(q) for q in arr if str(q).strip()][:n]
    except Exception:  # noqa: BLE001
        return []


def create_assistant(api_key: str, db_path: Path, notes: str = "", cache: Optional[LLMCache] = None,
                     llm_answers: bool = True, llm_chart_spec: bool = False):
    """One-call setup used by app.py. Returns (assistant, schema_context)."""
    sql_llm, text_llm = make_llms(api_key, cache)
    context = build_schema_context(db_path, notes)
    engine = build_query_engine(sql_llm, db_path, context)
    assistant = TableAssistant(sql_llm=sql_llm, text_llm=text_llm, sql_engine=engine, db_path=Path(db_path),
                               notes=notes, llm_answers=llm_answers, llm_chart_spec=llm_chart_spec)
    return assistant, context