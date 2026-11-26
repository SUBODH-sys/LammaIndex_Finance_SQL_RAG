"""
app.py - Streamlit UI.
Run:  streamlit run app.py
All logic lives in main.py; this file only handles uploads, session state and rendering.
"""
import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import main as core

# Load GROQ_API_KEY from the project's .env (next to app.py, or any parent folder / current folder)
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

st.set_page_config(page_title="Talk to Your Tables", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------------
# Session state (Streamlit re-runs this script on every interaction)
# ---------------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "workdir": tempfile.mkdtemp(prefix="talk_to_tables_"),   # private per browser session
        "upload_sig": None,
        "db_path": None,
        "tables": {},                 # {table: [columns]}
        "assistant": None,
        "assistant_sig": None,
        "schema_context": "",
        "messages": [],               # [{"role": "user"/"assistant", "content": str, "reply": Reply|None}]
        "suggestions": [],
        "pending_q": None,
        "cache": core.LLMCache(core.DEFAULT_CACHE_PATH),
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()
S = st.session_state


def _default_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:  # no secrets.toml
        pass
    return os.getenv("GROQ_API_KEY", "")


def _digest(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    env_key = _default_api_key()
    if env_key:
        st.success("Groq API key loaded from .env", icon="🔑")
        with st.expander("Use a different key"):
            typed_key = st.text_input("Groq API key", type="password", key="override_key")
    else:
        st.warning("No GROQ_API_KEY found in .env, secrets or environment.")
        typed_key = st.text_input("Groq API key", type="password",
                                  help="Get one free at console.groq.com. Never stored on disk.")
    api_key = typed_key or env_key

    st.caption(f"SQL model: `{core.SQL_MODEL}`  \nAnswer model: `{core.TEXT_MODEL}`")

    st.subheader("1 · Upload your tables")
    uploads = st.file_uploader("CSV or Excel files", type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"],
                               accept_multiple_files=True)

    st.subheader("2 · Notes about your data (optional)")
    notes = st.text_area(
        "Units, meanings, join rules ...", height=140, placeholder=core.EXAMPLE_NOTES,
        help="The app already profiles your columns automatically. Add anything it cannot know: units, "
             "what a column means, how tables relate. This is the biggest accuracy boost.")

    st.subheader("3 · Options")
    llm_answers = st.toggle("Write answers with the LLM", value=True,
                            help="Off = fast mode: only the SQL call is made; the table/chart is the answer.")
    llm_chart_spec = st.toggle("Let the LLM choose chart type", value=False,
                               help="Off = free rule-based chart choice. On = +1 small-model call per chart.")

    c1, c2 = st.columns(2)
    if c1.button("🧹 Clear chat", use_container_width=True):
        S.messages = []
        if S.assistant:
            S.assistant.reset()
        st.rerun()
    if c2.button("🗑️ Clear cache", use_container_width=True):
        S.cache.clear()
        st.toast("LLM cache cleared")
    st.caption(f"LLM cache: {S.cache.stats()}")


# ---------------------------------------------------------------------------------
# Build / rebuild the database when the uploaded files change
# ---------------------------------------------------------------------------------
st.title("📊 Talk to Your Tables")
st.caption("Upload any CSV / Excel files, then ask questions in plain English. "
           "Say *plot*, *chart* or *graph* to get an interactive chart.")

if not uploads:
    st.info("👈 Upload one or more CSV / Excel files in the sidebar to get started. "
            "Each file (or Excel sheet) becomes a table; you can ask questions across them.")
    st.stop()

files = {f.name: f.getvalue() for f in uploads}
upload_sig = _digest(*[x for name in sorted(files) for x in (name, _digest(files[name]))])

if upload_sig != S.upload_sig:
    with st.spinner("Loading files into a database..."):
        try:
            frames = core.load_uploads(files)
            db_path = Path(S.workdir) / f"data_{upload_sig[:10]}.db"
            S.tables = core.build_database(frames, db_path)
            old = S.db_path
            S.db_path, S.upload_sig = db_path, upload_sig
            S.assistant, S.assistant_sig = None, None
            S.messages, S.suggestions = [], []
            if old and Path(old).exists():
                Path(old).unlink(missing_ok=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load the files: {e}")
            st.stop()

# ---------------------------------------------------------------------------------
# Show tables + data dictionary
# ---------------------------------------------------------------------------------
with st.expander(f"📚 {len(S.tables)} table(s) loaded", expanded=not S.messages):
    tabs = st.tabs(list(S.tables))
    for tab, t in zip(tabs, S.tables):
        with tab:
            st.caption(f"Columns: {', '.join(S.tables[t])}")
            st.dataframe(core.table_preview(S.db_path, t, 10))

# ---------------------------------------------------------------------------------
# Create the assistant (needs the API key)
# ---------------------------------------------------------------------------------
if not api_key:
    st.warning("Enter your Groq API key in the sidebar to start asking questions.")
    st.stop()

assistant_sig = _digest(api_key, S.upload_sig, notes)
if S.assistant is None or S.assistant_sig != assistant_sig:
    with st.spinner("Profiling your tables and preparing the models..."):
        try:
            history = S.assistant.history if S.assistant else []
            S.assistant, S.schema_context = core.create_assistant(api_key, S.db_path, notes, S.cache)
            S.assistant.history = history          # keep the conversation when only the key/notes change
            S.assistant_sig = assistant_sig
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not start the assistant: {e}")
            st.stop()

assistant = S.assistant
assistant.llm_answers = llm_answers
assistant.llm_chart_spec = llm_chart_spec

with st.expander("🧾 Data dictionary sent to the SQL model"):
    st.code(S.schema_context, language="text")


# ---------------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------------
def _set_pending(q: str) -> None:
    S.pending_q = q


with st.expander("💡 Try asking", expanded=not S.messages):
    if st.button("Suggest questions for my data"):
        with st.spinner("Thinking of questions..."):
            S.suggestions = core.suggest_questions(assistant.text_llm, S.schema_context)
        if not S.suggestions:
            st.warning("Could not generate suggestions, just type your own question below.")
    for i, q in enumerate(S.suggestions):
        st.button(q, key=f"sug_{i}", on_click=_set_pending, args=(q,))


# ---------------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------------
def render_reply(reply: core.Reply, key: str) -> None:
    if reply.error:
        st.error(reply.answer)
    else:
        st.markdown(reply.answer)
    if reply.fig is not None:
        st.plotly_chart(reply.fig, key=f"fig_{key}")
    if reply.df is not None and not reply.df.empty:
        with st.expander(f"Data ({len(reply.df)} rows)", expanded=reply.fig is None and len(reply.df) > 1):
            st.dataframe(reply.df)
    if reply.sql:
        with st.expander("SQL"):
            st.code(reply.sql, language="sql")
    notes_line = f"{reply.llm_calls} LLM calls · {reply.cache_hits} cache hits · {reply.seconds}s"
    if reply.standalone_question != reply.question:
        notes_line = f"Interpreted as: “{reply.standalone_question}” · " + notes_line
    st.caption(notes_line)


for i, m in enumerate(S.messages):
    with st.chat_message(m["role"]):
        if m["reply"] is not None:
            render_reply(m["reply"], str(i))
        else:
            st.markdown(m["content"])

prompt = st.chat_input("Ask a question about your tables...")
if not prompt and S.pending_q:
    prompt = S.pending_q
S.pending_q = None

if prompt:
    S.messages.append({"role": "user", "content": prompt, "reply": None})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = assistant.ask(prompt)
        render_reply(reply, str(len(S.messages)))
    S.messages.append({"role": "assistant", "content": reply.answer, "reply": reply})