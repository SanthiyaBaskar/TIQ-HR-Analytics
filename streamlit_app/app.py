# streamlit_app/app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import io
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# -------------------------
# Config / paths
# -------------------------
st.set_page_config(
    page_title="T-IQ — Talent Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]  # repo root (repo_ready)
DATA_DIR = ROOT / "data"

EXPECTED = {
    "employees": "employees_clean.csv",
    "attrition": "employees_with_attrition_prob.csv",
    "matches": "resume_job_matches.csv",
    "top1": "resume_top1_job.csv",
    "sentiment": "reviews_with_sentiment.csv",
    "jobs": "jobs.csv",
}

# -------------------------
# Helpers & fallbacks
# -------------------------
def safe_read_csv(path: Path, nrows=None):
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return None

def sample_employees(n=1000):
    rng = np.random.default_rng(1)
    ids = np.arange(1, n+1)
    names = [f"Employee {i}" for i in ids]
    ages = rng.integers(21, 60, size=n)
    genders = rng.choice(["F","M","Other"], size=n, p=[0.45,0.45,0.10])
    hire_dates = pd.to_datetime("2015-01-01") + pd.to_timedelta(rng.integers(0, 365*8, size=n), unit="D")
    tenure = ((pd.Timestamp.today() - hire_dates).days / 365).round(2)
    salary = rng.integers(300_000, 1_200_000, size=n)
    satisfaction = rng.random(n).round(3)
    dept = rng.choice(["Engineering","Sales","HR","Finance","Product","Support"], size=n)
    df = pd.DataFrame({
        "employee_id": ids,
        "name": names,
        "age": ages,
        "gender": genders,
        "hire_date": hire_dates.dt.strftime("%Y-%m-%d"),
        "tenure_years": tenure,
        "salary": salary,
        "satisfaction": satisfaction,
        "department": dept
    })
    return df

def nice_number(x):
    try:
        return f"{int(x):,}"
    except:
        return x

# -------------------------
# Session keys init (safe)
# -------------------------
if "last_resume_text" not in st.session_state:
    st.session_state["last_resume_text"] = ""
if "last_match_df" not in st.session_state:
    st.session_state["last_match_df"] = None
if "saved_note" not in st.session_state:
    st.session_state["saved_note"] = None

# -------------------------
# Sidebar / nav (kept minimal)
# -------------------------
logo_path = ROOT / "streamlit_app" / "logo.png"
if logo_path.exists():
    try:
        st.sidebar.image(str(logo_path), width=120)
    except Exception:
        pass

st.sidebar.title("T-IQ HR Analytics")
st.sidebar.caption("Talent Intelligence & Workforce Optimization")

page = st.sidebar.radio("Choose view", [
    "Overview",
    "Attrition Risk",
    "Resume → Job Match",
    "Sentimental analysis",
    "Visualizations",
    "Download Data"
])

# Developer / data status (skip jobs.csv presence messaging)
with st.sidebar.expander("Developer / Data status", expanded=False):
    for k, fname in EXPECTED.items():
        if k == "jobs":
            continue
        p = DATA_DIR / fname
        if p.exists():
            st.success(f"{fname} — {p.stat().st_size // 1024} KB")
        else:
            st.info(f"{fname} — not present")

# small note box in sidebar
st.sidebar.markdown("### Quick note")
note_val = st.sidebar.text_area("Note (saved to session)", key="sidebar_note", height=80)
if st.sidebar.button("Save note"):
    st.session_state["saved_note"] = {"text": note_val, "ts": datetime.utcnow().isoformat()}
    st.sidebar.success("Saved (session)")

# -------------------------
# Header
# -------------------------
st.markdown("<h1 style='margin-bottom:0.1rem;'>🚀 T-IQ — Talent Intelligence</h1>", unsafe_allow_html=True)
st.write("Enterprise-ready HR analytics: attrition, matching, sentimental analysis & dashboards.")
st.markdown("---")

# -------------------------
# Load data (defensive)
# -------------------------
df_emp_raw = safe_read_csv(DATA_DIR / EXPECTED["employees"])
df_attr_raw = safe_read_csv(DATA_DIR / EXPECTED["attrition"])
df_sent_raw = safe_read_csv(DATA_DIR / EXPECTED["sentiment"])
df_jobs_raw = safe_read_csv(DATA_DIR / EXPECTED["jobs"])

df_employees = df_emp_raw if df_emp_raw is not None else sample_employees(1000)

# jobs fallback (silent)
if df_jobs_raw is None:
    df_jobs = pd.DataFrame([
        {"job_id": 1, "job_title": "Data Scientist", "job_desc": "Analyze data, build ML models, Python, pandas, statistics"},
        {"job_id": 2, "job_title": "Backend Engineer", "job_desc": "Develop APIs, backend services, databases, Python/Node"},
        {"job_id": 3, "job_title": "DevOps Engineer", "job_desc": "CI/CD, Docker, Kubernetes, cloud infrastructure"},
        {"job_id": 4, "job_title": "Product Manager", "job_desc": "Product strategy, roadmap, stakeholder management"},
        {"job_id": 5, "job_title": "Data Engineer", "job_desc": "ETL, data pipelines, Spark, SQL"}
    ])
else:
    df_jobs = df_jobs_raw.copy()

# Pre-build TF-IDF on jobs corpus (defensive)
_job_corpus = (df_jobs.get("job_desc", "").astype(str).fillna("") + " " + df_jobs.get("job_title", "").astype(str).fillna("")).tolist()
_tfidf_vectorizer = None
_tfidf_matrix = None
try:
    if len(_job_corpus) > 0:
        _tfv = TfidfVectorizer(stop_words="english", min_df=1)
        _X = _tfv.fit_transform(_job_corpus)
        _tfidf_vectorizer = _tfv
        _tfidf_matrix = _X
except Exception:
    _tfidf_vectorizer = None
    _tfidf_matrix = None

# -------------------------
# Resume parser helpers (robust, restore)
# -------------------------
# Optional dependencies
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx
except Exception:
    docx = None

EMAIL_RE = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-\(\)]{6,}\d)")

def extract_text_from_pdf_bytes(b: bytes) -> str:
    if not pdfplumber:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(b)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception:
        return ""

def extract_text_from_docx_bytes(b: bytes) -> str:
    if not docx:
        return ""
    try:
        from io import BytesIO
        doc = docx.Document(BytesIO(b))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def parse_resume_text_basic(text: str, jobs_skills=None) -> dict:
    text = text or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = lines[0] if lines else ""
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    # simple skills detection
    if jobs_skills is None or len(jobs_skills) == 0:
        common = ["python","sql","aws","docker","kubernetes","pandas","spark","tensorflow","keras","java","node","react","git"]
        skills = [s for s in common if s in text.lower()]
    else:
        skills = [s for s in jobs_skills if s.lower() in text.lower()]
    # dedupe
    return {
        "name": name,
        "emails": list(dict.fromkeys(emails)),
        "phones": list(dict.fromkeys(phones)),
        "skills": list(dict.fromkeys(skills))[:20],
    }

# helper: extract small hint keywords from job_desc
def job_keywords(desc, top_n=5):
    tokens = [t.strip(".,()[]:;\"'").lower() for t in desc.split() if len(t)>3]
    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
        if len(seen) >= top_n:
            break
    return seen

# -------------------------
# Pages
# -------------------------

# OVERVIEW
if page == "Overview":
    st.header("Employee Overview")
    avg_salary = df_employees["salary"].mean() if "salary" in df_employees.columns else np.nan
    avg_satis = df_employees["satisfaction"].mean() if "satisfaction" in df_employees.columns else np.nan
    c1, c2, c3 = st.columns(3)
    c1.metric("Employees", f"{len(df_employees):,}")
    c2.metric("Avg Salary", nice_number(avg_salary))
    c3.metric("Avg Satisfaction", f"{avg_satis:.2f}" if not np.isnan(avg_satis) else "N/A")
    st.subheader("Sample records (first 100 rows)")
    st.dataframe(df_employees.head(100), height=420)

# ATTRITION
elif page == "Attrition Risk":
    st.header("Attrition Risk — Model Results")
    df_attr = df_attr_raw
    if df_attr is None:
        st.warning("Attrition predictions not found. Place `employees_with_attrition_prob.csv` in /data to enable full results.")
        df_attr = sample_employees(200)
    if "attrition_probability" in df_attr.columns:
        top = df_attr.sort_values("attrition_probability", ascending=False).head(20)
        st.subheader("Top 20 high-risk employees")
        st.dataframe(top[["employee_id","name","department","attrition_probability"]].fillna("N/A"))
        st.markdown("**Risk distribution**")
        st.bar_chart(df_attr["attrition_probability"].fillna(0))
        st.markdown("**Quick insights / action points**")
        high = top[top["attrition_probability"] > 0.6]
        if not high.empty:
            st.write(f"- {len(high)} employees have probability > 0.6 — recommend immediate 1:1 and retention steps.")
            for _, r in high.iterrows():
                st.write(f"  - {r.get('name','N/A')} (dept: {r.get('department','N/A')}), prob: {r.get('attrition_probability'):.3f}")
        else:
            st.write("- No employee exceeds 0.6 in the current predictions.")
    else:
        st.info("No `attrition_probability` column found — showing sample table.")
        st.dataframe(df_attr.head(200))

# RESUME → JOB MATCH (RESTORED AND FIXED)
elif page == "Resume → Job Match":
    st.header("Resume — Parse & Job Matching (restored)")

    st.markdown("**Step 1 —** Paste resume text OR upload a resume file (PDF / DOCX / TXT). Then click **Parse resume** to extract fields. After parsing click **Find matching jobs**.")

    left, right = st.columns([2,1])

    with left:
        paste_val = st.text_area("Paste resume text here", value=st.session_state.get("last_resume_text",""), height=230, key="resume_paste_area")
        uploaded = st.file_uploader("Or upload resume (pdf/docx/txt)", type=["pdf","docx","txt"], key="resume_upload")

        # If upload present, parse bytes immediately and set in session (so user can see it)
        uploaded_text = ""
        if uploaded is not None:
            try:
                raw_bytes = uploaded.read()
                suffix = Path(uploaded.name).suffix.lower()
                if suffix == ".pdf":
                    uploaded_text = extract_text_from_pdf_bytes(raw_bytes)
                elif suffix == ".docx":
                    uploaded_text = extract_text_from_docx_bytes(raw_bytes)
                else:
                    try:
                        uploaded_text = raw_bytes.decode("utf-8", errors="ignore")
                    except Exception:
                        uploaded_text = str(raw_bytes)
                if uploaded_text and uploaded_text.strip():
                    # save to session and notify user
                    st.session_state["last_resume_text"] = uploaded_text
                    st.success("Uploaded file parsed and saved in session (preview available above).")
            except Exception as e:
                st.warning("Could not read uploaded file; you can paste the resume text instead.")

        # Parse button: use paste area first, then uploaded/session text
        if st.button("Parse resume"):
            text_to_parse = paste_val.strip() or st.session_state.get("last_resume_text","").strip()
            if not text_to_parse:
                st.warning("Please paste resume text or upload a file first.")
            else:
                parsed = parse_resume_text_basic(text_to_parse, jobs_skills=[])
                st.subheader("Parsed fields")
                st.write(f"**Name (heuristic):** {parsed['name'] or '—'}")
                st.write(f"**Emails:** {', '.join(parsed['emails']) if parsed['emails'] else '—'}")
                st.write(f"**Phones:** {', '.join(parsed['phones']) if parsed['phones'] else '—'}")
                st.write(f"**Top skills (heuristic):** {', '.join(parsed['skills']) if parsed['skills'] else '—'}")
                # save parsed text for matching
                st.session_state["last_resume_text"] = text_to_parse

    with right:
        st.subheader("Parser notes")
        st.markdown("- Heuristic parser: extracts first non-empty line as name, finds emails/phones with regex, and detects common skills.")
        st.markdown("- Uploaded PDF/DOCX parsing requires `pdfplumber`/`python-docx` installed for best results; pasted text always works.")
        st.info("After parsing, click **Find matching jobs** below to get ranked matches.")

    st.markdown("---")
    st.subheader("Find matching jobs")

    # Matching: use session saved parsed text if available, else paste area
    resume_for_matching = st.session_state.get("last_resume_text","").strip() or st.session_state.get("resume_paste_area","").strip()

    if st.button("Find matching jobs"):
        if not resume_for_matching:
            st.warning("No resume text available. Paste or parse a resume first.")
        else:
            # primary: TF-IDF similarity
            matches_df = pd.DataFrame(columns=["job_id","job_title","score","job_desc"])
            if _tfidf_vectorizer is not None and _tfidf_matrix is not None:
                try:
                    q = _tfidf_vectorizer.transform([resume_for_matching])
                    sims = cosine_similarity(q, _tfidf_matrix).ravel()
                    idx = sims.argsort()[::-1]
                    rows = []
                    for i in idx[:20]:
                        rows.append({
                            "job_id": int(df_jobs.iloc[i].get("job_id", i+1)),
                            "job_title": str(df_jobs.iloc[i].get("job_title","")),
                            "score": float(sims[i]),
                            "job_desc": str(df_jobs.iloc[i].get("job_desc",""))
                        })
                    matches_df = pd.DataFrame(rows)
                except Exception:
                    matches_df = pd.DataFrame()  # fallback below
            # fallback: token overlap
            if matches_df.empty:
                def tokens(s):
                    return set([t.strip(".,()[]:;\"'").lower() for t in s.split() if len(t)>2])
                r_tok = tokens(resume_for_matching)
                rows = []
                for i, jd in enumerate(_job_corpus):
                    j_tok = tokens(jd)
                    union = r_tok | j_tok or {"_"}
                    score = len(r_tok & j_tok) / len(union)
                    rows.append({
                        "job_id": int(df_jobs.iloc[i].get("job_id", i+1)),
                        "job_title": str(df_jobs.iloc[i].get("job_title","")),
                        "score": float(score),
                        "job_desc": str(df_jobs.iloc[i].get("job_desc",""))
                    })
                matches_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

            # show results
            if matches_df.empty:
                st.info("No matches found.")
            else:
                st.subheader("Top matches")
                # pretty score formatting
                display_df = matches_df[["job_id","job_title","score"]].copy()
                display_df["score"] = display_df["score"].map(lambda x: f"{x:.4f}")
                st.dataframe(display_df.head(20))
                # save full matches for session (used in summary bullets below)
                st.session_state["last_match_df"] = matches_df.reset_index(drop=True)

    # Instead of printing the same table again, show a compact professional summary of top-3 from last run (if exists)
    if st.session_state.get("last_match_df") is not None:
        last = st.session_state["last_match_df"]
        if not last.empty:
            st.markdown("**Last run — concise summary (top 3)**")
            top3 = last.sort_values("score", ascending=False).head(3).reset_index(drop=True)
            for i, r in top3.iterrows():
                score = float(r["score"])
                st.write(f"**{i+1}. {r['job_title']}** — score {score:.4f}")
                # show short professional hints about why this match may fit
                kw = job_keywords(r.get("job_desc",""), top_n=6)
                if kw:
                    st.write(f"  - Key skills / terms: {', '.join(kw)}")
                # give 1-2 professional action points for candidate
                st.write(f"  - What to highlight on resume: emphasize {', '.join(kw[:3])} and concrete project outcomes.")
            st.markdown("---")
            st.markdown("**Notes:** the table above shows ranked matches; the bullets list the top 3 jobs and the key skills/terms you should emphasize in your resume for a stronger match.")

# SENTIMENTAL ANALYSIS (add output table + points)
elif page == "Sentimental analysis":
    st.header("Employee Sentimental Analysis & Reviews")
    df_sent = df_sent_raw if df_sent_raw is not None else None
    if df_sent is None:
        st.warning("Reviews file not found. Place `reviews_with_sentiment.csv` in /data for full results.")
        df_sent = pd.DataFrame({
            "review_text":["Great place to work","Needs better communication","Loved the team"],
            "sentiment_label":["positive","negative","positive"],
            "rating":[5,2,5]
        })

    if "sentiment_label" in df_sent.columns:
        counts = df_sent["sentiment_label"].value_counts().rename_axis("label").reset_index(name="count")
        fig = px.bar(counts, x="label", y="count", title="Sentiment distribution")
        st.plotly_chart(fig, use_container_width=True)

        # explicit output table
        st.subheader("Sentimental analysis — output table (sample)")
        show_cols = [c for c in ["review_text","sentiment_label","rating"] if c in df_sent.columns]
        st.dataframe(df_sent[show_cols].head(200))

        # enhanced insights (added)
        total = len(df_sent)
        pos = int((df_sent["sentiment_label"] == "positive").sum()) if "sentiment_label" in df_sent.columns else 0
        neg = int((df_sent["sentiment_label"] == "negative").sum()) if "sentiment_label" in df_sent.columns else 0
        neu = int((df_sent["sentiment_label"] == "neutral").sum()) if "sentiment_label" in df_sent.columns else 0
        avg_rating = df_sent["rating"].dropna().mean() if "rating" in df_sent.columns else None

        st.markdown("**Quick insights / action points**")
        st.write(f"- Total reviews: **{total}**. Positive: **{pos}**, Neutral: **{neu}**, Negative: **{neg}**.")
        if avg_rating is not None:
            st.write(f"- Average rating: **{avg_rating:.2f}** (check rating distribution below).")
        # top negative samples
        neg_samples = df_sent[df_sent["sentiment_label"] == "negative"]["review_text"].dropna().head(3).tolist() if "sentiment_label" in df_sent.columns else []
        if neg_samples:
            st.write("- Representative negative comments (investigate):")
            for t in neg_samples:
                st.write(f"  - {t[:200]}")
        # recommended next steps
        st.write("- Recommended next steps:")
        st.write("  1. Investigate top negative reviews for root causes (team, manager, process).")
        st.write("  2. Cross-reference low-satisfaction employees with negative reviews and high attrition probability.")
        st.write("  3. Use sentiment trends monthly to measure the impact of interventions.")
    else:
        st.info("No `sentiment_label` column found.")
        st.dataframe(df_sent.head(50))

# VISUALIZATIONS
elif page == "Visualizations":
    st.header("Visualizations — Quick EDA")
    df_emp = df_employees
    left, right = st.columns([2,3])
    with left:
        cat_col = st.selectbox("Categorical column", ["department","gender"], index=0)
        dist = df_emp[cat_col].value_counts().rename_axis(cat_col).reset_index(name="count")
        fig_l = px.bar(dist, x=cat_col, y="count", title=f"Distribution — {cat_col}")
        st.plotly_chart(fig_l, use_container_width=True)
        st.caption("Left chart: distribution for the selected categorical column.")
    with right:
        num_col = st.selectbox("Numeric column for relationship", ["salary","tenure_years"], index=0)
        sample_df = df_emp.sample(min(800, len(df_emp)), random_state=1)
        fig_r = px.scatter(sample_df, x="employee_id", y=num_col, title=f"{num_col} vs employee_id")
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption("Right chart: scatter for numeric spread/outliers.")
    st.subheader("Correlation (numeric)")
    numeric_cols = df_emp.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df_emp[numeric_cols].corr().round(2)
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.write("- Correlation values range -1 to 1, showing linear relationships.")
    else:
        st.info("Not enough numeric columns for correlation matrix.")

# DOWNLOADS
elif page == "Download Data":
    st.header("Download prepared outputs")
    for key, fname in EXPECTED.items():
        if key == "jobs":
            continue
        p = DATA_DIR / fname
        if p.exists():
            st.markdown(f"**{fname}** — {p.stat().st_size // 1024} KB")
            with open(p, "rb") as fh:
                st.download_button(f"Download {fname}", fh, file_name=fname)
        else:
            st.info(f"{fname} not present")

# saved note show
if st.session_state.get("saved_note"):
    saved = st.session_state["saved_note"]
    st.markdown(f"**Saved note (UTC):** {saved['ts']}")
    st.write(saved["text"])
