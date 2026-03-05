#!/usr/bin/env python3
"""
app.py — Streamlit web interface for Scout.

Run with:
    streamlit run app.py
"""

import os
import json
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import anthropic

from scout import (
    scrape_site,
    extract_products,
    detect_founding_year,
    extract_copyright_year,
    get_wayback_candidates,
    get_earliest_snapshot_year,
    search_founding_year_web,
    fetch_page,
    html_to_text,
    is_parked_page,
    find_discontinued,
    read_group_files,
    match_group,
    extract_outreach_paragraph,
    personalize_paragraph,
    resolve_matched_file,
    display_name,
    extract_address,
    search_restaurants,
    pick_top_restaurants,
    ask_claude_for_restaurants,
)

load_dotenv()
# Load API key from Streamlit secrets if not found in .env (for cloud deployment)
if not os.environ.get("ANTHROPIC_API_KEY"):
    try:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

# ── Session state for batch processing ────────────────────────────────────────
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = []
if "batch_urls" not in st.session_state:
    st.session_state["batch_urls"] = []
if "num_url_fields" not in st.session_state:
    st.session_state["num_url_fields"] = 1


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis function  (logic is unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def run_scout(url: str, log) -> dict:
    """
    Run the full Scout pipeline on a URL.

    `log(message)` is called after each step completes so the UI can display
    live progress. Returns a results dict on success; raises on failure.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Step 1: Scrape current site ──────────────────────────────────────────
    log("Scraping current website...")
    current_text, homepage_html = scrape_site(url)

    if not current_text:
        raise ValueError(
            "Could not extract any text from this site. "
            "It may require a login, block web scrapers, or be built entirely in JavaScript. "
            "Try the main public marketing URL — not a portal or app URL."
        )

    log(f"Extracted {len(current_text):,} characters.")
    log("Extracting current products and services...")
    current_products = extract_products(client, current_text, "current")

    if current_products:
        preview = ", ".join(current_products[:5])
        suffix = "..." if len(current_products) > 5 else ""
        log(f"Found {len(current_products)} product(s): {preview}{suffix}")
    else:
        log("No named products/services found on the current site.")

    # ── Founding year detection (all sources → take minimum) ─────────────────
    log("Detecting company founding year...")
    year_candidates = []

    yr = extract_copyright_year(homepage_html)
    if yr:
        year_candidates.append(("copyright footer", yr))

    yr = detect_founding_year(client, current_text)
    if yr:
        year_candidates.append(("page text", yr))

    yr = get_earliest_snapshot_year(url)
    if yr:
        year_candidates.append(("earliest web archive", yr))

    yr = search_founding_year_web(client, url)
    if yr:
        year_candidates.append(("web search", yr))

    if year_candidates:
        founding_year = min(y for _, y in year_candidates)
        best_source   = next(src for src, y in year_candidates if y == founding_year)
        log(f"Founding year: {founding_year} (from {best_source})")
    else:
        founding_year = None
        log("Founding year not found — using wide Wayback Machine window.")

    if founding_year and founding_year >= 2010:
        wb_from  = f"{founding_year}0101"
        wb_to    = f"{founding_year + 2}1231"
        wb_label = f"{founding_year}–{founding_year + 2}"
    elif founding_year:
        wb_from  = "20060101"
        wb_to    = "20101231"
        wb_label = "2006–2010"
    else:
        wb_from  = "20060101"
        wb_to    = "20201231"
        wb_label = "2006–2020"

    # ── Step 2: Wayback Machine ───────────────────────────────────────────────
    log(f"Fetching Wayback Machine snapshot from {wb_label}...")
    candidates = get_wayback_candidates(url, from_date=wb_from, to_date=wb_to)

    old_products  = []
    old_text      = ""
    archive_url   = None
    timestamp     = None
    domain_stem   = urlparse(url).netloc.replace("www.", "").split(".")[0].lower()

    if not candidates:
        log("No archived snapshots found in that date range.")
    else:
        log(f"Found {len(candidates)} candidate snapshot(s) — checking each...")
        for candidate_url, candidate_ts in candidates:
            year          = candidate_ts[:4]
            old_html      = fetch_page(candidate_url, timeout=25)
            old_text_cand = html_to_text(old_html, max_chars=8000)

            if not old_text_cand or len(old_text_cand) < 300:
                log(f"Skipping {year} snapshot — too little content.")
                continue
            if is_parked_page(old_text_cand):
                log(f"Skipping {year} snapshot — looks like a parked domain page.")
                continue
            if domain_stem not in old_text_cand.lower():
                log(f"Skipping {year} snapshot — company name not found (likely a prior domain owner).")
                continue

            archive_url = candidate_url
            timestamp   = candidate_ts
            old_text    = old_text_cand
            log(f"Valid snapshot found from {year}.")
            break

        if archive_url:
            log(f"Extracted {len(old_text):,} characters from the archived page.")
            log("Extracting archived products and services...")
            old_products = extract_products(client, old_text, f"archived ({timestamp[:4]})")
            if old_products:
                preview = ", ".join(old_products[:5])
                suffix  = "..." if len(old_products) > 5 else ""
                log(f"Found {len(old_products)} archived product(s): {preview}{suffix}")
            else:
                log("No named products/services found in the archived page.")
        else:
            log("No valid snapshot passed all checks.")

    # ── Step 3: Compare product lines ────────────────────────────────────────
    log("Comparing product lines...")
    discontinued      = None
    discontinued_note = None

    if old_products and current_products:
        discontinued = find_discontinued(
            client, old_products, current_products, period_label=wb_label
        )
        if discontinued and timestamp:
            discontinued_note = (
                f"Found on the {timestamp[:4]} archived version of the site "
                f"(Wayback Machine, {wb_label} window)."
            )
        log(f"Discontinued item identified: {discontinued or 'none'}")
    elif not archive_url:
        log("Skipping comparison — no valid archived snapshot was available.")
    else:
        log("Skipping comparison — not enough product data from one or both versions.")

    # ── Step 4: Find address and nearby restaurants (always runs) ────────────
    log("Finding company address...")
    address = extract_address(client, current_text, url)
    restaurants = []
    if address:
        log(f"Address found: {address}")
        log(f"Searching web for business dinner restaurants near {address}...")
        try:
            snippets = search_restaurants(address)
            log(f"Web search returned {len(snippets)} result(s).")
        except Exception as e:
            log(f"Web search error: {e}")
            snippets = []

        if snippets:
            try:
                restaurants = pick_top_restaurants(client, snippets, address)
            except Exception:
                restaurants = []

        if not restaurants:
            log("Web search pipeline returned no results — asking Claude directly...")
            try:
                restaurants = ask_claude_for_restaurants(client, address)
            except Exception as e:
                log(f"Claude restaurant lookup failed: {e}")

        if restaurants:
            log(f"Found {len(restaurants)} restaurant recommendation(s).")
        else:
            log("Could not retrieve restaurant recommendations for this address.")
    else:
        log("Company address not found — skipping restaurant search.")

    # ── Step 5: Read group files ──────────────────────────────────────────────
    log("Loading portfolio group files...")
    groups = read_group_files(script_dir)

    if not groups:
        raise ValueError("No group .md files found in the application folder.")

    log(f"Loaded {len(groups)} group file(s).")

    # ── Step 6: Match to portfolio group ─────────────────────────────────────
    log("Matching to portfolio group...")
    raw_match = match_group(client, current_text, groups)

    if raw_match.upper() == "NO_MATCH":
        log("No portfolio group is a strong fit for this company.")
        return {
            "matched_group":      None,
            "discontinued":       discontinued,
            "discontinued_note":  discontinued_note,
            "outreach_paragraph": None,
            "archive_url":        archive_url,
            "archive_year":       timestamp[:4] if timestamp else None,
            "wb_label":           wb_label,
            "founding_year":      founding_year,
            "address":            address,
            "restaurants":        restaurants,
        }

    matched_file = resolve_matched_file(raw_match, groups)
    group_name   = display_name(matched_file)
    log(f"Best match: {group_name}")

    # ── Step 7: Personalize outreach paragraph ────────────────────────────────
    log("Drafting outreach paragraph...")
    base_paragraph = extract_outreach_paragraph(groups[matched_file])
    personalized   = personalize_paragraph(client, base_paragraph, url, current_text, products=current_products)
    log("Outreach paragraph complete.")

    return {
        "matched_group":      group_name,
        "discontinued":       discontinued,
        "discontinued_note":  discontinued_note,
        "outreach_paragraph": personalized,
        "archive_url":        archive_url,
        "archive_year":       timestamp[:4] if timestamp else None,
        "wb_label":           wb_label,
        "founding_year":      founding_year,
        "address":            address,
        "restaurants":        restaurants,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Valstone theme CSS
# ─────────────────────────────────────────────────────────────────────────────

VALSTONE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body {
    font-family: 'Poppins', sans-serif !important;
}

.stApp {
    background-color: #0B0F1A !important;
    font-family: 'Poppins', sans-serif !important;
}

.main .block-container {
    background-color: #0B0F1A !important;
    color: #E8EDF5 !important;
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1140px !important;
}

/* ── Equal-height result columns ── */
div[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
    gap: 0.75rem !important;
}
div[data-testid="stHorizontalBlock"] > div {
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="stHorizontalBlock"] > div > div[data-testid="stVerticalBlock"] {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
div[data-testid="stHorizontalBlock"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    flex: 1 !important;
}
div[data-testid="stHorizontalBlock"] > div > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] > div {
    height: 100% !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu              { visibility: hidden !important; }
footer                 { visibility: hidden !important; }
[data-testid="stToolbar"]    { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] {
    background-color: #0B0F1A !important;
    border-bottom: 1px solid #171E30 !important;
}

/* ── Hero block ── */
.vs-hero {
    padding: 0.85rem 0 0.65rem 0;
    border-bottom: 1px solid #171E30;
    margin-bottom: 1rem;
}

.vs-eyebrow {
    font-family: 'Poppins', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.28em;
    color: #C97826;
    margin-bottom: 0.3rem;
}

.vs-wordmark {
    font-family: 'Poppins', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #FFFFFF;
    line-height: 1;
    margin-bottom: 0.35rem;
}

.vs-tagline {
    font-family: 'Poppins', sans-serif;
    font-size: 0.8rem;
    color: #4E6080;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── Results section header ── */
.vs-results-header {
    margin: 0.75rem 0 0.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #171E30;
}

.vs-results-eyebrow {
    font-family: 'Poppins', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.28em;
    color: #C97826;
    display: block;
    margin-bottom: 0.2rem;
}

.vs-results-title {
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Poppins', sans-serif !important;
    color: #FFFFFF !important;
}
h2 { font-weight: 700 !important; }
h3 { font-weight: 600 !important; }

/* ── Body text ── */
p, li {
    font-family: 'Poppins', sans-serif;
    color: #8FA3BE;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid #171E30 !important;
    margin: 1.75rem 0 !important;
}

/* ── Input label ── */
.stTextInput label {
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: #4E6080 !important;
}

/* ── Text input field ── */
.stTextInput input {
    background-color: #111827 !important;
    color: #E8EDF5 !important;
    border: 1px solid #1C2640 !important;
    border-radius: 8px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

.stTextInput input:focus {
    border-color: #C97826 !important;
    box-shadow: 0 0 0 3px rgba(201, 120, 38, 0.15) !important;
    outline: none !important;
}

.stTextInput input::placeholder {
    color: #253040 !important;
}

/* ── Button ── */
.stButton button {
    background: linear-gradient(135deg, #B06820, #D4832F) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 50px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 18px rgba(180, 104, 32, 0.22) !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #C97826, #E09040) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(201, 120, 38, 0.38) !important;
}

.stButton button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(180, 104, 32, 0.2) !important;
}

/* ── Secondary buttons (Add URL, Clear Results) ── */
button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: #4E6080 !important;
    border: 1px dashed #1C2640 !important;
    border-radius: 8px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: none !important;
    padding: 0.45rem 1.2rem !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}

button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(201, 120, 38, 0.08) !important;
    border-color: #C97826 !important;
    color: #C97826 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background-color: #111827 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid #1C2640 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    color: #4E6080 !important;
    background-color: transparent !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #8FA3BE !important;
    background-color: rgba(201, 120, 38, 0.06) !important;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(201, 120, 38, 0.12) !important;
    color: #C97826 !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #C97826 !important;
    height: 2px !important;
}

.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Status / progress widget ── */
[data-testid="stStatusWidget"] {
    background-color: #111827 !important;
    border: 1px solid #1C2640 !important;
    border-radius: 12px !important;
}

[data-testid="stStatusWidget"] summary {
    color: #8FA3BE !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] div {
    font-family: 'Poppins', sans-serif !important;
    color: #4E6080 !important;
    font-size: 0.82rem !important;
}

/* ── Bordered containers → Valstone dark cards ── */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #111827 !important;
    border: 1px solid #1C2640 !important;
    border-radius: 14px !important;
}

/* ── Card section label (h4) — styled like "Our Industries" on the site ── */
[data-testid="stVerticalBlockBorderWrapper"] h4 {
    font-size: 0.6rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    color: #C97826 !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.75rem !important;
    border-bottom: 1px solid #1C2640 !important;
}

/* ── Card group name (h3) ── */
[data-testid="stVerticalBlockBorderWrapper"] h3 {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    margin-top: 0 !important;
    letter-spacing: 0.02em !important;
}

/* ── Captions ── */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: #354760 !important;
    font-size: 0.77rem !important;
    font-family: 'Poppins', sans-serif !important;
}

/* ── Alerts (error, warning, info) ── */
[data-testid="stAlert"] {
    background-color: #111827 !important;
    border-radius: 8px !important;
    font-family: 'Poppins', sans-serif !important;
}

[data-testid="stAlert"] p {
    color: #8FA3BE !important;
}

/* ── Links ── */
a {
    color: #C97826 !important;
    font-weight: 500 !important;
    text-decoration: none !important;
}

a:hover {
    color: #E09040 !important;
    text-decoration: underline !important;
}

/* ── Bold text ── */
strong, b {
    color: #D8E2EF;
    font-weight: 600;
}

/* ── Markdown paragraphs inside cards ── */
[data-testid="stVerticalBlockBorderWrapper"] p {
    color: #8FA3BE !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar               { width: 5px; }
::-webkit-scrollbar-track         { background: #0B0F1A; }
::-webkit-scrollbar-thumb         { background: #1C2640; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover   { background: #C97826; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Scout | Valstone", page_icon="🔍", layout="wide")

# Inject Valstone theme
st.markdown(VALSTONE_CSS, unsafe_allow_html=True)

# ── Branded hero header ───────────────────────────────────────────────────────
st.markdown("""
<div class="vs-hero">
    <div class="vs-eyebrow">Valstone</div>
    <div class="vs-wordmark">Scout</div>
    <div class="vs-tagline">Construction &amp; Diversified Materials</div>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
single = st.session_state["num_url_fields"] == 1

for i in range(st.session_state["num_url_fields"]):
    label = "Company URL" if single else f"Company URL {i + 1}"
    st.text_input(
        label,
        key=f"url_{i}",
        placeholder="https://www.example.com",
    )

if st.session_state["num_url_fields"] < 5:
    add_col, _ = st.columns([1, 3])
    with add_col:
        if st.button("+ Add another URL", type="secondary", key="add_url_btn"):
            st.session_state["num_url_fields"] += 1
            st.rerun()

run_clicked = st.button("Run Scout", type="primary", use_container_width=True)

if run_clicked:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error(
            "**ANTHROPIC_API_KEY not found.**  \n"
            "Create a file called `.env` in the same folder as `app.py` and add this line:  \n"
            "```\nANTHROPIC_API_KEY=your-key-here\n```"
        )
        st.stop()

    # ── Collect URLs from individual input fields ────────────────────────────
    urls = []
    for i in range(st.session_state["num_url_fields"]):
        val = st.session_state.get(f"url_{i}", "").strip()
        if not val:
            continue
        if not val.startswith("http"):
            val = "https://" + val
        urls.append(val)

    if not urls:
        st.error("Please enter at least one company URL before running.")
        st.stop()

    # Remove duplicates while keeping the order
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    urls = unique_urls

    # ── Batch processing loop ────────────────────────────────────────────────
    st.session_state["batch_results"] = []
    st.session_state["batch_urls"] = urls
    total = len(urls)

    for idx, url in enumerate(urls, 1):
        domain = urlparse(url).netloc or url

        if total == 1:
            label = f"Running Scout analysis — {domain}"
        else:
            label = f"Analyzing company {idx} of {total}: {domain}"

        with st.status(label, expanded=True) as status:
            def log(msg: str):
                st.write(msg)

            try:
                results = run_scout(url, log)
                st.session_state["batch_results"].append({
                    "url": url,
                    "domain": domain,
                    "results": results,
                    "error": None,
                    "status": "done",
                })
                if total == 1:
                    status.update(label="Analysis complete!", state="complete", expanded=False)
                else:
                    status.update(label=f"Company {idx} of {total} complete: {domain}", state="complete", expanded=False)
            except Exception as e:
                st.session_state["batch_results"].append({
                    "url": url,
                    "domain": domain,
                    "results": None,
                    "error": str(e),
                    "status": "failed",
                })
                if total == 1:
                    status.update(label="Analysis failed", state="error", expanded=False)
                else:
                    status.update(label=f"Company {idx} of {total} failed: {domain}", state="error", expanded=False)

# ── Helper: render one company's results ──────────────────────────────────────
def _render_company(entry, entry_idx):
    """Display results for a single company entry."""
    domain = entry["domain"]

    # If this URL failed, show error and return
    if entry["status"] == "failed":
        st.error(f"Scout could not analyze this company: {entry['error']}")
        return

    results = entry["results"]

    # ── Three-column results layout ───────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1.8])

    with col1:
        with st.container(border=True):
            st.markdown("#### Matched Portfolio Group")
            if results["matched_group"]:
                st.markdown(f"### {results['matched_group']}")
            else:
                st.warning(
                    "No match — this company does not fit any current portfolio group."
                )

    with col2:
        with st.container(border=True):
            st.markdown("#### Discontinued Product")

            if results.get("founding_year"):
                st.caption(f"Est. founded: {results['founding_year']}")
                if results["founding_year"] > 2016:
                    st.warning(
                        f"Founded ~{results['founding_year']} — "
                        "verify age before reaching out."
                    )

            if results["discontinued"]:
                st.markdown(f"**{results['discontinued']}**")
                if results["archive_url"] and results.get("archive_year"):
                    st.caption(
                        f"Archived snapshot used: {results['archive_year']} "
                        f"(Wayback Machine, {results['wb_label']} window)"
                    )
                if results["archive_url"]:
                    st.markdown(f"[View archived page →]({results['archive_url']})")
            else:
                st.write("None identified.")
                if not results.get("archive_url"):
                    st.caption(
                        f"No valid Wayback Machine snapshot was found for the "
                        f"{results['wb_label']} window."
                    )

    with col3:
        with st.container(border=True):
            st.markdown("#### Outreach Paragraph")
            if results["outreach_paragraph"]:
                st.write(results["outreach_paragraph"])
                st.markdown("---")

                safe_text = json.dumps(results["outreach_paragraph"])
                btn_id = f"copy-btn-{entry_idx}"
                components.html(
                    f"""
                    <link
                      href="https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap"
                      rel="stylesheet"
                    />
                    <style>
                      body {{ margin: 0; padding: 0; background: transparent; }}
                    </style>
                    <button
                      id="{btn_id}"
                      onmouseover="
                        this.style.background='linear-gradient(135deg,#C97826,#E09040)';
                        this.style.transform='translateY(-2px)';
                        this.style.boxShadow='0 8px 24px rgba(201,120,38,0.38)';
                      "
                      onmouseout="
                        this.style.background='linear-gradient(135deg,#B06820,#D4832F)';
                        this.style.transform='translateY(0)';
                        this.style.boxShadow='0 4px 18px rgba(180,104,32,0.22)';
                      "
                      onclick="
                        navigator.clipboard.writeText({safe_text}).then(function() {{
                            var btn = document.getElementById('{btn_id}');
                            btn.textContent = 'Copied!';
                            btn.style.background = 'linear-gradient(135deg,#1a6e38,#22a84f)';
                            btn.style.boxShadow  = '0 4px 18px rgba(34,168,79,0.3)';
                            setTimeout(function() {{
                                btn.textContent  = 'Copy to Clipboard';
                                btn.style.background = 'linear-gradient(135deg,#B06820,#D4832F)';
                                btn.style.boxShadow  = '0 4px 18px rgba(180,104,32,0.22)';
                            }}, 2200);
                        }});
                      "
                      style="
                        background: linear-gradient(135deg, #B06820, #D4832F);
                        color: #FFFFFF;
                        border: none;
                        border-radius: 50px;
                        padding: 9px 26px;
                        font-family: 'Poppins', sans-serif;
                        font-weight: 600;
                        font-size: 0.72rem;
                        letter-spacing: 0.13em;
                        text-transform: uppercase;
                        cursor: pointer;
                        box-shadow: 0 4px 18px rgba(180, 104, 32, 0.22);
                        transition: all 0.25s ease;
                      "
                    >Copy to Clipboard</button>
                    """,
                    height=52,
                )
            else:
                st.write("No outreach paragraph generated (no group match).")

    # ── Restaurant recommendations ────────────────────────────────────
    if results.get("address"):
        st.markdown(
            f'<div class="vs-results-header">'
            f'<span class="vs-results-eyebrow">Nearby Restaurants</span>'
            f'<p class="vs-results-title" style="font-size:1rem;">'
            f'{results["address"]}</p></div>',
            unsafe_allow_html=True,
        )
        if results.get("restaurants"):
            r1, r2, r3 = st.columns(3)
            for col, restaurant in zip([r1, r2, r3], results["restaurants"]):
                with col:
                    with st.container(border=True):
                        st.markdown("#### Business Dinner")
                        st.markdown(f"**{restaurant['name']}**")
                        st.write(restaurant["description"])
        else:
            st.caption("Restaurant recommendations could not be retrieved for this address.")
    else:
        st.caption("No company address found on the website — restaurant recommendations skipped.")


# ── Display results from session state ───────────────────────────────────────
if st.session_state.get("batch_results"):
    batch = st.session_state["batch_results"]

    # ── Results header + Clear button ─────────────────────────────────────
    header_col, clear_col = st.columns([4, 1])
    with header_col:
        st.markdown("""
        <div class="vs-results-header">
            <span class="vs-results-eyebrow">Analysis Output</span>
            <p class="vs-results-title">Results</p>
        </div>
        """, unsafe_allow_html=True)
    with clear_col:
        st.markdown("")  # vertical alignment spacing
        if st.button("Clear Results", type="secondary"):
            st.session_state["batch_results"] = []
            st.session_state["batch_urls"] = []
            st.rerun()

    # ── Single company: render directly (no tabs) ─────────────────────────
    if len(batch) == 1:
        _render_company(batch[0], 0)

    # ── Multiple companies: render inside tabs ────────────────────────────
    else:
        tab_labels = [entry["domain"] for entry in batch]
        tabs = st.tabs(tab_labels)
        for tab, entry, idx in zip(tabs, batch, range(len(batch))):
            with tab:
                _render_company(entry, idx)
