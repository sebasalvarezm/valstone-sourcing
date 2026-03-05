#!/usr/bin/env python3
"""
scout.py — Company research and outreach matching tool.

Usage:
    python scout.py <company_url>

Example:
    python scout.py https://www.example.com

Dependencies:
    pip install requests beautifulsoup4 anthropic

Requires:
    ANTHROPIC_API_KEY environment variable to be set.
"""

import sys
import os
import glob
import json
import re
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import anthropic
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# Sub-pages to crawl after the homepage for richer product coverage
CRAWL_PATHS = [
    "/about", "/about-us", "/products", "/solutions",
    "/services", "/features", "/platform", "/software",
    "/company", "/company.htm", "/company.html",
    "/location", "/locations", "/offices",
]

# Text signals that indicate a parked, placeholder, or unrelated domain page
PARKING_SIGNALS = [
    "domain for sale", "buy this domain", "parked by",
    "1&1 internet", "1and1", "namecheap", "register.com",
    "sedo.com", "godaddy", "this domain has been registered",
    "under construction", "free website builder",
    "web hosting provider", "this web page is parked",
    "domain parking", "domain registrar",
]

# Subdomains that indicate a customer portal rather than the marketing site
PORTAL_PREFIXES = ("my.", "app.", "login.", "portal.", "auth.", "account.", "dashboard.")


# ---------------------------------------------------------------------------
# Helpers: web fetching and text extraction
# ---------------------------------------------------------------------------

def fetch_page(url: str, timeout: int = 20, retries: int = 2) -> str:
    """
    Fetch raw HTML from a URL.
    - Retries on timeout with exponential backoff.
    - Returns empty string if the request ends up on a portal/login subdomain.
    - Returns empty string on any other failure (silently — caller handles messaging).
    """
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            # Detect redirect to a customer portal (e.g. tronia.com → my.tronia.com)
            if _is_portal_redirect(url, r.url):
                return ""
            r.raise_for_status()
            return r.text
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return ""
        except Exception:
            return ""
    return ""


def _is_portal_redirect(original_url: str, final_url: str) -> bool:
    """
    Return True if the request was silently redirected from the main marketing
    domain to a customer-portal subdomain (my.*, app.*, login.*, etc.).
    """
    orig_netloc = urlparse(original_url).netloc.lower()
    final_netloc = urlparse(final_url).netloc.lower()
    if orig_netloc == final_netloc:
        return False
    return any(final_netloc.startswith(p) for p in PORTAL_PREFIXES)


def html_to_text(html: str, max_chars: int = 4000) -> str:
    """
    Strip HTML tags and return clean plain text.
    - Removes script, style, nav, footer, header, aside blocks.
    - Strips non-printable / binary characters.
    - Caps output at max_chars.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = " ".join(text.split())
    # Remove binary / non-printable characters that confuse Claude
    text = "".join(c for c in text if c.isprintable())
    return text[:max_chars]


def is_parked_page(text: str) -> bool:
    """Return True if the page looks like a domain-parking or placeholder page."""
    lower = text.lower()
    return any(signal in lower for signal in PARKING_SIGNALS)


def extract_copyright_year(html: str):
    """
    Scan raw HTML (before tag-stripping) for copyright year patterns.
    Copyright years live in the <footer> which we normally strip — this
    reads them from the raw source instead.

    Returns the earliest year found (int) only if it looks like a founding
    year (i.e. it's at least 2 years before today). Returns None if the
    only year found is a current-year notice like '© 2026'.
    """
    if not html:
        return None
    # Match: © 2012, &copy; 2012, (c) 2012, Copyright 2012, Copyright © 2012-2024
    pattern = r'(?:©|&copy;|\(c\)|copyright)[^0-9]{0,10}(\d{4})'
    matches = re.findall(pattern, html, re.IGNORECASE)
    import datetime
    current_year = datetime.datetime.now().year
    # Only keep years that are clearly historical (at least 2 years old)
    years = [int(y) for y in matches if 1990 <= int(y) <= current_year - 2]
    return min(years) if years else None


def scrape_site(base_url: str, max_total_chars: int = 12000):
    """
    Fetch the homepage PLUS key sub-pages and return (combined_text, homepage_html).

    - Tries the URL as-is first, then the www / non-www variant as a fallback.
    - Crawls /about, /products, /solutions, etc. to find product names that
      may not appear on the homepage.
    - Stops once max_total_chars of content is collected.
    - Returns ("", "") if no content could be retrieved.
    """
    base = base_url.rstrip("/")
    parsed = urlparse(base)
    netloc = parsed.netloc

    # Build www / non-www fallback
    if netloc.startswith("www."):
        alt_netloc = netloc[4:]
    else:
        alt_netloc = "www." + netloc
    alt_base = f"{parsed.scheme}://{alt_netloc}{parsed.path}".rstrip("/")

    # Try to get a working homepage
    working_base = None
    homepage_html = ""
    homepage_text = ""
    for variant in [base, alt_base]:
        html = fetch_page(variant)
        if html:
            text = html_to_text(html)
            if len(text) > 300:
                working_base = variant
                homepage_html = html        # keep raw HTML for copyright scanning
                homepage_text = text
                break

    if not working_base:
        return "", ""

    collected = [homepage_text]
    total = len(homepage_text)

    # Crawl sub-pages for richer product/service coverage
    for path in CRAWL_PATHS:
        if total >= max_total_chars:
            break
        html = fetch_page(working_base + path)
        if not html:
            continue
        text = html_to_text(html)
        if len(text) > 200:
            collected.append(text)
            total += len(text)

    return " ".join(collected)[:max_total_chars], homepage_html


# ---------------------------------------------------------------------------
# Helpers: Wayback Machine
# ---------------------------------------------------------------------------

def get_wayback_candidates(url: str, from_date: str = "20060101", to_date: str = "20201231"):
    """
    Query Wayback Machine CDX API for snapshots of the domain
    captured between from_date and to_date (YYYYMMDD strings).
    Retries up to 3 times with increasing backoff on timeout.
    Returns a list of (archive_url, timestamp) tuples (oldest first),
    or an empty list if none found.
    """
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path

    cdx_endpoint = (
        "http://web.archive.org/cdx/search/cdx"
        f"?url={domain}&output=json"
        f"&from={from_date}&to={to_date}"
        "&limit=15&filter=statuscode:200"
        "&collapse=timestamp:4"
        "&fl=timestamp,original"
    )

    for attempt in range(3):
        try:
            r = requests.get(cdx_endpoint, timeout=30)
            r.raise_for_status()
            data = r.json()
            if len(data) < 2:
                return []
            # data[0] is the header row; data[1:] are results (oldest first)
            return [
                (f"https://web.archive.org/web/{row[0]}/{row[1]}", row[0])
                for row in data[1:]
            ]
        except requests.exceptions.Timeout:
            if attempt < 2:
                wait = 3 * (attempt + 1)
                print(f"         [warn] Wayback CDX timed out, retrying in {wait}s ({attempt + 2}/3)...")
                time.sleep(wait)
            else:
                print("         [warn] Wayback Machine CDX API timed out after 3 attempts.")
                return []
        except Exception as e:
            print(f"         [warn] Wayback Machine lookup failed: {e}")
            return []

    return []


def get_earliest_snapshot_year(url: str):
    """
    Query the Wayback Machine CDX API for the OLDEST available snapshot of
    this domain (no date range filter), and return just the year.

    This gives a reliable lower bound on company age — if the first archived
    page is from 2004, the company definitely existed in 2004.

    Returns an integer year, or None on any failure.
    """
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path

    cdx_endpoint = (
        "http://web.archive.org/cdx/search/cdx"
        f"?url={domain}&output=json&limit=1"
        "&filter=statuscode:200&fl=timestamp"
    )

    for attempt in range(3):
        try:
            r = requests.get(cdx_endpoint, timeout=20)
            r.raise_for_status()
            data = r.json()
            # data[0] is the header row ["timestamp"]; data[1] is the first result
            if len(data) < 2:
                return None
            timestamp = data[1][0]  # e.g. "20040318123045"
            year = int(timestamp[:4])
            if 1996 <= year <= 2030:
                return year
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            return None
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------------
# Helpers: Claude calls
# ---------------------------------------------------------------------------

def _call_claude(client: anthropic.Anthropic, max_retries: int = 4, **kwargs):
    """
    Wrapper around client.messages.create() with automatic retry logic.

    Retries on transient server-side errors:
      - 529  overloaded_error  (Anthropic servers under heavy load)
      - 500  internal_server_error
      - 503  service_unavailable

    Backoff schedule: 5s, 10s, 20s, 40s  (doubles each attempt).
    Raises the original exception after all retries are exhausted.
    """
    RETRYABLE = {500, 503, 529}
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code in RETRYABLE and attempt < max_retries:
                wait = 5 * (2 ** attempt)          # 5 → 10 → 20 → 40 seconds
                print(f"         [warn] Claude API {e.status_code} — retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
    # unreachable, but satisfies linters
    raise RuntimeError("Retry loop exited unexpectedly")


def extract_products(client: anthropic.Anthropic, text: str, label: str) -> list:
    """
    Ask Claude to extract product and service names from page text.
    Returns a list of strings.
    """
    if not text:
        return []

    # Use a strict/branded prompt for archived pages (to find named old products)
    # and a permissive prompt for current pages (to find any product/service name)
    if "archived" in label:
        content_prompt = (
            f"Extract named products and services from this {label} website text.\n"
            "PRIORITY 1: Proprietary or branded product names — items with a specific name "
            "the company gave them (e.g. 'ProSuite', 'DataBridge 2.0', 'FieldMotion Go'). "
            "These are the most valuable.\n"
            "PRIORITY 2: Specific service lines with a distinct name (e.g. 'Managed Print Service', "
            "'24/7 Emergency Support Programme').\n"
            "PRIORITY 3 (fallback only, if nothing else found): A unique company tagline, "
            "positioning statement, or notable capability that was clearly prominent at the time "
            "(e.g. 'First cloud-based CAFM for SMEs').\n\n"
            "Do NOT include: generic categories ('consulting', 'software development', 'support'), "
            "company name variants, or vague descriptions.\n\n"
            "Return a JSON array of strings only. No commentary.\n"
            "If you find nothing specific, return an empty array: []\n\n"
            f"Text:\n{text}"
        )
    else:
        content_prompt = (
            f"Extract every distinct product name and service name from this {label} website text.\n"
            "Return a JSON array of strings only. No commentary, no explanation.\n"
            "If you find nothing, return an empty array: []\n\n"
            f"Text:\n{text}"
        )

    resp = _call_claude(client,
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": content_prompt}]
    )

    raw = resp.content[0].text.strip()
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return []


def detect_founding_year(client: anthropic.Anthropic, text: str):
    """
    Ask Claude to extract the company's founding year from website text.
    Returns an integer year, or None if it cannot be determined with confidence.
    """
    resp = _call_claude(client,
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": (
                "What year was this company founded? "
                "Look for clues like 'founded in', 'established', 'since XXXX', "
                "or founding stories in 'About Us' sections.\n\n"
                "Important: do NOT return years from the last 2 years — those are almost "
                "always recent site content (news, certifications, awards), not founding dates. "
                "If the only year clues you find are recent, return null instead.\n\n"
                "Return only the 4-digit year as a plain number (e.g. 2014). "
                "If you cannot determine it with reasonable confidence, return null.\n\n"
                f"Text:\n{text}"
            )
        }]
    )
    raw = resp.content[0].text.strip()
    match = re.search(r"\b(19|20)\d{2}\b", raw)
    if match:
        year = int(match.group())
        import datetime
        current_year = datetime.datetime.now().year
        # Reject years that are too recent — they almost always reflect recent
        # site content (awards, certifications, roadmap items), not a founding date.
        # This mirrors the same filter used in extract_copyright_year().
        if 1900 <= year <= current_year - 2:
            return year
    return None


def search_founding_year_web(client: anthropic.Anthropic, url: str):
    """
    Search DuckDuckGo for the company's founding year and ask Claude to
    extract it from the search results.

    Useful for established companies whose website doesn't mention founding
    history prominently, but who appear in industry directories or news.

    Returns an integer year, or None if nothing reliable is found.
    """
    try:
        from ddgs import DDGS
        import datetime
        parsed = urlparse(url)
        # Extract the company stem (e.g. "givenhansco" from "givenhansco.com")
        stem = parsed.netloc.replace("www.", "").split(".")[0]
        query = f"{stem} company founded year established history"
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=5))
        if not results:
            return None

        snippets = "\n\n".join(
            f"Result {i+1}: {r.get('title','')}\n{r.get('body', r.get('snippet',''))}"
            for i, r in enumerate(results)
        )

        resp = _call_claude(client,
            model="claude-sonnet-4-6",
            max_tokens=30,
            messages=[{
                "role": "user",
                "content": (
                    f"Based on these web search results, what year was the company "
                    f"at {url} founded or established?\n\n"
                    "Return only a 4-digit year if you are confident (e.g. 1998). "
                    "Do not return years from the last 2 years. "
                    "If you cannot determine it, return null.\n\n"
                    f"Search results:\n{snippets}"
                )
            }]
        )
        raw = resp.content[0].text.strip()
        match = re.search(r"\b(19|20)\d{2}\b", raw)
        if match:
            year = int(match.group())
            current_year = datetime.datetime.now().year
            if 1900 <= year <= current_year - 2:
                return year
    except Exception:
        pass
    return None


def find_discontinued(
    client: anthropic.Anthropic,
    old_products: list,
    current_products: list,
    period_label: str = "2006–2010"
) -> str:
    """
    Ask Claude to identify one product present on the old site
    but clearly absent from the current site.
    Returns a single product name as a string.
    """
    resp = _call_claude(client,
        model="claude-sonnet-4-6",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": (
                "Below are two lists of products/services from the same company at different points in time.\n\n"
                f"OLD SITE ({period_label}):\n"
                f"{json.dumps(old_products, indent=2)}\n\n"
                "CURRENT SITE:\n"
                f"{json.dumps(current_products, indent=2)}\n\n"
                "Identify ONE product or service that was present on the old site "
                "but is clearly absent from the current site. "
                "Pick the most specific and interesting one — not a generic category.\n"
                "Return only the product/service name. No explanation."
            )
        }]
    )
    return resp.content[0].text.strip()


def match_group(
    client: anthropic.Anthropic,
    current_text: str,
    groups: dict
) -> str:
    """
    Ask Claude to pick the best-fit group file for this company.
    Returns the filename (e.g. 'mining.md'), or 'NO_MATCH' if none fit well.
    """
    summaries = "\n\n".join(
        f"FILE: {name}\n{content[:700]}"
        for name, content in groups.items()
    )

    resp = _call_claude(client,
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                "Based on this company's website content, which group file is the best fit?\n\n"
                f"COMPANY WEBSITE:\n{current_text[:8000]}\n\n"
                f"GROUP FILES:\n{summaries}\n\n"
                "If the company is a clear fit for one of the groups, return only the exact filename (e.g. mining.md).\n"
                "If none of the groups are a reasonable fit, return exactly: NO_MATCH\n"
                "No explanation."
            )
        }]
    )
    return resp.content[0].text.strip().strip('"').strip("'")


def personalize_paragraph(
    client: anthropic.Anthropic,
    paragraph: str,
    url: str,
    current_text: str,
    products: list = None
) -> str:
    """
    Ask Claude to add a company-specific reference to the outreach paragraph.
    If a product list is provided, Claude is directed to name a specific product.
    """
    products_hint = ""
    if products:
        products_hint = (
            f"\nThe company's specific named products include: {', '.join(products[:6])}. "
            "If possible, mention one of these by name rather than describing the company generically.\n"
        )

    resp = _call_claude(client,
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"Here is an outreach paragraph. Add one company-specific reference "
                f"to the company at {url} that makes it feel written for them specifically.\n"
                f"{products_hint}\n"
                "Rules:\n"
                "- Do NOT rewrite the paragraph\n"
                "- Do NOT change the structure or length meaningfully\n"
                "- Keep the tone identical\n"
                "- Prefer mentioning a specific product name or niche market over generic industry descriptions\n"
                "- Do NOT use em dashes (—) anywhere in the output\n\n"
                f"PARAGRAPH:\n{paragraph}\n\n"
                f"COMPANY CONTEXT:\n{current_text[:2500]}\n\n"
                "Return only the modified paragraph. Nothing else."
            )
        }]
    )
    result = resp.content[0].text.strip()
    # Safety net: replace any em dashes that slipped through (including from
    # the base template) with a comma, preserving sentence flow.
    result = re.sub(r"\s*—\s*", ", ", result)
    return result


# ---------------------------------------------------------------------------
# Helpers: address extraction and restaurant search
# ---------------------------------------------------------------------------

def extract_address(client: anthropic.Anthropic, current_text: str, base_url: str):
    """
    Try to find the company's physical address.

    1. Ask Claude to extract from already-scraped text.
    2. Re-fetch homepage with footer preserved.
    3. Crawl homepage links to discover real contact/about page URLs.
    4. Try hardcoded fallback paths (/contact, /get-in-touch, etc.).
    5. DuckDuckGo web search.
    6. Final fallback: ask Claude for city/country only (e.g. "Reno, NV").

    Returns an address string or None if no location could be found.
    """
    def _ask_claude_for_address(text: str):
        if not text:
            return None
        resp = _call_claude(client,
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    "Extract the company's HEADQUARTERS or main office street address from this text.\n"
                    "If multiple addresses appear, use these priority rules:\n"
                    "1. An address explicitly labeled 'headquarters', 'HQ', 'corporate office', or 'main office'\n"
                    "2. The address associated with the company's home city or founding location\n"
                    "3. If no label distinguishes them, return the first address listed\n\n"
                    "Return only the full street address as a single line "
                    "(e.g. '123 Main St, Denver, CO 80202'). "
                    "Do not include P.O. boxes or branch/satellite office addresses when a headquarters is identifiable. "
                    "If no street address is present, return exactly: null\n\n"
                    f"Text:\n{text[:4000]}"
                )
            }]
        )
        raw = resp.content[0].text.strip()
        if raw.lower() in ("null", "none", "n/a", ""):
            return None
        # Sanity check: must look like an address (has a digit and a comma or state abbrev)
        if re.search(r'\d', raw) and len(raw) > 10:
            return raw
        return None

    # Attempt 1: existing scraped text (footers already stripped)
    address = _ask_claude_for_address(current_text)
    if address:
        return address

    # Attempt 1b: re-fetch homepage WITH footer preserved (addresses often live there)
    base = base_url.rstrip("/")
    try:
        homepage_html = fetch_page(base)
        if homepage_html:
            soup = BeautifulSoup(homepage_html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            footer_text = soup.get_text(separator=" ", strip=True)
            footer_text = " ".join(footer_text.split())[:4000]
            if footer_text and footer_text != current_text[:len(footer_text)]:
                address = _ask_claude_for_address(footer_text)
                if address:
                    return address
    except Exception:
        pass

    # Attempt 2a: discover real contact/about links from homepage
    try:
        homepage_html_links = fetch_page(base)
        if homepage_html_links:
            soup_links = BeautifulSoup(homepage_html_links, "html.parser")
            keywords = ("contact", "reach", "touch", "find-us", "location", "office", "about")
            discovered_urls = []
            for a in soup_links.find_all("a", href=True):
                href = a["href"]
                if any(k in href.lower() for k in keywords):
                    full = href if href.startswith("http") else base + "/" + href.lstrip("/")
                    if full not in discovered_urls and full != base and full != base + "/":
                        discovered_urls.append(full)
            for disc_url in discovered_urls[:6]:
                html = fetch_page(disc_url)
                if not html:
                    continue
                contact_text = html_to_text(html, max_chars=3000)
                if len(contact_text) > 100:
                    address = _ask_claude_for_address(contact_text)
                    if address:
                        return address
    except Exception:
        pass

    # Attempt 2b: hardcoded path fallback
    for path in ("/contact", "/contact-us", "/contactus", "/get-in-touch",
                 "/reach-us", "/find-us", "/about-us", "/about",
                 "/offices", "/locations"):
        html = fetch_page(base + path)
        if not html:
            continue
        contact_text = html_to_text(html, max_chars=3000)
        if len(contact_text) > 100:
            address = _ask_claude_for_address(contact_text)
            if address:
                return address

    # Attempt 3: web search via DuckDuckGo (use full domain to avoid wrong-company matches)
    try:
        from ddgs import DDGS
        parsed = urlparse(base_url)
        domain = parsed.netloc.replace("www.", "")
        query = f'"{domain}" headquarters address location'
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=5))
        if results:
            snippets = "\n\n".join(
                f"Result {i+1}: {r.get('title','')}\n{r.get('body', r.get('snippet',''))}"
                for i, r in enumerate(results)
            )
            # Use a domain-aware prompt so Claude ignores other companies
            resp = _call_claude(client,
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": (
                        f"I need the headquarters street address for the company whose website is {domain}.\n"
                        "Below are web search results. ONLY extract an address that clearly belongs to this specific company.\n"
                        "Ignore addresses for other companies with similar names.\n\n"
                        "Return only the full street address as a single line "
                        "(e.g. '123 Main St, Denver, CO 80202'). "
                        "If no address for this specific company is found, return exactly: null\n\n"
                        f"Search results:\n{snippets[:4000]}"
                    )
                }]
            )
            raw = resp.content[0].text.strip()
            if raw.lower() not in ("null", "none", "n/a", ""):
                if re.search(r'\d', raw) and len(raw) > 10:
                    return raw
            # Street address not found — try extracting just city from these same snippets
            resp_city = _call_claude(client,
                model="claude-sonnet-4-6",
                max_tokens=60,
                messages=[{
                    "role": "user",
                    "content": (
                        f"What city and state/country is the company {domain} based in?\n"
                        "Return only 'City, State/Country' (e.g. 'Reno, NV'). "
                        "If you cannot determine it, return exactly: null\n\n"
                        f"Search results:\n{snippets[:3000]}"
                    )
                }]
            )
            raw_city = resp_city.content[0].text.strip()
            if raw_city.lower() not in ("null", "none", "n/a", "") and "," in raw_city and len(raw_city) > 4:
                return raw_city
    except Exception:
        pass

    # Final fallback: ask for city/country if no full street address found
    # Useful for companies that list only "Belfast, UK" or "Reno, NV" without a street number
    try:
        resp = _call_claude(client,
            model="claude-sonnet-4-6",
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": (
                    "What city and country (or US state) is this company based in?\n"
                    "Return only 'City, State/Country' (e.g. 'Reno, NV' or 'Belfast, UK'). "
                    "If you cannot determine the city at all, return exactly: null\n\n"
                    f"Text:\n{current_text[:3000]}"
                )
            }]
        )
        raw = resp.content[0].text.strip()
        if raw.lower() not in ("null", "none", "n/a", "") and "," in raw and len(raw) > 4:
            return raw
    except Exception:
        pass

    return None


def search_restaurants(address: str) -> list:
    """
    Search DuckDuckGo for business dinner restaurants near the given address.
    Uses city+state only (not full street address) for better results.
    Returns a list of result dicts (title, body). Raises on failure.
    """
    from ddgs import DDGS

    # Extract city+state from the address — full street addresses confuse
    # DuckDuckGo and return unrelated pages. "Hamden, CT" works far better
    # than "2911 Dixwell Ave, Hamden, CT 06518".
    parts = [p.strip() for p in address.split(",")]
    if len(parts) >= 2:
        # Take last 2 parts (city + state/country), drop street and zip
        city_state = ", ".join(parts[-2:]).strip()
        # Strip trailing zip code if present (e.g. "CT 06518" → "CT")
        city_state = re.sub(r"\s+\d{5}(-\d{4})?$", "", city_state).strip()
    else:
        city_state = address

    query = f"fine dining restaurants {city_state} business dinner"
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=8))
    return results


def pick_top_restaurants(client: anthropic.Anthropic, search_results: list, address: str) -> list:
    """
    Ask Claude to select the 3 best business dinner restaurants from raw
    DuckDuckGo search snippets.

    Returns a list of up to 3 dicts: [{"name": ..., "description": ...}]
    Raises on Claude API failure; returns [] only if the response cannot be parsed.
    """
    if not search_results:
        return []

    snippets = "\n\n".join(
        f"Result {i + 1}:\nTitle: {r.get('title', '')}\n{r.get('body', r.get('snippet', ''))}"
        for i, r in enumerate(search_results)
    )

    resp = _call_claude(client,
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                f"Based on these search results for restaurants near {address}, "
                "pick up to 3 real dine-in restaurants suitable for a professional business dinner.\n\n"
                "IMPORTANT: If the search results do not contain actual restaurant listings "
                "(e.g. they are about unrelated products, homepages, or generic pages), "
                "return an empty array [] — do NOT invent placeholder names or explain why.\n\n"
                "Return a JSON array of up to 3 objects. Each object must have:\n"
                '  "name": the actual restaurant name\n'
                '  "description": one sentence on why it suits a business dinner\n\n'
                "No commentary, no markdown, no explanation — only the JSON array.\n\n"
                f"Search results:\n{snippets}"
            )
        }]
    )

    raw = resp.content[0].text.strip()

    _bad_name_signals = ("no suitable", "not found", "no restaurant", "cannot",
                         "unable", "invalid", "no result", "n/a")

    def _is_real_restaurant(r: dict) -> bool:
        name = r.get("name", "").lower().strip()
        if not name:
            return False
        return not any(sig in name for sig in _bad_name_signals)

    # Attempt 1: direct JSON parse (Claude returns a clean array)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            results = [
                {"name": r.get("name", ""), "description": r.get("description", "")}
                for r in parsed
                if isinstance(r, dict) and _is_real_restaurant(r)
            ][:3]
            if results:
                return results
            return []  # Claude returned placeholder objects — trigger fallback
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract a JSON array-of-objects from prose/markdown wrapper.
    match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                results = [
                    {"name": r.get("name", ""), "description": r.get("description", "")}
                    for r in parsed
                    if isinstance(r, dict) and _is_real_restaurant(r)
                ][:3]
                if results:
                    return results
                return []  # placeholder objects — trigger fallback
        except json.JSONDecodeError:
            pass

    return []


def ask_claude_for_restaurants(client: anthropic.Anthropic, address: str) -> list:
    """
    Ask Claude directly for restaurant recommendations based on its training knowledge.
    Used as a fallback when the DuckDuckGo web search pipeline returns no usable results.
    Returns a list of up to 3 dicts: [{"name": ..., "description": ...}]
    """
    resp = _call_claude(client,
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                f"Recommend 3 restaurants near this address for a professional business dinner: {address}\n\n"
                "Prefer well-known, established restaurants — fine dining, upscale gastropubs, or "
                "hotel restaurants with private dining. Only recommend places you are confident exist.\n\n"
                "Return a JSON array of exactly 3 objects. Each object must have:\n"
                '  "name": restaurant name\n'
                '  "description": one sentence on why it suits a business dinner\n\n'
                "No commentary, no markdown, no explanation — only the JSON array."
            )
        }]
    )
    raw = resp.content[0].text.strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [
                {"name": r.get("name", ""), "description": r.get("description", "")}
                for r in parsed
                if isinstance(r, dict) and r.get("name")
            ][:3]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [
                    {"name": r.get("name", ""), "description": r.get("description", "")}
                    for r in parsed
                    if isinstance(r, dict) and r.get("name")
                ][:3]
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# Helpers: group file handling
# ---------------------------------------------------------------------------

def read_group_files(folder: str) -> dict:
    """
    Return {filename: content} for every .md file in folder,
    excluding CLAUDE.md and README.md.
    """
    groups = {}
    for path in sorted(glob.glob(os.path.join(folder, "*.md"))):
        name = os.path.basename(path)
        if name.upper() in ("CLAUDE.MD", "README.MD"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            groups[name] = f.read()
    return groups


def extract_outreach_paragraph(content: str) -> str:
    """
    Pull the text block under '## Core Outreach Paragraph' from a group file.
    Falls back to the full file content if the header is not found.
    """
    marker = "## Core Outreach Paragraph"
    if marker in content:
        after = content.split(marker, 1)[1].strip()
        lines = []
        for line in after.splitlines():
            if line.startswith("##"):
                break
            lines.append(line)
        return "\n".join(lines).strip()
    return content.strip()


def resolve_matched_file(matched: str, groups: dict) -> str:
    """
    Attempt to resolve the filename Claude returned to an actual key in groups.
    Falls back to the first file if nothing matches.
    """
    if matched in groups:
        return matched
    matched_lower = matched.lower().replace('"', "").replace("'", "")
    for fname in groups:
        if matched_lower in fname.lower() or fname.lower().replace(".md", "") in matched_lower:
            return fname
    return next(iter(groups))


def display_name(filename: str) -> str:
    """Convert 'bulk-materials.md' → 'Bulk Materials'."""
    return filename.replace(".md", "").replace("-", " ").title()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python scout.py <company_url>")
        print("Example: python scout.py https://www.example.com")
        sys.exit(1)

    url = sys.argv[1]
    if not url.startswith("http"):
        url = "https://" + url

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with: set ANTHROPIC_API_KEY=your-key-here (Windows)")
        print("         or: export ANTHROPIC_API_KEY=your-key-here (Mac/Linux)")
        sys.exit(1)

    client = anthropic.Anthropic()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\nRunning scout on: {url}\n")

    # ------------------------------------------------------------------
    # Step 1: Scrape current site (homepage + sub-pages)
    # ------------------------------------------------------------------
    print("Step 1/7 — Scraping current website...")
    current_text, homepage_html = scrape_site(url)

    if not current_text:
        print("ERROR: Could not extract text from the site.")
        print("       Possible reasons: site requires login, blocks scrapers, or is JavaScript-only.")
        print("       Tip: make sure you are using the public marketing URL, not a portal or app URL.")
        sys.exit(1)

    print(f"         {len(current_text):,} characters extracted.")

    print("         Extracting current products and services...")
    current_products = extract_products(client, current_text, "current")
    if current_products:
        preview = ", ".join(current_products[:5])
        suffix = "..." if len(current_products) > 5 else ""
        print(f"         Found {len(current_products)} item(s): {preview}{suffix}")
    else:
        print("         No named products/services found — site may not list them prominently.")

    # Detect founding year — three attempts in priority order:
    #   1. Copyright year from raw homepage HTML (e.g. © 2012 in footer)
    #   2. Claude reading the visible page text for "founded in", "since", etc.
    #   3. Give up — fall back to wide Wayback window
    print("         Detecting company founding year...")
    founding_year = extract_copyright_year(homepage_html)
    if founding_year:
        print(f"         Founding year from copyright: {founding_year}")
    else:
        founding_year = detect_founding_year(client, current_text)
        if founding_year:
            print(f"         Founding year from page text: {founding_year}")
        else:
            print("         Founding year not found — will use wide Wayback window.")

    # Compute Wayback date range:
    #   Known year >= 2010 → first 36 months from founding
    #   Known year <  2010 → 2006–2010 (early history for older companies)
    #   Unknown            → 2006–2020 (wide window; we'll validate the snapshot below)
    if founding_year and founding_year >= 2010:
        wb_from  = f"{founding_year}0101"
        wb_to    = f"{founding_year + 2}1231"
        wb_label = f"{founding_year}–{founding_year + 2} (first 36 months)"
    elif founding_year:
        wb_from  = "20060101"
        wb_to    = "20101231"
        wb_label = "2006–2010"
    else:
        wb_from  = "20060101"
        wb_to    = "20201231"
        wb_label = "2006–2020 (wide — founding year unknown)"

    # ------------------------------------------------------------------
    # Step 2: Wayback Machine snapshot
    # ------------------------------------------------------------------
    print(f"\nStep 2/7 — Searching Wayback Machine for a {wb_label} snapshot...")
    candidates = get_wayback_candidates(url, from_date=wb_from, to_date=wb_to)

    old_products = []
    old_text = ""
    archive_url = None
    timestamp = None

    # Domain stem used to validate the snapshot belongs to THIS company
    # e.g. "evoleap" from "evoleap.com" — guards against a previous domain owner's content
    domain_stem = urlparse(url).netloc.replace("www.", "").split(".")[0].lower()

    if not candidates:
        print("         No snapshots found in that date range.")
    else:
        print(f"         Found {len(candidates)} candidate snapshot(s) — checking each for valid content...")
        valid_snapshots_checked = 0
        all_old_products: list = []

        for candidate_url, candidate_ts in candidates:
            year = candidate_ts[:4]
            old_html = fetch_page(candidate_url, timeout=25)
            old_text_candidate = html_to_text(old_html, max_chars=8000)

            if not old_text_candidate or len(old_text_candidate) < 300:
                print(f"         [skip {year}] Too little content.")
                continue
            if is_parked_page(old_text_candidate):
                print(f"         [skip {year}] Looks like a parked or placeholder page.")
                continue
            _text_lower = old_text_candidate.lower()
            _text_nospace = _text_lower.replace(" ", "").replace("-", "").replace("_", "")
            if domain_stem not in _text_lower and domain_stem not in _text_nospace:
                print(f"         [skip {year}] Company name '{domain_stem}' not found — likely a prior domain owner.")
                continue

            # Valid snapshot — use it (keep the first one as the canonical archive_url)
            if not archive_url:
                archive_url = candidate_url
                timestamp = candidate_ts
                old_text = old_text_candidate
            print(f"         Valid snapshot found from {year}: {candidate_url}")
            snapshot_products = extract_products(client, old_text_candidate, f"archived ({year})")
            if snapshot_products:
                all_old_products.extend(snapshot_products)
                preview = ", ".join(snapshot_products[:5])
                suffix = "..." if len(snapshot_products) > 5 else ""
                print(f"         Found {len(snapshot_products)} item(s) in {year} snapshot: {preview}{suffix}")
            valid_snapshots_checked += 1
            if valid_snapshots_checked >= 3:
                break  # cap at 3 homepage snapshots

        # Also probe archived interior product pages using CDX prefix discovery
        # This finds actual URLs like /products.php, /solutions/, /platform/index.html etc.
        interior_keywords = ["product", "solution", "service", "platform", "feature", "software"]
        interior_checked = 0
        parsed_base = urlparse(url)
        domain_only = parsed_base.netloc.replace("www.", "")
        for ikw in interior_keywords:
            if interior_checked >= 5:
                break
            try:
                disc_cdx = (
                    "http://web.archive.org/cdx/search/cdx"
                    f"?url={domain_only}/{ikw}*&matchType=prefix&output=json"
                    f"&from={wb_from}&to={wb_to}"
                    "&limit=3&filter=statuscode:200"
                    "&collapse=timestamp:4"
                    "&fl=timestamp,original"
                )
                r_disc = requests.get(disc_cdx, timeout=20)
                r_disc.raise_for_status()
                disc_data = r_disc.json()
                if len(disc_data) < 2:
                    continue
                for disc_row in disc_data[1:]:
                    ic_url = f"https://web.archive.org/web/{disc_row[0]}/{disc_row[1]}"
                    ic_ts = disc_row[0]
                    ic_html = fetch_page(ic_url, timeout=25)
                    ic_text = html_to_text(ic_html, max_chars=8000)
                    if not ic_text or len(ic_text) < 200:
                        continue
                    if is_parked_page(ic_text):
                        continue
                    print(f"         Interior page snapshot ({disc_row[1]}, {ic_ts[:4]}): {ic_url}")
                    ic_products = extract_products(client, ic_text, f"archived ({ic_ts[:4]}) interior")
                    if ic_products:
                        all_old_products.extend(ic_products)
                        preview = ", ".join(ic_products[:5])
                        suffix = "..." if len(ic_products) > 5 else ""
                        print(f"         Found {len(ic_products)} item(s) on /{ikw}*: {preview}{suffix}")
                    interior_checked += 1
                    break  # one snapshot per keyword is enough
            except Exception:
                continue

        # Deduplicate the combined pool (case-insensitive)
        seen = set()
        old_products = []
        for p in all_old_products:
            key = p.strip().lower()
            if key and key not in seen:
                seen.add(key)
                old_products.append(p.strip())

        if not archive_url:
            print("         No valid snapshot passed all checks.")
        else:
            print(f"         Total unique archived items across all snapshots: {len(old_products)}")

    # ------------------------------------------------------------------
    # Step 3: Compare old and current — find discontinued product
    # ------------------------------------------------------------------
    print("\nStep 3/7 — Identifying a discontinued product or service...")
    discontinued = None

    if old_products and current_products:
        discontinued = find_discontinued(client, old_products, current_products, period_label=wb_label)
        print(f"         Discontinued item: {discontinued}")
    elif not archive_url:
        print("         Skipped — no valid archived snapshot was available.")
    else:
        print("         Skipped — not enough product data from one or both versions.")

    # ------------------------------------------------------------------
    # Step 4: Read group files
    # ------------------------------------------------------------------
    print("\nStep 4/7 — Reading group files...")
    groups = read_group_files(script_dir)

    if not groups:
        print("ERROR: No group .md files found in the script directory.")
        sys.exit(1)

    print(f"         Loaded {len(groups)} files: {', '.join(groups.keys())}")

    # ------------------------------------------------------------------
    # Step 5: Match to best group
    # ------------------------------------------------------------------
    print("\nStep 5/7 — Matching company to best group...")
    raw_match = match_group(client, current_text, groups)

    # Handle NO_MATCH: output what we have and exit cleanly
    if raw_match.upper() == "NO_MATCH":
        print("         No group is a strong fit for this company.")
        print("\n" + "=" * 62)
        print("  SCOUT RESULTS")
        print("=" * 62)
        print(f"\n  Matched Group\n  {'─' * 50}")
        print("  No match — this company does not fit any current group.\n")
        print(f"  Discontinued Product\n  {'─' * 50}")
        print(f"  {discontinued if discontinued else 'None identified'}\n")
        print("=" * 62 + "\n")
        sys.exit(0)

    matched_file = resolve_matched_file(raw_match, groups)
    group_name = display_name(matched_file)
    print(f"         Best fit: {group_name}  ({matched_file})")

    # ------------------------------------------------------------------
    # Step 6: Personalize outreach paragraph
    # ------------------------------------------------------------------
    print("\nStep 6/7 — Personalizing outreach paragraph...")
    group_content = groups[matched_file]
    base_paragraph = extract_outreach_paragraph(group_content)
    personalized = personalize_paragraph(client, base_paragraph, url, current_text)
    print("         Done.")

    # ------------------------------------------------------------------
    # Step 7: Output results
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  SCOUT RESULTS")
    print("=" * 62)

    print(f"\n  Matched Group\n  {'─' * 50}")
    print(f"  {group_name}\n")

    print(f"  Discontinued Product\n  {'─' * 50}")
    print(f"  {discontinued if discontinued else 'None identified'}\n")

    print(f"  Outreach Paragraph\n  {'─' * 50}\n")
    for line in personalized.splitlines():
        print(f"  {line}")

    print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    main()
