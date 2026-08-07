#!/usr/bin/env python3
"""Check an AMC theatre page for a matching, bookable showtime and email an alert.

Meant to be invoked periodically (e.g. every 5 minutes via cron) rather than
run as a long-lived loop -- a single failed run then just waits for the next
cron tick instead of taking down a whole monitoring process.

All configuration is via environment variables; see README.md next to this
file for the full list and a cron example.

AMC's showtime data is normally rendered client-side and the theatre's own
bot protection returned an HTTP 403 when this script's author tried to
inspect the page's structure ahead of time, so the exact HTML/JSON layout
this script parses against is unverified. It tries a structured path first
(a Next.js "__NEXT_DATA__" JSON blob, if present) and falls back to a
keyword/date/time proximity heuristic over the visible page text. Use
--debug to dump the fetched HTML for manual inspection if it never matches
even when you know tickets are up.
"""

import argparse
import hashlib
import json
import os
import re
import smtplib
import sys
import time
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path

import requests
from bs4 import BeautifulSoup

STATE_FILE = Path(__file__).with_name(".monitor_state.json")
STATE_MAX_AGE_DAYS = 14
FAILURE_ALERT_THRESHOLD = 3
WINDOW_CHARS = 400

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

CTA_RE = re.compile(r"get tickets|buy tickets|select seats|reserve seats", re.I)
BLOCKED_RE = re.compile(r"sold out|no showtimes|not available|unavailable", re.I)
TIME_RE = re.compile(r"\b(1[0-2]|0?[1-9]):([0-5]\d)\s*([APap])\.?[Mm]\.?\b")


def env(name, default=None, required=False):
    value = os.environ.get(name, default)
    if required and not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


def load_config():
    return {
        "theatre_url": env("AMC_THEATRE_URL", required=True),
        "movie_keywords": [k.strip().lower() for k in env("MOVIE_KEYWORDS", "odyssey").split(",")],
        "format_keywords": [k.strip().lower() for k in env("FORMAT_KEYWORDS", "70mm,imax").split(",")],
        "target_dates": [d.strip() for d in env("TARGET_DATES", required=True).split(",")],
        "min_time": env("MIN_TIME", "13:30"),
        "smtp_host": env("SMTP_HOST", required=True),
        "smtp_port": int(env("SMTP_PORT", "587")),
        "smtp_user": env("SMTP_USER", required=True),
        "smtp_password": env("SMTP_PASSWORD", required=True),
        "alert_to": env("ALERT_EMAIL_TO", required=True),
        "alert_from": env("ALERT_EMAIL_FROM", env("SMTP_USER")),
    }


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"alerted": {}, "consecutive_failures": 0, "failure_alert_sent": False}


def save_state(state):
    cutoff = (datetime.now() - timedelta(days=STATE_MAX_AGE_DAYS)).isoformat()
    state["alerted"] = {k: v for k, v in state["alerted"].items() if v > cutoff}
    STATE_FILE.write_text(json.dumps(state, indent=2))


def date_labels(d):
    weekday_abbr, weekday_full = d.strftime("%a"), d.strftime("%A")
    month_abbr, month_full = d.strftime("%b"), d.strftime("%B")
    day, day_padded = str(d.day), f"{d.day:02d}"
    labels = {
        d.isoformat(),
        f"{weekday_abbr} {month_abbr} {day}",
        f"{weekday_abbr}, {month_abbr} {day}",
        f"{weekday_full}, {month_full} {day}",
        f"{month_abbr} {day}",
        f"{month_full} {day}",
        f"{month_abbr} {day_padded}",
    }
    return {label.lower() for label in labels}


def parse_target_dates(date_strs):
    parsed = []
    for s in date_strs:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        parsed.append((d, date_labels(d)))
    return parsed


def parse_min_time(s):
    return datetime.strptime(s, "%H:%M").time()


def parse_clock_time(hour, minute, ampm):
    hour = int(hour)
    minute = int(minute)
    if ampm.lower() == "p" and hour != 12:
        hour += 12
    if ampm.lower() == "a" and hour == 12:
        hour = 0
    return datetime.strptime(f"{hour:02d}:{minute:02d}", "%H:%M").time()


def extract_visible_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" | ")


def find_movie_windows(text, movie_keywords):
    lowered = text.lower()
    windows = []
    for keyword in movie_keywords:
        start = 0
        while True:
            idx = lowered.find(keyword, start)
            if idx == -1:
                break
            windows.append(text[max(0, idx - WINDOW_CHARS): idx + len(keyword) + WINDOW_CHARS])
            start = idx + len(keyword)
    return windows


def is_available_near(window, position):
    # Availability (Get Tickets vs Sold Out) is judged by whichever phrase is
    # textually closest to this specific time match, not by presence anywhere
    # in the whole card-spanning window -- otherwise one showtime's "Sold Out"
    # can veto an unrelated, actually-bookable showtime elsewhere in the window.
    cta_positions = [m.start() for m in CTA_RE.finditer(window)]
    if not cta_positions:
        return False
    nearest_cta = min(abs(p - position) for p in cta_positions)
    blocked_positions = [m.start() for m in BLOCKED_RE.finditer(window)]
    if not blocked_positions:
        return True
    nearest_blocked = min(abs(p - position) for p in blocked_positions)
    return nearest_cta < nearest_blocked


def evaluate_window(window, format_keywords, target_dates, min_time):
    lowered = window.lower()

    if not any(fmt in lowered for fmt in format_keywords):
        return None

    matched_date = None
    for d, labels in target_dates:
        if any(label in lowered for label in labels):
            matched_date = d
            break
    if matched_date is None:
        return None

    for match in TIME_RE.finditer(window):
        hour, minute, ampm = match.groups()
        clock = parse_clock_time(hour, minute, ampm)
        if clock < min_time:
            continue
        if not is_available_near(window, match.start()):
            continue
        return {
            "date": matched_date.isoformat(),
            "time": match.group(0),
            "snippet": " ".join(window.split())[:500],
        }
    return None


def parse_next_data(html):
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


SOLD_OUT_FIELD_RE = re.compile(r"sold[_ -]?out|unavailable", re.I)


def evaluate_json_node(node, format_keywords, target_dates, min_time):
    # The real AMC JSON schema is unknown (see module docstring), so this can't
    # check a specific "available" field by name -- only flag a plausible
    # match for the human to confirm, and skip it if a field name suggests
    # it's explicitly sold out.
    blob = " ".join(str(v) for v in node.values() if isinstance(v, (str, int, float)))
    lowered = blob.lower()

    if not any(fmt in lowered for fmt in format_keywords):
        return None

    for key, value in node.items():
        if SOLD_OUT_FIELD_RE.search(key) and value in (True, "true", "True"):
            return None

    matched_date = None
    for d, labels in target_dates:
        if any(label in lowered for label in labels):
            matched_date = d
            break
    if matched_date is None:
        return None

    for match in TIME_RE.finditer(blob):
        hour, minute, ampm = match.groups()
        clock = parse_clock_time(hour, minute, ampm)
        if clock >= min_time:
            return {
                "date": matched_date.isoformat(),
                "time": match.group(0),
                "snippet": f"(from page data, please verify manually) {' '.join(blob.split())[:400]}",
            }
    return None


def scan_next_data(node, movie_keywords, format_keywords, target_dates, min_time, found):
    if isinstance(node, dict):
        blob = " ".join(str(v) for v in node.values() if isinstance(v, (str, int, float))).lower()
        if any(k in blob for k in movie_keywords) and any(f in blob for f in format_keywords):
            evaluated = evaluate_json_node(node, format_keywords, target_dates, min_time)
            if evaluated:
                found.append(evaluated)
        for value in node.values():
            scan_next_data(value, movie_keywords, format_keywords, target_dates, min_time, found)
    elif isinstance(node, list):
        for item in node:
            scan_next_data(item, movie_keywords, format_keywords, target_dates, min_time, found)


def find_matches(html, config):
    target_dates = parse_target_dates(config["target_dates"])
    min_time = parse_min_time(config["min_time"])

    next_data = parse_next_data(html)
    if next_data is not None:
        found = []
        scan_next_data(next_data, config["movie_keywords"], config["format_keywords"], target_dates, min_time, found)
        if found:
            return found

    text = extract_visible_text(html)
    windows = find_movie_windows(text, config["movie_keywords"])
    matches = []
    for window in windows:
        evaluated = evaluate_window(window, config["format_keywords"], target_dates, min_time)
        if evaluated:
            matches.append(evaluated)
    return matches


def signature(match):
    return hashlib.sha256(f"{match['date']}|{match['time']}".encode()).hexdigest()


def send_email(config, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = config["alert_from"]
    msg["To"] = config["alert_to"]
    with smtplib.SMTP(config["smtp_host"], config["smtp_port"], timeout=30) as smtp:
        smtp.starttls()
        smtp.login(config["smtp_user"], config["smtp_password"])
        smtp.sendmail(config["alert_from"], [config["alert_to"]], msg.as_string())


def run(config, debug_dump):
    state = load_state()

    try:
        response = requests.get(config["theatre_url"], headers=BROWSER_HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        state["consecutive_failures"] += 1
        print(f"fetch failed ({exc})", file=sys.stderr)
        if state["consecutive_failures"] >= FAILURE_ALERT_THRESHOLD and not state["failure_alert_sent"]:
            send_email(
                config,
                "Ticket monitor: AMC page fetch is failing",
                f"The ticket monitor has failed to fetch {config['theatre_url']} "
                f"{state['consecutive_failures']} times in a row.\n\nLast error: {exc}\n\n"
                "This often means the site is blocking automated requests from this "
                "network/IP -- try running the script from a normal home/office "
                "connection rather than a server or VPN.",
            )
            state["failure_alert_sent"] = True
        save_state(state)
        return

    state["consecutive_failures"] = 0
    state["failure_alert_sent"] = False

    if debug_dump:
        debug_dump.write_text(response.text)
        print(f"wrote fetched HTML to {debug_dump}")

    matches = find_matches(response.text, config)
    new_matches = [m for m in matches if signature(m) not in state["alerted"]]

    if new_matches:
        lines = [f"- {m['date']} at {m['time']}: {m['snippet']}" for m in new_matches]
        send_email(
            config,
            "Odyssey 70mm IMAX tickets available at AMC Lincoln Square",
            "Found matching showtime(s):\n\n" + "\n\n".join(lines) + f"\n\n{config['theatre_url']}",
        )
        for m in new_matches:
            state["alerted"][signature(m)] = datetime.now().isoformat()
        print(f"alert sent for {len(new_matches)} new match(es)")
    else:
        print(f"no new matches ({len(matches)} total match(es), all previously alerted)" if matches else "no matches")

    save_state(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", type=Path, help="Write the fetched HTML to this path for inspection")
    args = parser.parse_args()

    config = load_config()
    run(config, args.debug)


if __name__ == "__main__":
    main()
