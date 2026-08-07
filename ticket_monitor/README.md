# Odyssey 70mm IMAX ticket monitor

Checks the AMC Lincoln Square 13 theatre page for a bookable "Odyssey" 70mm
IMAX showtime on Fri Aug 7 or Sat Aug 8, 2026 after 1:30pm, and emails an
alert the moment one appears.

Run it yourself (locally, or via cron) -- it is not scheduled by anything in
this repo.

## Important caveat

AMC's site returned an HTTP 403 (bot-blocking) when this script was written,
so the exact page structure it parses against is **unverified**. The script
tries a structured path first (a `__NEXT_DATA__` JSON blob, if the page has
one) and falls back to scanning the visible text for the movie title near a
format keyword, a matching date, a time after your cutoff, and a booking
call-to-action. If it never alerts even once tickets are known to be on
sale, run it with `--debug path/to/dump.html` and open that file to see what
AMC actually served, then adjust `MOVIE_KEYWORDS`/`FORMAT_KEYWORDS` or the
parsing logic in `monitor.py` accordingly.

It's also more likely to get blocked (403) running from a server/VPN/cloud
IP than from a normal home or office connection -- run it from your own
machine.

## Setup

```bash
cd ticket_monitor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

All configuration is via environment variables:

| Variable | Required | Default | Description |
|---|---|---|---|
| `AMC_THEATRE_URL` | yes | - | Theatre or showtimes page to poll |
| `TARGET_DATES` | yes | - | Comma-separated `YYYY-MM-DD` dates to match, e.g. `2026-08-07,2026-08-08` |
| `MOVIE_KEYWORDS` | no | `odyssey` | Comma-separated title keywords to look for |
| `FORMAT_KEYWORDS` | no | `70mm,imax` | Comma-separated format keywords to look for |
| `MIN_TIME` | no | `13:30` | Earliest showtime (24h `HH:MM`) to alert on |
| `SMTP_HOST` | yes | - | e.g. `smtp.gmail.com` |
| `SMTP_PORT` | no | `587` | STARTTLS port |
| `SMTP_USER` | yes | - | SMTP login (also used as the From address by default) |
| `SMTP_PASSWORD` | yes | - | SMTP password / app password |
| `ALERT_EMAIL_TO` | yes | - | Where to send the alert |
| `ALERT_EMAIL_FROM` | no | `SMTP_USER` | Override the From address |

For Gmail, `SMTP_HOST=smtp.gmail.com`, `SMTP_PORT=587`, and `SMTP_PASSWORD`
must be a 16-character [app password](https://myaccount.google.com/apppasswords)
(regular account passwords won't work with SMTP).

Example `.env` (source it, or export the vars another way -- the script does
not read `.env` files itself):

```bash
export AMC_THEATRE_URL="https://www.amctheatres.com/movie-theatres/new-york-city/amc-lincoln-square-13"
export TARGET_DATES="2026-08-07,2026-08-08"
export MIN_TIME="13:30"
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="you@gmail.com"
export SMTP_PASSWORD="your-16-char-app-password"
export ALERT_EMAIL_TO="schakravarty@coactionspecialty.com"
```

## Running once

```bash
source .env   # or export the vars however you prefer
python3 monitor.py
```

Add `--debug dump.html` to also save the fetched page for inspection.

## Running every 5 minutes via cron

```
*/5 * * * * cd /path/to/ticket_monitor && . .env && .venv/bin/python3 monitor.py >> monitor.log 2>&1
```

The script runs once and exits each time, so a stuck or crashed run just
waits for the next tick rather than blocking future checks. It keeps a
small `.monitor_state.json` file (already checked in as a template-free,
empty state at first run) next to the script to avoid re-alerting for a
showtime it already told you about, and to avoid spamming you with repeated
"fetch failed" emails -- it sends that one only once per run of consecutive
failures.

Delete `.monitor_state.json` to reset and re-alert on anything currently
matching.
