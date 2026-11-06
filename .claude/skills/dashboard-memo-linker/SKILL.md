---
name: dashboard-memo-linker
description: Link supporting documentation (investment memos, closing docs, pitch emails) into the Coaction Private Placement Pipeline dashboard's Funded Allocations table, at https://claude.ai/artifact/S9PebabLqHDyn36u938xBq (edit URL https://claude.ai/code/artifact/cba05fc6-0369-420c-bf8c-b6dc594a00e8). Use this whenever Sidd asks to "link the memo", "add the document", "pull up the backup", or "show detail" for a deal row in that dashboard's Coaction Allocations tab — and proactively whenever a new funded deal is added to that dashboard's `allocations` array, since every funded row should eventually carry a link + hover summary. Do not use this for other dashboards or artifacts; the link patterns here were tested specifically against this artifact's sandboxed iframe and may not generalize.
---

# Dashboard memo linker

This dashboard tracks Coaction's funded private placement / private credit allocations. Each row in the
`allocations` JS array (manager, deal, amount, structure, etc.) can carry a link to its supporting
documentation plus a hover summary. This was built and debugged end-to-end for one deal (AP Demeter
Holdings, Voya) — the steps and especially the dead ends below are what actually happened, not theory.

## Step 1 — find the source email

Search the deal's manager's mailbox for the pitch or closing-documents email that matches the deal name:

- **NEAM** deals → Kelly Sullivan, `Kelly.Sullivan@neamgroup.com`, subject usually "Private Placement Deal – [Issuer]"
- **NB** deals → Colleen Herron or Kyle O'Hara, `@nb.com`, subject usually "New PPD Deal: [Issuer]" or "Coaction Circle Note (NYMPP): [Issuer]"
- **Voya** deals → Tatyana Kaykova or other `@voya.com` contacts, subject usually "[Issuer] Closing Documents Notification"

Use `outlook_email_search` (query the deal name, filter by sender domain) then `read_resource` on the
message. If you're not sure which email is the right one, search a few variants of the deal name — these
mailboxes get plenty of unrelated traffic.

## Step 2 — pick a case

**Case A — there's a PDF that reads like a closing memo or term sheet.** Call `read_resource` on the
attachment URI. This connector returns **extracted/OCR'd text only — never the original PDF bytes**.
There is no way through this connector to re-upload a byte-identical copy of the original file. This is a
hard limitation of the tooling, not something to work around with a clever retry. State it plainly to the
user before building anything if they ask for "the original file" to be hosted somewhere — don't quietly
substitute a text version and call it the PDF.

**Case B — no formal memo, just a pitch or closing-notice email body.** Pull the key terms straight from
the email body text (size, spread, rating, tenor, use of proceeds — whatever's there). No file to link in
this case, just a summary.

Before investing time extracting and archiving a memo, confirm with the user whether they actually want a
durable re-hosted copy (SharePoint / dashboard asset) or would rather just link to the live email. On the
pilot deal we built both, then the user reversed course and asked to link to the original email instead —
re-hosting a lossy text extract wasn't worth it to them once they saw the tradeoff. Don't assume; ask, or
default to linking the live email (Step 4, pattern 1) since it's cheaper and loses nothing.

## Step 3 — (only if the user wants a durable copy) archive to SharePoint

Find or create a per-manager, per-year "Investment Memo" folder:
`Investments/Shared Documents/2. Reporting/{year}/2. Manager Updates/{Manager}/Investment Memo/`
(this convention already existed for 2025; create the 2026 — or current-year — equivalent via
`sharepoint_folder_search` + `sharepoint_create_folder` if it's missing).

Upload with `sharepoint_upload_file` as a `.txt` file (not `.pdf` — the API rejects plain text under a
`.pdf` extension, since it validates PDF extensions need real PDF bytes). Head the file clearly:

```
[DEAL NAME] — CLOSING MEMORANDUM (TEXT EXTRACT)
================================================
NOTE: This is a text extraction of the original PDF attachment ("...") sent by
[sender] ([date], subject "..."). Formatting/tables from the original PDF are
lost in this extraction — no connector was available to re-upload the original
PDF bytes byte-for-byte. For the original, pull the attachment from that email.
================================================
```

Never let the filename or the surrounding dashboard text imply this is the original PDF.

## Step 4 — link it into the dashboard

Two patterns actually work in this artifact. Two others were tried and failed — don't re-attempt them,
they're dead ends specific to this artifact's sandboxed iframe, not something a different approach to the
same idea will fix:

**Pattern 1 (preferred) — link the live email.** Use the email's own Graph API `webLink` field as a plain
`<a href="..." target="_blank">`. This works fine because it's an external site, not a same-origin
Artifact asset.

**Pattern 2 — in-page modal for a dashboard-hosted text asset.** If you did upload a text extract as an
Artifact asset (`Artifact` tool, `action: "publish"`, `asset: true` — requires the artifact to declare the
`assets` capability first, see the `artifact-capabilities` skill), do **not** link to it with a plain
`<a href target="_blank">` or with `window.open`. Instead:

```js
// Already in this page's <script>:
var memoModal = document.getElementById("memo-modal");       // display:none until opened
var memoModalTitle = document.getElementById("memo-modal-title");
var memoModalBody = document.getElementById("memo-modal-body"); // a <pre>
document.getElementById("memo-modal-close").addEventListener("click", closeMemoModal);
memoModal.addEventListener("click", function (e) { if (e.target === memoModal) closeMemoModal(); });
function closeMemoModal() { memoModal.style.display = "none"; }

window.openAssetBlob = function (url, ev, title) {
  if (ev && ev.preventDefault) ev.preventDefault();
  memoModalTitle.textContent = title || "Memo";
  memoModalBody.textContent = "Loading…";
  memoModal.style.display = "flex";
  fetch(url).then(function (r) {
    if (!r.ok) throw new Error("fetch failed: " + r.status);
    return r.text();
  }).then(function (text) { memoModalBody.textContent = text; })
    .catch(function (err) { memoModalBody.textContent = "Could not load this file (" + err.message + ")."; });
  return false;
};
```

This renders the fetched text inline in a modal already sitting in the page's DOM — nothing ever
navigates or opens a new window, so there's nothing for the browser to block.

**Dead ends — do not retry these:**
- `<a href="/_blob/<assetId>" target="_blank">` — opens a fresh top-level browser tab navigated directly
  at the asset path. Returns "not found." Assets don't resolve via a bare top-level navigation; they need
  to be fetched same-origin from inside the page's own script (as `<img src>` would, or as `fetch()` does).
- `fetch(url).then(blob => window.open(URL.createObjectURL(blob)))` — the `window.open` call happens
  inside an async callback, which most browsers' popup blockers treat as disconnected from the original
  click gesture and silently swallow.
- `var win = window.open("", "_blank")` synchronously at click time, then `win.location = ...` once the
  fetch resolves — still didn't open on this artifact. `window.open` appears to be blocked outright inside
  this artifact's sandboxed iframe regardless of timing, so there's no variant of "open a new
  tab/window" worth trying here — go straight to the in-page modal (Pattern 2).

## Step 5 — row data shape and icon convention

Add whichever of these fields apply to the row in the `allocations` array:

| Field | Meaning |
|---|---|
| `memoUrl` | the email's `webLink` (Pattern 1) — optional |
| `memoAssetUrl` | the uploaded text-extract asset's `/_blob/<id>` path (Pattern 2) — optional |
| `memoSummary` | plain-text hover summary, 3–5 sentences — used as the `title` on whichever link is present, or shown via a plain hover icon if neither link exists |

Icons used in the `issuer` table cell, in this order when more than one applies:

- ✉️ (`📧`) — opens the original email (`memoUrl`, Pattern 1)
- 📄 (`📄`) — opens the memo text extract in the in-page modal (`memoAssetUrl`, Pattern 2)
- ⓘ (`ⓘ`) — hover-only summary, no file at all (deal has `memoSummary` but neither link)

Every icon gets a `title` attribute carrying `memoSummary`. HTML-escape it with a small helper
(`escAttr`: escape `&`, `"`, `<`) since it's going into a double-quoted attribute. If the icon's `onclick`
needs to pass a JS string (e.g. a title for the modal), escape it separately for JS single-quote context
(`jsStr`: wrap in `'...'`, escape `\` and `'`) — don't reuse `escAttr` for that, the two escaping contexts
are different and mixing them breaks either the HTML or the JS.

## Step 6 — summary content

Write `memoSummary` from the deal's actual numbers, not boilerplate — pull the real structure, sponsor,
rating, spread, tenor, and anything genuinely distinctive about the deal (e.g. "19 stabilized Sun Belt
multifamily properties" beats "a multifamily deal"). If the source memo gives you a more precise number
than what's already in the row (exact funding date, real spread-to-benchmark instead of an earlier
rate-implied estimate, an actual rating instead of a blank), update the row's own fields too — the memo is
often a better source than whatever got the row into the dashboard in the first place.

## Step 7 — validate before publishing

Every publish of this dashboard needs:
1. Extract the `<script>...</script>` contents to a temp file and run `node --check` on it.
2. Count open vs. close tags for every block-level tag touched (`div`, `table`, `thead`, `tbody`, `tr`,
   `td`, `th`, `span`, `select`, `input`, `label`, `button`, `pre`) across the whole file.
3. The file already has a few pre-existing **self-closing** SVG shapes (`<circle .../>`, `<line .../>`)
   elsewhere in the page (the spread-trend chart). A naive open/close tag counter will flag these as
   "imbalanced" — they aren't; self-closing tags don't need a separate closing tag. Don't chase this as a
   bug.
4. Only after both checks pass, publish with the `Artifact` tool to the same `file_path`, same `url`
   (`https://claude.ai/code/artifact/cba05fc6-0369-420c-bf8c-b6dc594a00e8`) so it updates in place rather
   than creating a new artifact.

## Rolling this out to more deals

This was proven on one deal. When doing more: batch the email searches (one background `Agent` per
manager mailbox is efficient, since Kelly/Colleen-Kyle/Voya are three independent searches), but do the
SharePoint upload / asset upload / row-patching / template edit / validate / publish steps yourself in the
main session — those touch the one shared dashboard file sequentially, so parallelizing them risks two
agents racing to patch the same `allocations` array or publish over each other.
