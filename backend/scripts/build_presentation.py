#!/usr/bin/env python3
"""Generate the PS-14 fraud-detection presentation deck (.pptx).

Run:  python scripts/build_presentation.py  ->  docs/PS-14_fraud_detection.pptx

16 slides, 16:9, dark theme matching the front page. Speaker notes are
included per slide (visible in PowerPoint's Notes pane).
"""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
OUT = ROOT / "docs" / "PS-14_fraud_detection.pptx"

# Theme (matches the front page)
BG = RGBColor(0x0D, 0x11, 0x17)
PANEL = RGBColor(0x16, 0x1B, 0x22)
ACCENT = RGBColor(0x58, 0xA6, 0xFF)
TEXT = RGBColor(0xE6, 0xED, 0xF3)
MUTED = RGBColor(0x8B, 0x94, 0x9E)
FONT = "Calibri"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def new_slide() -> object:
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def box(slide, left, top, width, height) -> object:
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def style(p, size=18, color=TEXT, bold=False, space_after=10, italic=False):
    p.font.name = FONT
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.italic = italic
    p.space_after = Pt(space_after)


def title_bar(slide, kicker, title):
    k = box(slide, 0.7, 0.45, 12, 0.4)
    style(k.paragraphs[0], size=12, color=ACCENT, bold=True)
    k.paragraphs[0].text = kicker.upper()
    t = box(slide, 0.7, 0.85, 12, 0.9)
    style(t.paragraphs[0], size=34, color=TEXT, bold=True)
    t.paragraphs[0].text = title


def bullets(slide, items, top=1.95, left=0.7, width=12, size=18):
    tf = box(slide, left, top, width, 4.8)
    for i, item in enumerate(items):
        level, text = (item if isinstance(item, tuple) else (0, item))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("•  " if level == 0 else "–  ") + text
        style(p, size=size - 3 * level, color=TEXT if level == 0 else MUTED, space_after=12)
    return tf


def numbered(slide, steps, top=1.95):
    tf = box(slide, 0.7, top, 12, 4.8)
    for i, (head, desc) in enumerate(steps):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{i + 1}.  {head}"
        style(p, size=18, color=TEXT, bold=True, space_after=2)
        p2 = tf.add_paragraph()
        p2.text = "      " + desc
        style(p2, size=15, color=MUTED, space_after=12)


def table_slide(slide, headers, rows, note=None):
    tf = box(slide, 0.7, 1.9, 12, 0.5)
    style(tf.paragraphs[0], size=13, color=MUTED, italic=True)
    tf.paragraphs[0].text = "Each store is isolated — a breach of one doesn't expose the rest." if note is None else note
    shp = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(0.7), Inches(2.5), Inches(11.9), Inches(3.6))
    tbl = shp.table
    for c, h in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PANEL
        for p in cell.text_frame.paragraphs:
            style(p, size=14, color=ACCENT, bold=True)
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if r % 2 else BG
            for p in cell.text_frame.paragraphs:
                style(p, size=14, color=TEXT if c == 0 else MUTED, space_after=0)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---------------------------------------------------------------- Slide 1 — title
s = new_slide()
t = box(s, 1.2, 2.1, 10.9, 1.4)
style(t.paragraphs[0], size=52, color=TEXT, bold=True)
t.paragraphs[0].text = "PS-14"
p = t.add_paragraph()
p.text = "Privacy-First AI Fraud Detection"
style(p, size=30, color=ACCENT, bold=True)
sub = box(s, 1.2, 3.9, 10.9, 1.0)
style(sub.paragraphs[0], size=20, color=TEXT)
sub.paragraphs[0].text = "Detect fraud without unnecessarily knowing who the person is."
p2 = sub.add_paragraph()
p2.text = "AI recommends — verification decides."
style(p2, size=20, color=TEXT)
ftr = box(s, 1.2, 6.2, 10.9, 0.5)
style(ftr.paragraphs[0], size=13, color=MUTED)
ftr.paragraphs[0].text = "Built against the DPDP Act 2023 · RBI fraud-risk guidelines · GDPR-equivalent controls"
notes(s, "Open with the tagline: privacy is a design principle, not an afterthought.")

# ---------------------------------------------------------------- Slide 2 — problem
s = new_slide()
title_bar(s, "The problem", "Fraud is expensive. So is over-collecting data.")
bullets(s, [
    "Card fraud costs billions every year — and attackers keep innovating.",
    "Traditional systems demand large amounts of personal data to score risk.",
    "That data is a privacy risk AND a compliance headache (DPDP Act, GDPR, RBI guidelines).",
    "The question: can we catch fraudsters without spying on everyone?",
])
notes(s, "Frame the tension: security vs privacy. The system is the answer to that tension.")

# ---------------------------------------------------------------- Slide 3 — core idea
s = new_slide()
title_bar(s, "The core idea", "Privacy by design, not privacy as a patch.")
bullets(s, [
    "Data minimization — collect only what is needed to score risk.",
    "Pseudonyms — every person becomes a random ID; nobody knows who is who.",
    "AI recommends, humans verify — the machine flags, the customer confirms.",
    "Privacy by design — a breach leaks almost nothing sensitive.",
])
notes(s, "The pseudonym is the load-bearing wall: all downstream scoring works on fake IDs.")

# ---------------------------------------------------------------- Slide 4 — pipeline
s = new_slide()
title_bar(s, "One glance", "The pipeline: event in, decision out, learning back.")
numbered(s, [
    ("Transaction happens", "A payment, login, or transfer arrives from any channel."),
    ("Anonymize", "The Privacy Layer turns it into features — amount ratio, unusual time, new device."),
    ("Score", "The Risk Engine fuses four models + rules into a 0–100 risk score."),
    ("Decide", "Allow / Step-up / Verify by band."),
    ("Verify", "The customer confirms: \u201cThis was me / wasn't me.\u201d"),
    ("Learn", "Confirmation feeds back to retrain the AI."),
])
notes(s, "Walk the loop top to bottom, then close it: the last step feeds the first.")

# ---------------------------------------------------------------- Slide 5 — services table
s = new_slide()
title_bar(s, "Architecture", "Five services, five databases — no single point of truth.")
table_slide(s,
    ["Service", "Job", "Store"],
    [
        ["Identity Service", "Who you are · issues the JWT", "DB-1 · PII (encrypted)"],
        ["Privacy Layer", "Anonymize events → features", "DB-2 · features only"],
        ["Risk Engine", "Score 0–100 · rules · limits", "DB-3 · risk scores"],
        ["Verification", "Alerts · \u201cwas it you?\u201d · feedback", "DB-3 · outcomes"],
        ["Audit Service", "Tamper-evident record", "DB-4 · hash chain"],
    ])
notes(s, "Emphasize physical separation: even a full breach of one store exposes almost nothing.")

# ---------------------------------------------------------------- Slide 6 — privacy layer
s = new_slide()
title_bar(s, "The privacy layer", "Turning events into features — and discarding the rest.")
bullets(s, [
    "Raw data never leaves the layer: amounts, exact geo, and device IDs are not stored.",
    "Instead we keep: \u201camount is 2.3× your usual\u201d, \u201cat 3 AM\u201d, \u201cnew device\u201d, \u201cmany failed logins\u201d.",
    "Even the Risk Engine never sees who you are — only the pseudonym and its features.",
    "Extra shields: k-anonymity gate on exports, no raw amounts in features.",
])
notes(s, "The killer detail: the scoring models literally cannot see identity or raw amounts.")

# ---------------------------------------------------------------- Slide 7 — the AI
s = new_slide()
title_bar(s, "The AI", "Four models, one calibrated probability.")
bullets(s, [
    "Logistic Regression — classic, explainable baseline.",
    "Random Forest / XGBoost — learn complex fraud patterns from history.",
    "Isolation Forest — catches never-seen anomalies (zero-shot fraud).",
    "A stacker fuses all four into a single calibrated probability.",
    "Output is an honest probability — not a confident guess.",
])
notes(s, "Why four models: diversity. What one memorizes, another generalizes.")

# ---------------------------------------------------------------- Slide 8 — rules engine
s = new_slide()
title_bar(s, "The rules engine", "Human expertise, written down and enforced.")
bullets(s, [
    "Declarative rules: \u201camount > 1.3× usual → flag\u201d, \u201cnew device + high amount → verify\u201d.",
    "Every flag maps to a category-level reason code the customer can understand.",
    "Velocity limits run first: >10 txns/day or >3× daily spend → step up or block.",
    "Rules + ML are fused into the final 0–100 risk score.",
])
notes(s, "Rules are the explainability layer: the customer always gets a reason, never a threshold.")

# ---------------------------------------------------------------- Slide 9 — decision bands
s = new_slide()
title_bar(s, "The decision", "Three bands, no black boxes.")
numbered(s, [
    ("0–30 → Allow", "Low risk. No friction for the customer."),
    ("31–70 → Step up", "Extra check — e.g., an OTP."),
    ("71–100 → Verify", "Human or customer confirmation required."),
])
bullets(s, ["Every decision ships with reason codes — explainable by design, not retrofitted."], top=4.6, size=16)
notes(s, "The bands map to business action: friction is proportional to risk.")

# ---------------------------------------------------------------- Slide 10 — human loop
s = new_slide()
title_bar(s, "The human loop", "This is the secret sauce.")
bullets(s, [
    "High-risk alert asks the customer: \u201cThis was me / This wasn't me.\u201d",
    "Confirmed = legitimate → the model learns not to flag that pattern again.",
    "Disputed = fraud → becomes labeled training data.",
    "The feedback retrains the model — it gets smarter with every real case.",
    "This is the difference between a static rule list and a learning system.",
])
notes(s, "The loop is the moat: every customer click is a labeled training example.")

# ---------------------------------------------------------------- Slide 11 — audit trail
s = new_slide()
title_bar(s, "Trust", "The tamper-evident audit trail.")
bullets(s, [
    "Every important action lands on an append-only hash chain.",
    "Each record carries the fingerprint of the one before it — change one, everything after breaks.",
    "A compliance viewer (separate passphrase) verifies the whole chain anytime.",
    "Regulators can verify a signed export independently — no trust required.",
])
notes(s, "Like a blockchain, but private: integrity without broadcasting the data.")

# ---------------------------------------------------------------- Slide 12 — privacy deep-dive
s = new_slide()
title_bar(s, "Privacy deep-dive", "The extra shields.")
bullets(s, [
    "Federated learning — institutions train locally, share only model weights, never data.",
    "Differential privacy — calibrated noise on shared weights with provable guarantees.",
    "k-anonymity gate — exports are blocked if anyone could be uniquely identified.",
    "Break-glass — the only pseudonym→person path, and it is always logged.",
])
notes(s, "These are the answers to \u201chow do you share data safely\u201d and \u201cwho can deanonymize\u201d.")

# ---------------------------------------------------------------- Slide 13 — live demo
s = new_slide()
title_bar(s, "Live demo", "Two minutes, no code.")
numbered(s, [
    ("Register & login", "A 15-minute session token, carrying only your pseudonym."),
    ("Seed a demo account", "One click → 7 events, 1 high-risk alert with reason chips."),
    ("Verify", "\u201cThis wasn't me\u201d → the case lands in history and the audit chain."),
    ("Compliance view", "The hash-verified trail, end to end."),
])
notes(s, "Do the demo for real: seed, click wasn't-me, then show the chain update live.")

# ---------------------------------------------------------------- Slide 14 — results
s = new_slide()
title_bar(s, "Results", "Honest numbers, enforced quality.")
bullets(s, [
    "The five live scenarios return exactly the documented decisions: 29 / 14 / 86 / 70 / 27.",
    "Honest generalization — leave-one-archetype-out recall, not inflated splits.",
    "Retrains are gated: held-out fraud recall below a floor fails the build.",
    "Velocity limits cut the false-positive challenge rate while fraud recall holds.",
])
notes(s, "Call out the honesty: we publish the OOD numbers alongside the optimistic ones.")

# ---------------------------------------------------------------- Slide 15 — limitations / next
s = new_slide()
title_bar(s, "Limitations & next steps", "What's real, what's next.")
bullets(s, [
    "Synthetic data — the models need a real-world pilot to prove out.",
    "Prototype shared tokens → production uses mTLS, KMS, and per-store credentials.",
    "More fraud archetypes, more institutions in the federated ring.",
    "Pilot plan: one institution, shadow-mode scoring, then live.",
])
notes(s, "Be upfront: this is a compliance-grade prototype, not yet a production service.")

# ---------------------------------------------------------------- Slide 16 — takeaway
s = new_slide()
t = box(s, 1.2, 2.6, 10.9, 1.6)
style(t.paragraphs[0], size=36, color=TEXT, bold=True)
t.paragraphs[0].text = "Privacy isn't the enemy of fraud detection —"
p = t.add_paragraph()
p.text = "it's the design."
style(p, size=36, color=ACCENT, bold=True)
sub = box(s, 1.2, 4.6, 10.9, 0.8)
style(sub.paragraphs[0], size=18, color=MUTED)
sub.paragraphs[0].text = "Less data collected → less to leak → more customer trust → cleaner, better AI."
notes(s, "Close on the paradox: less data makes the system MORE trustworthy and easier to operate.")

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"wrote {OUT} ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    sys.exit(0)
