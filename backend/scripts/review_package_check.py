#!/usr/bin/env python3
"""Independent-evidence review-package checker.

Companion to ``docs/evaluation/INDEPENDENT_EVIDENCE_REVIEW_PACKAGE.md`` — the
pre-review packet that collects the dataset inventory, the licence worksheet,
the A-O audit checklist and the freeze-readiness matrix for the eventual
statistical/domain review and independent dataset/provenance audit.

This is a **document-consistency** checker, in the same class as
``review_resolution_check.py`` and ``claim_evidence_check.py``: it imports no
model, dataset, metric or evaluation path, runs no experiment, and writes
nothing. Its job is to fail if the packet ever starts asserting something the
repository has not established.

Rules enforced (exit 0 only when all hold):

  T1  sections 1..13 exist, exactly once each, in order
  T2  every required dataset is represented in the inventory
  T3  no licence outcome in the worksheet is anything other than
      NOT ESTABLISHED or NOT ACQUIRED / REVIEW BLOCKED, and the licence
      statement row of the acquired-dataset table is never a positive claim
  T4  no reviewer is marked assigned or approved, and no ruling is APPROVED
  T5  the 50M dataset is never claimed as acquired/generated/built
  T6  the six permitted audit outcomes are present and every A-O item is still
      at REQUIRES REVIEW (nothing pre-certified)
  T7  the observed-items block O-1..O-8 is intact
  T8  the eleven IBM-specific verification items are present
  T9  the packet declares its own boundaries and carries the draft status line
  R1  docs/FREEZE_RECORD.json is still absent
  R2  plan / metric-definition hashes are unchanged
  R3  production manifest hash, threshold and feature count are unchanged
  R4  native 48-feature source hashes are unchanged
  R5  acquired dataset byte sizes match the measured values
  R6  no dataset was acquired (data/external*, data/external_benchmark* absent)
  R7  the decision artifact is byte-unchanged (no approval fabricated there)
  A1  both reviewer role blocks exist in REVIEWER_ASSIGNMENT_RECORD.md
  A2  every required reviewer-qualification field is present with a value
  A3  a named assignment carries affiliation, date, conflict declaration and
      approval authority (never NOT ESTABLISHED)
  A4  a named assignment carries role-specific expertise and a §3 Identity entry
  A5  one person cannot be named for both reviewer roles
  A6  `Assignment status` agrees with `Assigned individual` (no fabricated state)
  A7  no approval appears while no reviewer identity exists
  A8  no non-eligible identity (AI/assistant/project-author markers, or an email)
  A9  the external-dependency statement is present
  A10 the engagement brief states its true status, its four roles, all fourteen
      decisions, the N1-N13 negative findings, and never claims review completion

Text rules run on a whitespace-normalised copy of the packet so that a
line-wrapped phrase cannot hide a claim. Negated boundary forms ("No independent
audit has occurred", "does not claim") are deliberately not matched: the packet
must be able to say what has *not* happened.

NOT registered in the repository battery: ``review_resolution_check.py`` already
covers anti-fabrication for /01../14, and this packet is a draft navigation
artifact rather than a frozen one. Registering it would duplicate that coverage
and change the battery's meaning.

Usage:
    python backend/scripts/review_package_check.py             # enforce
    python backend/scripts/review_package_check.py --self-test # fixtures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = ROOT / "docs" / "evaluation" / "INDEPENDENT_EVIDENCE_REVIEW_PACKAGE.md"

SECTIONS = [
    "## 1. Purpose and review boundary",
    "## 2. Current project state",
    "## 3. Authoritative dataset inventory",
    "## 4. IBM v2 provenance evidence",
    "## 5. Dataset exposure / independence boundary",
    "## 6. Licence / provenance review worksheet",
    "## 7. Independent audit checklist",
    "## 8. IBM-specific audit items",
    "## 9. Exposure / independence audit",
    "## 10. Reviewer input contract",
    "## 11. Freeze readiness matrix",
    "## 12. 50M status",
    "## 13. External / Independent Dependencies",
]

REQUIRED_DATASETS = [
    "credit_card_transactions-ibm_v2.csv",
    "creditcard.csv",
    "fraudTrain.csv",
    "fraudTest.csv",
    "User0_credit_card_transactions.csv",
    "ealtman2019",
    "PS-14 synthetic",
    "IEEE-CIS",
    "BAF",
    "Zenodo",
    "FreeFraudDetection50M",
    "Dal Pozzolo",
    "Worldline",
    "Novatti",
    "Elliptic",
]

# The only two values a dataset's licence decision line may carry in this packet.
LICENCE_OK = {"NOT ESTABLISHED", "NOT ACQUIRED / REVIEW BLOCKED", "N/A"}
# Tokens that would turn a licence cell into an unsupported positive claim.
LICENCE_FORBIDDEN = ("cleared", "approved", "verified", "open", "permitted",
                     "f-p", "f-r")

AUDIT_OUTCOMES = [
    "ESTABLISHED",
    "NOT ESTABLISHED",
    "FAILED",
    "BLOCKED",
    "INCONCLUSIVE",
    "REQUIRES REVIEW",
]

IBM_ITEMS = [
    "24,386,900 rows",
    "15 actual raw columns",
    "No `hour_diff` field",
    "No 168-column file is being used",
    "six previously incorrect manifest claims were corrected",
    "is a duplicate extract",
    "is a duplicate corpus",
    "not counted as independent evidence",
    "consistent with `models/production/manifest.json`",
    "authoritative feature contract",
    "24,386,899 remains preserved",
]

PHASES_REQUIRED = [
    "not an independent audit",
    "not independent certification",
    "REVIEW PACKAGE READY / PASS WITH LIMITATIONS",
    "no reviewer identity is invented",
    "unchecked for every dataset",
]

# Unambiguous positive-assertion forms. Negated forms ("No ... has occurred",
# "does not claim", "Has any 50M dataset been generated? | No") are not matched.
FALSE_AUDIT_STRINGS = [
    "independent audit has occurred: ",
    "this packet is the audit",
    "independent audit passed",
    "independent audit is complete",
    "independent audit completed",
]
BAD_50M_STRINGS = [
    "50M dataset was generated",
    "50M dataset has been generated",
    "50M corpus is complete",
]

DECISION_RE = re.compile(r"Reviewer decision:\s*\**`?(?P<v>[^`\n*|]*)")
LICENCE_ROW_RE = re.compile(r"^\|\s*Licence statement\s*\|(?P<cells>.*)\|\s*$", re.M)
BAD_APPROVAL_RES = [
    re.compile(r"(?i)Reviewer decision:\s*\**`?APPROVED"),
    re.compile(r"(?i)^\s*Approval:\s*APPROVED", re.M),
    re.compile(r"(?i)\b(?:statistical|domain) reviewer\b[^|\n]{0,20}\|\s*\**`?APPROVED"),
    re.compile(r"(?i)\b(?:reviewer decision|approval)\s*\|\s*`?APPROVED"),
    re.compile(r"(?i)\breviewers? (?:are|is) (?:assigned|approved)\b"),
]
BAD_50M_RE = re.compile(
    r"(?i)50[,.]?0{6}[^.\n]{0,60}\b(?:acquired|generated|built|completed)\b",
)


# ------------------------------------------------------------------ text rules

def normalize(text: str) -> str:
    """Collapse whitespace and blockquote markers so a wrapped phrase cannot
    hide a claim across a line break or a ``>`` prefix."""
    return " ".join(text.replace(">", " ").split())


def _licence_values(norm: str) -> list[str]:
    return [" ".join(m.group("v").split()).strip()
            for m in DECISION_RE.finditer(norm)]


def check_text(raw: str) -> list[str]:
    p: list[str] = []
    norm = normalize(raw)

    # T1 — required sections, once each, in order
    pos = -1
    for sec in SECTIONS:
        n = raw.count(sec)
        if n != 1:
            p.append(f"T1 section not present exactly once ({n}): {sec!r}")
        here = raw.find(sec)
        if here <= pos:
            p.append(f"T1 section out of order: {sec!r}")
        pos = max(pos, here)

    # T2 — required datasets
    for name in REQUIRED_DATASETS:
        if name not in norm:
            p.append(f"T2 required dataset missing from the packet: {name!r}")

    # T3 — licence worksheet: decision lines and statement cells
    vals = _licence_values(norm)
    if len(vals) < 13:
        p.append(f"T3 licence worksheet carries only {len(vals)} decision line(s)")
    for v in vals:
        if v not in LICENCE_OK:
            p.append(f"T3 licence worksheet value not permitted: {v!r}")
    m = LICENCE_ROW_RE.search(raw)
    if not m:
        p.append("T3 'Licence statement' row is missing from the worksheet table")
    else:
        for cell in m.group("cells").split("|"):
            c = cell.strip().strip("`").strip()
            if not c:
                continue
            low = c.lower()
            if low.startswith("n/a") or low.startswith("not established"):
                if any(t in low for t in LICENCE_FORBIDDEN):
                    p.append(f"T3 licence statement cell carries a positive "
                             f"claim: {c!r}")
                continue
            p.append(f"T3 licence statement cell carries a positive claim: {c!r}")
    if "unchecked for every dataset" not in norm:
        p.append("T3 the packet does not restate that plan §35 is unchecked "
                 "for every dataset")

    # T4 — reviewer assignment / approvals
    for name in ("Statistical reviewer", "Domain reviewer"):
        if f"{name} | **`NOT ASSIGNED`**" not in norm:
            p.append(f"T4 {name} is not recorded as NOT ASSIGNED")
    for rx in BAD_APPROVAL_RES:
        m = rx.search(raw)
        if m:
            p.append(f"T4 approval/assignment asserted: {m.group(0)!r}")

    # T5 — 50M
    if "50M dataset generation = NOT AUTHORIZED / NOT YET SCIENTIFICALLY JUSTIFIED." not in norm:
        p.append("T5 the 50M non-authorization statement is missing")
    m = BAD_50M_RE.search(norm)
    if m:
        p.append(f"T5 a 50M dataset is claimed as acquired/generated: {m.group(0)!r}")
    for s in BAD_50M_STRINGS:
        if s in norm:
            p.append(f"T5 a 50M dataset is claimed as acquired/generated: {s!r}")

    # T6 — audit outcomes and REQUIRES REVIEW
    for w in AUDIT_OUTCOMES:
        if w not in norm:
            p.append(f"T6 permitted audit outcome missing from the vocabulary: {w}")
    n_req = norm.count("Current packet state: `REQUIRES REVIEW`")
    if n_req < 15:
        p.append(f"T6 only {n_req} of 15 audit items are at REQUIRES REVIEW")

    # T7 — observed items
    for i in range(1, 9):
        if f"**O-{i}**" not in norm:
            p.append(f"T7 observed item O-{i} is missing")

    # T8 — IBM items
    for item in IBM_ITEMS:
        if item not in norm:
            p.append(f"T8 IBM verification item missing: {item!r}")

    # T9 — declared boundaries and status
    low = norm.lower()
    for phrase in PHASES_REQUIRED:
        if phrase.lower() not in low:
            p.append(f"T9 packet boundary/status phrase missing: {phrase!r}")
    for s in FALSE_AUDIT_STRINGS:
        if s in low:
            p.append(f"T9 false audit claim: {s!r}")

    return p


# ------------------------------------------------------------------ state rules

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_repo_state(root: Path = ROOT) -> list[str]:
    p: list[str] = []

    # R1 — still no freeze record
    if (root / "docs" / "FREEZE_RECORD.json").exists():
        p.append("R1 docs/FREEZE_RECORD.json exists — the freeze state changed")

    # R2 — plan and metric definitions unchanged
    for rel, want in (
        ("docs/RESEARCH_PLAN.md",
         "eab5a0615801db7e50ab4d1c0ec73f93905e461b5bd8eacc617e0418d656b9ce"),
        ("docs/metric_definitions.md",
         "8665549b3d62fef36402d737910cd8aa1acfbde025dfee4aa1395a9e3d99ed07"),
    ):
        f = root / rel
        if not f.exists():
            p.append(f"R2 missing: {rel}")
        elif sha256(f) != want:
            p.append(f"R2 hash changed: {rel}")

    # R3 — production manifest, threshold, feature count, training pin
    man = root / "models" / "production" / "manifest.json"
    if not man.exists():
        p.append("R3 missing: models/production/manifest.json")
    else:
        if sha256(man) != "e6da1a433ec618576aa12208190d35df3b53e9349b1bb6e54c16465e324b2485":
            p.append("R3 models/production/manifest.json changed")
        d = json.loads(man.read_text(encoding="utf-8"))
        if d.get("locked_threshold") != 0.7847116291110687:
            p.append("R3 locked_threshold changed")
        if d.get("n_features") != 48:
            p.append("R3 n_features changed")
        if d.get("training_dataset_sha256") != (
            "b01fa323c98522f8c710c7f7242581860c97c50183f1c5fa9e772e5c674a7f15"
        ):
            p.append("R3 training dataset pin changed")

    # R4 — native 48-feature sources
    for rel, want in (
        ("backend/src/privacy_layer/native_features.py",
         "5760a20376306598312cfaf32d8d12b5e5f3ff268fcc0b4de4baff64e90f57d1"),
        ("backend/src/risk_engine/altman_native_ensemble.py",
         "55f4ad5284ec6f45481660dfe2f9b256e90ad6ee2ff74abf68e876b8ebec2d24"),
    ):
        f = root / rel
        if not f.exists():
            p.append(f"R4 missing: {rel}")
        elif sha256(f) != want:
            p.append(f"R4 native feature source changed: {rel}")

    # R5 — measured dataset sizes
    for rel, want in (
        ("data/creditcard.csv", 150_828_752),
        ("data/kaggle_fraud/fraudTrain.csv", 351_238_196),
        ("data/kaggle_fraud/fraudTest.csv", 150_354_339),
        ("data/credit_card_transactions-ibm_v2.csv", 2_350_744_057),
        ("data/User0_credit_card_transactions.csv", 1_899_258),
    ):
        f = root / rel
        if not f.exists():
            p.append(f"R5 missing dataset file: {rel}")
        elif f.stat().st_size != want:
            p.append(f"R5 dataset size changed: {rel} "
                     f"({f.stat().st_size} != {want})")

    # R6 — nothing newly acquired
    for d in ("data/external", "data/external_benchmark"):
        if (root / d).exists():
            p.append(f"R6 dataset directory appeared: {d}")

    # R7 — decision artifact unchanged (no approval fabricated)
    res = root / "docs" / "evaluation" / "STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md"
    if not res.exists():
        p.append("R7 missing: STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md")
    elif sha256(res) != (
        "17a95e2afc18b1dd2b189aa307f90b79ed1e03701e51f37727e2754ca2d42d24"
    ):
        p.append("R7 the review-resolution artifact changed — re-verify decisions")

    return p


# ------------------------------------------------- reviewer-qualification rules
#
# Applied to `docs/evaluation/REVIEWER_ASSIGNMENT_RECORD.md` (A1-A9) and to the
# engagement brief (A10). They are dormant while both roles remain
# `NOT ASSIGNED`; they exist so that a real assignment cannot be recorded
# half-qualified, duplicated across roles, or attributed to a non-eligible
# identity.

ASSIGNMENT = ROOT / "docs" / "evaluation" / "REVIEWER_ASSIGNMENT_RECORD.md"
BRIEF = ROOT / "docs" / "evaluation" / "REVIEWER_ENGAGEMENT_BRIEF.md"

ROLE_MARKERS = ("STATISTICAL REVIEWER", "DOMAIN REVIEWER")
QUAL_LABELS = (
    "Assigned individual",
    "Affiliation",
    "Assignment date",
    "Conflict-of-interest declaration",
    "Assignment status",
    "Approval authority",
)
UNASSIGNED = {"", "NOT ASSIGNED", "NOT ESTABLISHED"}
# An identity is non-eligible if it *is* one of these, or *contains* one of the
# unambiguous substring markers. Substring matching is deliberately restricted to
# markers that cannot occur inside a real person's name.
NON_ELIGIBLE_EXACT = {
    "ai", "a.i.", "a.i", "ai system", "ai agent", "chatgpt", "gpt", "gpt4",
    "claude", "assistant", "codebuff", "buffy", "none", "tbd", "todo", "author",
    "project author", "repository author", "the author",
}
NON_ELIGIBLE_SUBSTR = (
    "chatgpt", "openai", "gpt-4", "codebuff", "ai agent", "ai system",
    "project author", "repository author", "the author",
)
BRIEF_STATUS = "READY FOR REVIEWER RECRUITMENT — REVIEW NOT YET COMMENCED"
DEPENDENCY_STATEMENT = (
    "Reviewer assignment remains an external dependency and cannot be completed "
    "from repository evidence alone."
)
BRIEF_ROLES = (
    "Statistical reviewer",
    "Domain reviewer",
    "dataset/provenance auditor",
    "Licence/terms reviewer",
)

def _field_value(block: str, label: str) -> str | None:
    """Value on the line(s) following a `Label:` line, or None if absent."""
    m = re.search(rf"(?mi)^\s*{re.escape(label)}:\s*\n+\s*([^\n]+)", block)
    return m.group(1).strip() if m else None


def _role_blocks(text: str) -> dict[str, str]:
    """Slice each `Role: <MARKER>` role block out of the record."""
    out: dict[str, str] = {}
    for marker in ROLE_MARKERS:
        i = text.find(marker)
        if i == -1:
            continue
        ends = [x for x in (text.find("\n### ", i), text.find("\n## ", i)) if x != -1]
        out[marker] = text[i:min(ends)] if ends else text[i:]
    return out


def _qual_block(text: str, marker: str) -> str:
    """The §3.x qualification table for a role (statistical -> 3.1, domain -> 3.2)."""
    m = re.search(r"### 3\.1.*?(?=\n### 3\.2)", text, re.S) if marker == ROLE_MARKERS[0] \
        else re.search(r"### 3\.2.*?(?=\n---)", text, re.S)
    return m.group(0) if m else ""


def _table_value(block: str, label: str) -> str | None:
    """Value cell of `| label | value |` (label may carry a parenthetical)."""
    m = re.search(rf"\|\s*{re.escape(label)}(?:\s*\([^)|]*\))?\s*\|\s*([^|]+)\|", block)
    return m.group(1).strip().strip("`").strip() if m else None


def check_assignment(text: str) -> list[str]:
    """A1-A9: reviewer-qualification rules for REVIEWER_ASSIGNMENT_RECORD.md."""
    p: list[str] = []
    norm = normalize(text)
    blocks = _role_blocks(text)
    if len(blocks) != 2 or any(m not in text for m in ROLE_MARKERS):
        p.append("A1 both reviewer role blocks (STATISTICAL REVIEWER, DOMAIN REVIEWER) are required")
        return p

    identities: dict[str, str] = {}
    for marker in ROLE_MARKERS:
        blk = blocks[marker]
        for label in QUAL_LABELS:
            if _field_value(blk, label) is None:
                p.append(f"A2 {marker}: qualification field missing or valueless: {label}")
        ident = _field_value(blk, "Assigned individual") or ""
        status = _field_value(blk, "Assignment status") or ""
        identities[marker] = ident
        assigned = ident.strip().upper() not in UNASSIGNED

        if status.strip().upper() != ident.strip().upper():
            p.append(f"A6 {marker}: Assignment status ({status!r}) disagrees with "
                     f"Assigned individual ({ident!r})")
        if not assigned and "APPROVED" in blk.upper():
            p.append(f"A7 {marker}: an approval appears while no reviewer identity exists")
        if not assigned:
            continue

        for label in ("Affiliation", "Assignment date",
                      "Conflict-of-interest declaration", "Approval authority"):
            v = (_field_value(blk, label) or "").strip().upper()
            if v in UNASSIGNED:
                p.append(f"A3 {marker}: named assignment with {label} = NOT ESTABLISHED")

        qual = _qual_block(text, marker)
        expertise_label = ("statistical" if marker == ROLE_MARKERS[0] else "fraud")
        expert = _table_value(qual, "Relevant expertise") or ""
        if expertise_label not in qual.lower():
            p.append(f"A4 {marker}: role-specific qualification label missing "
                     f"({expertise_label})")
        elif expert.upper().startswith("NOT ESTABLISHED") or not expert:
            p.append(f"A4 {marker}: named assignment with no role-specific expertise recorded")
        ident_row = _table_value(qual, "Identity") or ""
        if ident_row.upper().startswith("NOT ESTABLISHED") or not ident_row:
            p.append(f"A4 {marker}: named assignment with §3 qualification Identity unrecorded")

        low = ident.strip().strip("`").lower()
        if low in NON_ELIGIBLE_EXACT:
            p.append(f"A8 {marker}: non-eligible identity recorded: {ident!r}")
        for tok in NON_ELIGIBLE_SUBSTR:
            if tok in low:
                p.append(f"A8 {marker}: non-eligible identity marker in Assigned "
                         f"individual: {tok!r}")
        if "@" in ident:
            p.append(f"A8 {marker}: an email address is recorded as the reviewer identity")

    s_ident = identities[ROLE_MARKERS[0]].strip()
    d_ident = identities[ROLE_MARKERS[1]].strip()
    if (s_ident.upper() not in UNASSIGNED and d_ident.upper() not in UNASSIGNED
            and s_ident.lower() == d_ident.lower()):
        p.append("A5 the same person is named for both reviewer roles")
    if DEPENDENCY_STATEMENT not in norm:
        p.append("A9 the external-dependency statement is missing from the assignment record")
    return p


def check_brief(text: str) -> list[str]:
    """A10: the engagement brief must carry its scope and its true status."""
    p: list[str] = []
    norm = normalize(text)
    if BRIEF_STATUS not in norm:
        p.append("A10 engagement status line missing or changed")
    if "REVIEW NOT YET COMMENCED" not in norm:
        p.append("A10 the brief does not state that review has not commenced")
    if re.search(r"(?i)\bREVIEW COMPLETE\b", norm):
        p.append("A10 the brief claims review completion")
    for role in BRIEF_ROLES:
        if role.lower() not in norm.lower():
            p.append(f"A10 reviewer role missing from the brief: {role!r}")
    for i in range(1, 15):
        if f"/{i:02d}" not in norm:
            p.append(f"A10 decision /{i:02d} is not mentioned in the brief")
    if "N1–N13" not in norm and "N1-N13" not in norm:
        p.append("A10 the negative findings N1-N13 are not referenced in the brief")
    return p


def check(packet: Path = DEFAULT_PACKET, root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    targets = (
        (packet, check_text, "review package"),
        (root / "docs" / "evaluation" / "REVIEWER_ASSIGNMENT_RECORD.md",
         check_assignment, "assignment record"),
        (root / "docs" / "evaluation" / "REVIEWER_ENGAGEMENT_BRIEF.md",
         check_brief, "engagement brief"),
    )
    for path, fn, label in targets:
        if not path.exists():
            problems.append(f"{label} not found: {path}")
            continue
        problems.extend(fn(path.read_text(encoding="utf-8")))
    problems += check_repo_state(root)
    return problems


# ------------------------------------------------------------------ self-test

# ASCII-only anchors: a tamper source that cannot be found is itself a failure.
# (label, needle, replacement, occurrence count; -1 = every occurrence)
TAMPERS: list[tuple[str, str, str, int]] = [
    ("licence outcome flipped to F-P",
     "NOT ESTABLISHED`\n\n**Kaggle fraudTrain**",
     "F-P`\n\n**Kaggle fraudTrain**", 1),
    ("licence statement cell cleared",
     "| Licence statement | `NOT ESTABLISHED` |",
     "| Licence statement | `CLEARED` |", 1),
    ("decision marked APPROVED",
     "NOT ESTABLISHED`\n\n**PS-14 synthetic**",
     "APPROVED`\n\n**PS-14 synthetic**", 1),
    ("reviewer marked approved",
     "| Statistical reviewer | **`NOT ASSIGNED`** |",
     "| Statistical reviewer | **`APPROVED`** |", 1),
    ("50M claimed generated",
     "50M dataset generation = NOT AUTHORIZED / NOT YET SCIENTIFICALLY JUSTIFIED.",
     "The 50,000,000-row corpus was generated and is ready.", 1),
    ("section 12 removed",
     "## 12. 50M status",
     "## 12x. 50M status", 1),
    ("required dataset removed",
     "IEEE-CIS",
     "some-other-dataset", -1),
    ("audit item pre-certified",
     "Current packet state: `REQUIRES REVIEW`.",
     "Current packet state: `ESTABLISHED`.", 1),
    ("status line removed",
     "> # REVIEW PACKAGE READY / PASS WITH LIMITATIONS",
     "", 1),
    ("false audit claim",
     "**This is NOT an independent audit",
     "The independent audit has occurred: this packet is the audit", 1),
    ("observed item removed",
     "**O-3**",
     "**O-XX**", 1),
    ("required section renamed",
     "## 10. Reviewer input contract (mapped to existing fields)",
     "## 10x. Reviewer input contract", 1),
]


def _record_fixture(
    stat: str = "NOT ASSIGNED",
    dom: str = "NOT ASSIGNED",
    *,
    aff: str = "NOT ESTABLISHED",
    date: str = "NOT ESTABLISHED",
    conflict: str = "NOT ESTABLISHED",
    authority: str = "NOT ESTABLISHED",
    stat_status: str | None = None,
    dom_status: str | None = None,
    dependency: bool = True,
    stat_expertise: str = "statistical method evaluation experience",
    dom_expertise: str = "fraud-operations semantics experience",
) -> str:
    """Minimal assignment-record text used by the A-rule self-tests."""

    def block(marker: str, name: str, status: str | None) -> str:
        return (f"Role:\n{marker}\n\nAssigned individual:\n{name}\n\nAffiliation:\n{aff}\n\n"
                f"Assignment date:\n{date}\n\nConflict-of-interest declaration:\n{conflict}\n\n"
                f"Assignment status:\n{status or name}\n\nApproval authority:\n{authority}\n\n")

    def qual(heading: str, name: str, expert: str) -> str:
        return (f"### {heading}\n\n| Field | Value |\n|---|---|\n| Identity | {name} |\n"
                f"| Relevant expertise | {expert} |\n\n")

    s_on = stat.strip().upper() not in UNASSIGNED
    d_on = dom.strip().upper() not in UNASSIGNED
    text = (block("STATISTICAL REVIEWER", stat, stat_status)
            + block("DOMAIN REVIEWER", dom, dom_status)
            + qual("3.1 Statistical Reviewer — qualification",
                   stat if s_on else "NOT ESTABLISHED",
                   stat_expertise if s_on else "NOT ESTABLISHED")
            + qual("3.2 Domain Reviewer — qualification",
                   dom if d_on else "NOT ESTABLISHED",
                   dom_expertise if d_on else "NOT ESTABLISHED")
            + "---\n")
    if dependency:
        text += ("## 11. External dependency statement\n\n> Reviewer assignment remains "
                 "an external dependency and cannot be completed from repository "
                 "evidence alone.\n")
    return text


def self_test() -> int:
    src = DEFAULT_PACKET
    if not src.exists():
        print("SELF-TEST: FAIL — packet not found; cannot build fixtures")
        return 1
    base = src.read_text(encoding="utf-8")
    failures: list[str] = []
    checks = 0

    problems = check_text(base)
    if problems:
        failures.append(f"pristine packet reports {len(problems)} problem(s): "
                        + "; ".join(problems))

    for label, needle, repl, count in TAMPERS:
        checks += 1
        if needle not in base:
            failures.append(f"tamper anchor not found in the packet: {label}")
            continue
        text = base.replace(needle, repl) if count < 0 else base.replace(needle, repl, count)
        if text == base:
            failures.append(f"tamper '{label}' did not change the fixture")
            continue
        if not check_text(text):
            failures.append(f"tamper not detected: {label}")

    # --- assignment-record qualification rules (A1-A9) ---
    def case(label: str, text: str, expect_problems: bool) -> None:
        nonlocal checks
        checks += 1
        probs = check_assignment(text)
        if expect_problems and not probs:
            failures.append(f"tamper not detected: {label}")
        if not expect_problems and probs:
            failures.append(f"fixture {label} reports {len(probs)} problem(s): "
                            + "; ".join(probs))

    qualified = dict(aff="Independent University", date="2026-10-06",
                     conflict="None declared - independent of the project",
                     authority="review authority accepted")
    case("valid two-reviewer assignment",
         _record_fixture("Jane Roe", "John Doe", **qualified), expect_problems=False)
    case("same person in both roles",
         _record_fixture("Jane Roe", "Jane Roe", **qualified), expect_problems=True)
    case("approval recorded while unassigned",
         _record_fixture(authority="APPROVED"), expect_problems=True)
    case("non-eligible identity (assistant)",
         _record_fixture("ChatGPT", "John Doe", **qualified), expect_problems=True)
    case("named assignment without qualification fields",
         _record_fixture("Jane Roe"), expect_problems=True)
    case("assignment status disagrees with identity",
         _record_fixture("Jane Roe", stat_status="NOT ASSIGNED", **qualified),
         expect_problems=True)
    case("dependency statement missing",
         _record_fixture(dependency=False), expect_problems=True)
    case("role block missing",
         _record_fixture().replace("Role:\nDOMAIN REVIEWER", "Role:\nREMOVED"),
         expect_problems=True)
    case("qualification field label removed",
         _record_fixture().replace("Conflict-of-interest declaration:\n", ""),
         expect_problems=True)

    # --- the real artifacts must satisfy their own rules ---
    checks += 1
    probs = check_assignment(ASSIGNMENT.read_text(encoding="utf-8"))
    if probs:
        failures.append(f"live assignment record reports {len(probs)} problem(s): "
                        + "; ".join(probs))
    checks += 1
    brief_text = BRIEF.read_text(encoding="utf-8")
    probs = check_brief(brief_text)
    if probs:
        failures.append(f"live engagement brief reports {len(probs)} problem(s): "
                        + "; ".join(probs))
    for label, tampered in (
        ("brief claims review completion",
         brief_text + "\n\nREVIEW COMPLETE\n"),
        ("brief status line removed",
         brief_text.replace(BRIEF_STATUS, "READY FOR SOMETHING ELSE")),
    ):
        checks += 1
        if not check_brief(tampered):
            failures.append(f"tamper not detected: {label}")

    if failures:
        print(f"SELF-TEST: FAIL ({len(failures)} failure(s) across {checks} tampers)")
        for f in failures:
            print("  FAIL", f)
        return 1
    print(f"SELF-TEST: PASS (pristine clean; {checks}/{checks} tamper and "
          f"clean-fixture checks passed)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="review-package checker")
    ap.add_argument("--self-test", action="store_true", help="run fixture tests")
    ap.add_argument("--packet", type=Path, default=None)
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    problems = check(args.packet or DEFAULT_PACKET, args.root or ROOT)
    if problems:
        print(f"REVIEW PACKAGE CHECK: FAIL ({len(problems)} problem(s))")
        for prob in problems:
            print(f"  FAIL {prob}")
        return 1
    print(
        "REVIEW PACKAGE CHECK: PASS (13/13 sections in order, all required "
        "datasets present, no licence cleared, no reviewer assigned or approved, "
        "no 50M claim, all 15 audit items at REQUIRES REVIEW, freeze state "
        "unchanged, no dataset acquired; assignment record qualification rules "
        "A1-A9 satisfied, engagement brief rules A10 satisfied)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
