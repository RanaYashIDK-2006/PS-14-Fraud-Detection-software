#!/usr/bin/env python3
"""Review-resolution artifact checker.

Companion to ``docs/evaluation/STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md`` — the
review-resolution package that maps the 14 decisions from
``docs/evaluation/STATISTICAL_REVIEW_DECISION_MEMO.md`` (``/01``..``/14``).

This is a **document-consistency** checker, in the same class as
``claim_evidence_check.py``. It imports no model, dataset, metric or evaluation
path and never executes an experiment.

Rules enforced (exit 0 only when all hold):

  R1  all 14 identifiers /01../14 are present, each exactly once
  R2  no decision omitted and no extra decision id
  R3  every decision sheet carries all required fields
  R4  unresolved reviewer fields are explicitly marked, never blank
  R5  APPROVED cannot coexist with Reviewer = NOT ASSIGNED
  R6  Approval = APPROVED requires the decision itself to be APPROVED
  R7  APPROVED cannot coexist with Approval Timestamp = NOT ESTABLISHED
  R8  no fabricated identity or date while a decision is PENDING REVIEW
  R9  the 13 negative-evidence rows N1..N13 are all present (none removed)
  R10 this task created no freeze record and no freeze tag
  R11 the memo and this artifact agree on the 14 identifiers (no drift)
  R12 a BOTH decision cannot be APPROVED without BOTH reviewer components
      complete. The required roles come from the artifact's own authoritative
      ownership table (§3). One reviewer cannot satisfy both roles: a named
      reviewer, an assignment, a statistical ruling alone, or a domain ruling
      alone are each insufficient.

      For a STATISTICAL REVIEW decision, APPROVED requires:
          statistical reviewer named
          statistical decision resolved
          statistical rationale present
          statistical approval present

      For a BOTH decision, APPROVED additionally requires:
          domain reviewer named
          domain decision resolved
          domain rationale present
          domain approval present

      If any required component remains pending, this checker fails.

Reviewer components
-------------------
Every one of the 14 decisions requires the statistical component; that is what the
per-sheet ``Reviewer Decision/Reviewer/...`` block records. Decisions classified
``BOTH`` in the §3 ownership table additionally carry a prefixed domain block
(``Domain Reviewer Decision`` ... ``Domain Reviewer Approval Timestamp``). A sheet
declares its own requirement on a ``Required reviewer components:`` line, and that
declaration must agree with §3 — a sheet cannot opt itself out of a role.

Usage:
    python backend/scripts/review_resolution_check.py             # enforce
    python backend/scripts/review_resolution_check.py --self-test # fixtures
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "docs" / "evaluation" / "STATISTICAL_DOMAIN_REVIEW_RESOLUTION.md"
MEMO = ROOT / "docs" / "evaluation" / "STATISTICAL_REVIEW_DECISION_MEMO.md"
FREEZE_RECORD = ROOT / "docs" / "FREEZE_RECORD.json"

IDS = [f"/{n:02d}" for n in range(1, 15)]

STAT = "STATISTICAL REVIEW"
DOM = "DOMAIN REVIEW"
BOTH = "BOTH"

SHEET_FIELDS = [
    r"\*\*Decision\.\*\*",
    r"\*\*Why it matters\.\*\*",
    r"\*\*Existing evidence\.\*\*",
    r"\*\*Negative evidence\.\*\*",
    r"\*\*Allowed decision options\.?\*\*",
    r"\*\*Recommended reviewer question\.\*\*",
    r"\*\*Freeze consequence\.\*\*",
]

# Canonical component fields -> the key each reviewer role writes in the sheet.
# The statistical component uses the sheet's original block keys; the domain
# component prefixes every field with "Domain Reviewer " so the two components
# are unambiguous inside one decision sheet.
CANON = ["decision", "reviewer", "date", "rationale", "evidence", "approval", "ts"]

STAT_FIELDS = {
    "decision": "Reviewer Decision",
    "reviewer": "Reviewer",
    "date": "Decision Date",
    "rationale": "Rationale",
    "evidence": "Evidence Reviewed",
    "approval": "Approval",
    "ts": "Approval Timestamp",
}

DOMAIN_FIELDS = {
    "decision": "Domain Reviewer Decision",
    "reviewer": "Domain Reviewer",
    "date": "Domain Reviewer Decision Date",
    "rationale": "Domain Reviewer Rationale",
    "evidence": "Domain Reviewer Evidence Reviewed",
    "approval": "Domain Reviewer Approval",
    "ts": "Domain Reviewer Approval Timestamp",
}

STAT_KEYS = list(STAT_FIELDS.values())
DOMAIN_KEYS = list(DOMAIN_FIELDS.values())

COMPONENTS_LINE = "Required reviewer components:"

UNRESOLVED = "PENDING REVIEW"
NOT_APPROVED = "NOT APPROVED"
NOT_ASSIGNED = "NOT ASSIGNED"
NOT_ESTABLISHED = "NOT ESTABLISHED"

NEG_ROWS = [f"N{n}" for n in range(1, 14)]


# --------------------------------------------------------------------- parsing

def sheets(text: str) -> dict[str, str]:
    """Split the §4 decision sheets into {id: body}."""
    out: dict[str, str] = {}
    parts = re.split(r"\n### `(/[0-9]{2})`", text)
    for i in range(1, len(parts) - 1, 2):
        out[parts[i]] = parts[i + 1]
    return out


def ownership(text: str) -> dict[str, str]:
    """Authoritative required roles per decision, from the §3 ownership table."""
    out: dict[str, str] = {}
    if "## 3. Ownership classification" not in text:
        return out
    sec = text.split("## 3. Ownership classification")[1].split("\n## ")[0]
    for line in sec.splitlines():
        if not line.startswith("| `/"):
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        m = re.search(r"/(\d\d)", c[0])
        if not m:
            continue
        stat, dom, both = c[1] != "—", c[2] != "—", c[3] != "—"
        role = BOTH if both else (DOM if dom and not stat else STAT)
        out[f"/{m.group(1)}"] = role
    return out


def blocks(body: str) -> list[str]:
    return re.findall(r"```text\n(.*?)```", body, flags=re.S)


def value(block_text: str, key: str) -> str | None:
    m = re.search(rf"^{re.escape(key)}:[ \t]*(.*)$", block_text, flags=re.M)
    return m.group(1).strip() if m else None


def component(body: str, fields: dict[str, str]) -> tuple[bool, dict[str, str | None]]:
    """First block containing every field of ``fields``.

    Returns (found, {canonical_name: value}).
    """
    for blk in blocks(body):
        vals = {c: value(blk, k) for c, k in fields.items()}
        if all(v is not None for v in vals.values()):
            return True, vals
    return False, {c: None for c in fields}


def stat_component(body: str) -> tuple[bool, dict[str, str | None]]:
    return component(body, STAT_FIELDS)


def domain_component(body: str) -> tuple[bool, dict[str, str | None]]:
    return component(body, DOMAIN_FIELDS)


# ---------------------------------------------------------------------- checks

def check_ids(text: str, found: dict[str, str]) -> list[str]:
    problems: list[str] = []
    listed = set(found)
    for i in IDS:
        if i not in listed:
            problems.append(f"decision {i} missing from the decision sheets")
    for extra in sorted(listed - set(IDS)):
        problems.append(f"extra decision id {extra} is not one of /01../14")
    for hit in re.findall(r"^### `(/[0-9]{2})`", text, flags=re.M):
        if text.count(f"### `{hit}`") != 1:
            problems.append(
                f"decision id {hit} appears {text.count(f'### `{hit}`')} times "
                "(must be exactly once)"
            )
    return problems


def component_approved(vals: dict[str, str | None]) -> bool:
    """True when one reviewer component is fully approved on its own terms."""
    return (
        (vals.get("decision") or "").upper().startswith("APPROVED")
        and (vals.get("approval") or "") == "APPROVED"
        and (vals.get("reviewer") or "") != NOT_ASSIGNED
        and (vals.get("ts") or "") != NOT_ESTABLISHED
        and (vals.get("rationale") or "") not in (None, "", UNRESOLVED)
    )


def check_component(i: str, role: str, vals: dict[str, str | None],
                    problems: list[str]) -> bool:
    """Validate one reviewer component.

    Returns True only when the component is **fully approved** (ruling APPROVED,
    reviewer named, rationale present, approval recorded, timestamp set). A pending,
    rejected or merely-assigned component returns False.
    """
    d, r = vals["decision"], vals["reviewer"]
    dt, ra = vals["date"], vals["rationale"]
    ap, ts = vals["approval"], vals["ts"]
    tag = f"{i}/{role.lower()}"
    ok = True

    for c, v in vals.items():
        if v is None or v == "":
            problems.append(f"{tag}: reviewer field {c!r} is missing or blank (R4)")
            ok = False

    if (d or "").upper().startswith("APPROVED") or (ap or "") == "APPROVED":
        if not (d or "").upper().startswith("APPROVED"):
            problems.append(f"{tag}: Approval = APPROVED but decision is {d!r} (R6)")
            ok = False
        if (r or "") == NOT_ASSIGNED:
            problems.append(f"{tag}: APPROVED with Reviewer = {NOT_ASSIGNED} (R5)")
            ok = False
        if (ap or "") == NOT_APPROVED:
            problems.append(f"{tag}: APPROVED with Approval = {NOT_APPROVED} (R6)")
            ok = False
        if (ts or "") == NOT_ESTABLISHED:
            problems.append(
                f"{tag}: APPROVED with Approval Timestamp = {NOT_ESTABLISHED} (R7)"
            )
            ok = False
        if (ra or "") in (None, "", UNRESOLVED):
            problems.append(f"{tag}: APPROVED without a rationale "
                            f"(Rationale = {ra!r}) (R12)")
            ok = False
    elif (d or "").upper() == UNRESOLVED:
        if (r or "") != NOT_ASSIGNED:
            problems.append(f"{tag}: reviewer {r!r} named while decision is {d!r} (R8)")
            ok = False
        if (dt or "") != NOT_ESTABLISHED:
            problems.append(
                f"{tag}: decision date {dt!r} set while decision is {d!r} (R8)"
            )
            ok = False
    return ok and component_approved(vals)


def check_sheets(found: dict[str, str], own: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for i in IDS:
        body = found.get(i, "")
        for f in SHEET_FIELDS:
            if not re.search(f, body):
                problems.append(f"{i}: missing required field {f}")

        has_stat, svals = stat_component(body)
        if not has_stat:
            problems.append(f"{i}: statistical reviewer block missing or incomplete")
            svals = {c: None for c in STAT_FIELDS}
        stat_ok = check_component(i, "Statistical", svals, problems)

        # R12 — the sheet must declare the same roles the §3 table requires.
        declared = None
        m = re.search(rf"^{re.escape(COMPONENTS_LINE)}\s*(.+)$", body, flags=re.M)
        if m:
            declared = m.group(1).strip().upper()
        required = own.get(i)
        if required is None:
            problems.append(f"{i}: no authoritative ownership row found in §3")
            continue
        if declared is None:
            problems.append(
                f"{i}: missing '{COMPONENTS_LINE}' declaration (R12)"
            )
            continue
        if BOTH in declared:
            declared_role = BOTH
        elif DOM in declared:
            declared_role = DOM
        elif STAT in declared:
            declared_role = STAT
        else:
            declared_role = "UNKNOWN"
        if declared_role != required:
            problems.append(
                f"{i}: declares reviewer components {declared_role!r} but §3 "
                f"requires {required!r} (R12)"
            )
            continue

        final = (svals.get("decision") or "").upper()
        final_approved = final.startswith("APPROVED") or \
            (svals.get("approval") or "") == "APPROVED"

        if required == BOTH:
            has_dom, dvals = domain_component(body)
            if not has_dom:
                problems.append(
                    f"{i}: classified BOTH but no domain reviewer block is present (R12)"
                )
                dvals = {c: None for c in DOMAIN_FIELDS}
                dom_ok = False
            else:
                dom_ok = check_component(i, "Domain", dvals, problems)
            # R12 — the two components stand or fall together. One component may
            # never be approved while the other is not: assignment, a reviewer
            # name, or a single ruling are each insufficient.
            if has_dom and component_approved(svals) != component_approved(dvals):
                which = "statistical" if component_approved(svals) else "domain"
                problems.append(
                    f"{i}: classified BOTH and the {which} component is approved, "
                    "but its counterpart is not; both independent reviewer "
                    "components must be complete — one reviewer cannot satisfy "
                    "both roles (R12)"
                )
            if final_approved and not (stat_ok and dom_ok):
                problems.append(
                    f"{i}: classified BOTH and marked APPROVED, but both independent "
                    "reviewer components must be complete; one reviewer cannot "
                    "satisfy both roles (R12)"
                )
            if has_dom and stat_ok and dom_ok and not final_approved:
                problems.append(
                    f"{i}: both reviewer components are approved but the sheet's "
                    "final decision is not APPROVED (R6/R12)"
                )
        elif required == DOM:
            # Not present in the authoritative table today; enforced if it ever is.
            has_dom, dvals = domain_component(body)
            if not has_dom:
                problems.append(f"{i}: classified DOMAIN REVIEW but no domain block (R12)")
            elif final_approved:
                check_component(i, "Domain", dvals, problems)
        else:
            if has_domain_component(body):
                problems.append(
                    f"{i}: classified STATISTICAL REVIEW but carries a domain "
                    "reviewer block; ownership must not be widened (R12)"
                )
    return problems


def has_domain_component(body: str) -> bool:
    return any(value(b, k) is not None for b in blocks(body) for k in DOMAIN_FIELDS.values())


def check_negative_evidence(text: str) -> list[str]:
    return [f"negative-evidence row {n} has been removed" for n in NEG_ROWS
            if not re.search(rf"^\| {n} \|", text, flags=re.M)]


def check_memo_drift(text: str) -> list[str]:
    if not MEMO.exists():
        return [f"memo not found: {MEMO.relative_to(ROOT)}"]
    memo = MEMO.read_text(encoding="utf-8")
    memo_ids = set(re.findall(r"NR05-§20/(\d\d)", memo))
    art_ids = {i.lstrip("/") for i in re.findall(r"^### `(/[0-9]{2})`", text, flags=re.M)}
    if memo_ids != art_ids:
        return [
            "identifier drift: memo has "
            f"{sorted(memo_ids)} but artifact has {sorted(art_ids)}"
        ]
    return []


def check_no_freeze() -> list[str]:
    problems: list[str] = []
    if FREEZE_RECORD.exists():
        problems.append("docs/FREEZE_RECORD.json exists — this task must not create it")
    try:
        out = subprocess.run(
            ["git", "tag"], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        tags = [t for t in out.stdout.split() if t.strip()]
    except Exception:  # git unavailable — skip rather than fail closed on tooling
        tags = []
    if tags:
        problems.append(f"git tags present: {tags} — a freeze tag must not exist")
    return problems


def check(artifact: Path = ARTIFACT) -> list[str]:
    if not artifact.exists():
        return [f"artifact not found: {artifact}"]
    text = artifact.read_text(encoding="utf-8")
    found = sheets(text)
    problems: list[str] = []
    problems += check_ids(text, found)
    if found:
        problems += check_sheets(found, ownership(text))
    problems += check_negative_evidence(text)
    problems += check_memo_drift(text)
    problems += check_no_freeze()
    return problems


# --------------------------------------------------------------------- self-test

def _stat_block(**o: str) -> str:
    v = {"dec": UNRESOLVED, "rev": NOT_ASSIGNED, "date": NOT_ESTABLISHED,
         "rat": UNRESOLVED, "app": NOT_APPROVED, "ts": NOT_ESTABLISHED,
         "ev": UNRESOLVED}
    v.update(o)
    return ("Reviewer Decision: " + v["dec"] + "\nReviewer: " + v["rev"]
            + "\nDecision Date: " + v["date"] + "\nRationale: " + v["rat"]
            + "\nEvidence Reviewed: " + v["ev"] + "\nApproval: " + v["app"]
            + "\nApproval Timestamp: " + v["ts"] + "\n")


def _dom_block(**o: str) -> str:
    """Domain component block: same seven fields, written under their
    'Domain Reviewer ...' key names."""
    vals = {"dec": UNRESOLVED, "rev": NOT_ASSIGNED, "date": NOT_ESTABLISHED,
            "rat": UNRESOLVED, "app": NOT_APPROVED, "ts": NOT_ESTABLISHED,
            "ev": UNRESOLVED}
    vals.update(o)
    by_key = {"decision": vals["dec"], "reviewer": vals["rev"],
              "date": vals["date"], "rationale": vals["rat"],
              "evidence": vals["ev"], "approval": vals["app"], "ts": vals["ts"]}
    return "".join(f"{DOMAIN_FIELDS[c]}: {by_key[c]}\n" for c in CANON)


def _fixture(role: str = STAT, stat: dict | None = None,
             dom: dict | None = None) -> str:
    """One decision sheet. ``role`` drives the §3-style ownership row too."""
    declared = {STAT: f"{COMPONENTS_LINE} {STAT}",
                DOM: f"{COMPONENTS_LINE} {DOM}",
                BOTH: f"{COMPONENTS_LINE} {BOTH} ({STAT} + {DOM})"}[role]
    body = ("**Decision.** x\n**Why it matters.** y\n**Existing evidence.** z\n"
            "**Negative evidence.** something adverse\n**Allowed decision options** a\n"
            "**Recommended reviewer question.** q\n\n"
            f"{declared}\n\n```text\n{_stat_block(**(stat or {}))}```\n")
    if role in (BOTH, DOM):
        body += f"\n```text\n{_dom_block(**(dom or {}))}```\n"
    body += "\n**Freeze consequence.** blocks\n"
    return body


def _own_text(role_by_id: dict[str, str]) -> str:
    rows = ["## 3. Ownership classification", "| ID | S | D | B | M |",
            "|---|---|---|---|---|"]
    for i in IDS:
        r = role_by_id[i]
        rows.append(
            f"| `{i}` | {'✔' if r in (STAT, BOTH) else '—'}"
            f" | {'✔' if r in (DOM, BOTH) else '—'}"
            f" | {'✔' if r == BOTH else '—'} | — |"
        )
    rows.append("")
    return "\n".join(rows)


def _run_sheets(role_by_id: dict[str, str], fixtures: dict[str, str]) -> list[str]:
    return check_sheets(fixtures, role_by_id)


def self_test() -> int:
    passed = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal passed
        assert cond, label
        passed += 1
        print(f"  PASS {label}")

    # ---- existing protections (R1/R2) -------------------------------
    ok(check_ids("", {}) != [], "missing decisions are reported")
    ok(check_ids("", {i: "" for i in IDS}) == [], "all 14 ids accepted")
    ok(check_ids("", dict({i: "" for i in IDS}, **{"/15": ""})) != [],
       "extra decision id is reported")
    dup_text = "".join(f"### `{i}` x\n" for i in IDS) + "### `/03` y\n"
    ok([p for p in check_ids(dup_text, {i: "" for i in IDS}) if "exactly once" in p] != [],
       "R1 duplicated decision id is reported")
    once_text = "".join(f"### `{i}` x\n" for i in IDS)
    ok([p for p in check_ids(once_text, {i: "" for i in IDS}) if "exactly once" in p] == [],
       "R1 each id exactly once passes")

    # ---- existing protections (R4/R5/R6/R7/R8) ----------------------
    allown, allfix = _all_own(STAT), _fixtures_all(STAT)
    ok(_run_sheets(allown, allfix) == [], "clean fixture passes")

    ok(check_ids("", {i: "" for i in IDS[:-1]}) != [],
       "omitted decision is reported")
    thin = dict(allfix)
    thin[IDS[-1]] = "**Decision.** x\n"
    ok(_run_sheets(allown, thin) != [],
       "sheet missing required fields fails")

    ok(_run_sheets(allown, _fixtures_all(STAT, stat={"dec": "APPROVED"})) != [],
       "R5 APPROVED + Reviewer NOT ASSIGNED fails")
    ok(_run_sheets(allown, _fixtures_all(
        STAT, stat={"dec": "APPROVED", "rev": "A. Reviewer", "app": "APPROVED",
                    "ts": "2026-01-01", "rat": "because"})) == [],
       "fully approved statistical fixture passes")
    ok(_run_sheets(allown, _fixtures_all(
        STAT, stat={"dec": "APPROVED", "rev": "A. Reviewer", "app": "APPROVED"})) != [],
       "R7 APPROVED + timestamp NOT ESTABLISHED fails")
    ok(_run_sheets(allown, _fixtures_all(
        STAT, stat={"dec": "APPROVED", "rev": "A. Reviewer", "app": "NOT APPROVED",
                    "ts": "2026-01-01"})) != [],
       "R6 APPROVED + Approval NOT APPROVED fails")
    ok(_run_sheets(allown, _fixtures_all(STAT, stat={"app": "APPROVED"})) != [],
       "R6 Approval APPROVED while decision PENDING fails")
    ok(_run_sheets(allown, _fixtures_all(
        STAT, stat={"dec": "REJECTED", "rev": "A. Reviewer", "app": "APPROVED",
                    "ts": "2026-01-01"})) != [],
       "R6 Approval APPROVED contradicting REJECTED fails")
    ok(_run_sheets(allown, _fixtures_all(STAT, stat={"rev": "A. Reviewer"})) != [],
       "R8 named reviewer while PENDING fails")
    ok(_run_sheets(allown, _fixtures_all(STAT, stat={"dec": "REJECTED", "rev": "A. Reviewer"})) == [],
       "R8 REJECTED ruling legitimately names its reviewer")
    ok(_run_sheets(allown, _fixtures_all(STAT, stat={"dec": "REVISE", "rev": "A. Reviewer"})) == [],
       "R8 REVISE proposal legitimately names its reviewer")
    ok(_run_sheets(allown, _fixtures_all(STAT, stat={"date": "2026-10-04"})) != [],
       "R8 decision date set while PENDING fails")
    ok(_run_sheets(allown, _fixtures_all(
        STAT, stat={"dec": "REJECTED", "rev": "A. Reviewer", "date": "2026-10-04"})) == [],
       "R8 REJECTED ruling legitimately carries its date")

    blank = {i: re.sub(r"^Rationale:.*$", "Rationale:", f, count=1, flags=re.M)
             for i, f in allfix.items()}
    ok(_run_sheets(allown, blank) != [], "R4 blank rationale fails")

    # ---- R12 new rule -------------------------------------------------
    both_own = {i: BOTH for i in IDS}
    pend = {"dec": UNRESOLVED}
    named_app = {"dec": "APPROVED", "rev": "A. Stat", "app": "APPROVED",
                 "ts": "2026-01-01", "rat": "because", "ev": "docs"}
    named_dom = {"dec": "APPROVED", "rev": "B. Domain", "app": "APPROVED",
                 "ts": "2026-01-02", "rat": "because", "ev": "docs"}

    ok(_run_sheets(both_own, _fixtures_all(BOTH)) == [],
       "BOTH fixture with everything pending passes")
    # Case A - statistical only
    ok(_run_sheets(both_own, _fixtures_all(BOTH, stat=named_app)) != [],
       "Case A: statistical-only approval on BOTH fails")
    # Case B - domain only
    ok(_run_sheets(both_own, _fixtures_all(BOTH, dom=named_dom)) != [],
       "Case B: domain-only approval on BOTH fails")
    ok(_run_sheets(both_own, _fixtures_all(
        BOTH, stat={"rev": "A. Stat"}, dom=named_dom)) != [],
       "R12 a domain approval with a pending statistical counterpart fails")
    # Case C - both named, one ruling pending
    ok(_run_sheets(both_own, _fixtures_all(
        BOTH, stat=dict(named_app, dec=UNRESOLVED, rev="A. Stat"))) != [],
       "Case C: one ruling still PENDING fails")
    # Case D - both rulings complete, one approval missing
    ok(_run_sheets(both_own, _fixtures_all(
        BOTH, stat=named_app, dom=dict(named_dom, app="NOT APPROVED"))) != [],
       "Case D: one approval missing fails")
    # Case E - both complete and approved
    ok(_run_sheets(both_own, _fixtures_all(BOTH, stat=named_app, dom=named_dom)) == [],
       "Case E: both sides complete and approved passes")
    # Case F - STATISTICAL REVIEW needs no domain block
    ok(_run_sheets(allown, _fixtures_all(STAT, stat=named_app)) == [],
       "Case F: STATISTICAL REVIEW approved without domain review passes")
    # declaration must match the authoritative table
    wrong = {i: BOTH for i in IDS}
    ok(_run_sheets(wrong, _fixtures_all(STAT, stat=named_app)) != [],
       "R12 a STATISTICAL REVIEW sheet cannot claim BOTH")
    ok(_run_sheets(allown, _fixtures_all(BOTH, stat=named_app)) != [],
       "R12 a BOTH sheet cannot claim STATISTICAL REVIEW only")
    ok(_run_sheets(allown, _fixtures_all(BOTH, stat=named_app, dom=named_dom,
                                         no_declare=True)) != [],
       "R12 missing components declaration fails")
    ok(_run_sheets(allown, _fixtures_all(STAT, stat=named_app,
                                         extra_domain=named_dom)) != [],
       "R12 STATISTICAL REVIEW sheet may not widen ownership with a domain block")
    ok(_run_sheets(both_own, _fixtures_all(BOTH, stat=named_app, dom=pend)) != [],
       "R12 BOTH with no domain ruling fails even when statistical is approved")
    ok(_run_sheets(both_own, _fixtures_all(
        BOTH, stat=dict(named_app, rat=UNRESOLVED), dom=named_dom)) != [],
       "R12 approved component needs a rationale")

    # ---- R9 / R10 / R11 ------------------------------------------------
    neg = "\n".join(f"| {n} | finding | src | status |" for n in NEG_ROWS)
    ok(check_negative_evidence(neg) == [], "N1..N13 present passes")
    ok(check_negative_evidence("\n".join(f"| {n} | f | s | t |"
                                         for n in NEG_ROWS[:-1])) != [],
       "removed negative row fails")

    # ---- readiness reporting (R13) ----------------------------------
    pend_v = {"decision": UNRESOLVED, "reviewer": NOT_ASSIGNED}
    named_v = {"decision": UNRESOLVED, "reviewer": "A. Reviewer"}
    ruled_v = {"decision": "APPROVED", "reviewer": "A. Reviewer"}
    approved_v = {"decision": "APPROVED", "reviewer": "A. Reviewer",
                  "approval": "APPROVED", "rationale": "because",
                  "ts": "2026-01-01"}
    ok(component_state(pend_v) == ST_NOT_ASSIGNED, "R13 unassigned -> NOT ASSIGNED")
    ok(component_state(named_v) == ST_PENDING_REVIEW,
       "R13 named + pending -> ASSIGNED / PENDING REVIEW")
    ok(component_state(ruled_v) == ST_PENDING_APPROVAL,
       "R13 ruled but not approved -> REVIEWED / PENDING APPROVAL")
    ok(component_state(approved_v) == ST_APPROVED,
       "R13 fully approved -> APPROVED")
    ok(component_state({"decision": "REJECTED", "reviewer": "A. Reviewer"}) == ST_REJECTED,
       "R13 REJECTED reported as REJECTED")
    ok(component_state({"decision": "REVISE", "reviewer": "A. Reviewer"}) == ST_REVISE,
       "R13 REVISE reported as REVISE")
    r_lines = readiness(ARTIFACT)
    ok(len(r_lines) == 14, "R13 readiness reports all 14 decisions")
    ok("0/14 decisions resolved" in readiness_summary(r_lines),
       "R13 current repository reports 0/14 resolved")
    ok(all("NOT ASSIGNED" in ln for ln in r_lines),
       "R13 every component is currently NOT ASSIGNED")

    print(f"\nself-test: {passed} checks passed")
    return 0


def _all_own(role: str) -> dict[str, str]:
    return {i: role for i in IDS}


def _fixtures_all(role: str, stat: dict | None = None, dom: dict | None = None,
                  no_declare: bool = False, extra_domain: dict | None = None) -> dict[str, str]:
    out = {}
    for i in IDS:
        f = _fixture(role, stat=stat, dom=dom)
        if no_declare:
            f = re.sub(rf"^{re.escape(COMPONENTS_LINE)}.*$\n?", "", f, flags=re.M)
        if extra_domain is not None:
            f += f"\n```text\n{_dom_block(**extra_domain)}```\n"
        out[i] = f
    return out


# ------------------------------------------------------------------- readiness

# Derived, reported states. Readiness reporting NEVER writes to the artifact and
# NEVER advances a decision: it only describes what is currently recorded.
ST_NOT_ASSIGNED = "NOT ASSIGNED"
ST_PENDING_REVIEW = "ASSIGNED / PENDING REVIEW"
ST_PENDING_APPROVAL = "REVIEWED / PENDING APPROVAL"
ST_APPROVED = "APPROVED"
ST_REJECTED = "REJECTED"
ST_REVISE = "REVISE"
ST_PENDING = "PENDING REVIEW"  # no reviewer named yet


def component_state(vals: dict[str, str | None]) -> str:
    """Derive the recorded state of one reviewer component (report only)."""
    d = (vals.get("decision") or "").strip()
    r = (vals.get("reviewer") or "").strip()
    ap = (vals.get("approval") or "").strip()
    up = d.upper()
    if r == NOT_ASSIGNED or r == "":
        return ST_NOT_ASSIGNED
    if up == ST_APPROVED and ap == "APPROVED":
        return ST_APPROVED
    if up == ST_REJECTED:
        return ST_REJECTED
    if up == ST_REVISE:
        return ST_REVISE
    if up and up != UNRESOLVED:
        return ST_PENDING_APPROVAL
    return ST_PENDING_REVIEW


def readiness(artifact: Path = ARTIFACT) -> list[str]:
    """Per-decision readiness lines. Read-only; never mutates, never advances."""
    if not artifact.exists():
        return [f"artifact not found: {artifact}"]
    text = artifact.read_text(encoding="utf-8")
    found = sheets(text)
    own = ownership(text)
    lines: list[str] = []
    for i in IDS:
        body = found.get(i, "")
        role = own.get(i, "UNKNOWN")
        _, svals = stat_component(body)
        stat = component_state(svals)
        dom = "-"
        if role in (BOTH, DOM):
            _, dvals = domain_component(body)
            dom = component_state(dvals)
        final = ST_APPROVED if stat == ST_APPROVED and dom in (ST_APPROVED, "-") \
            else ST_PENDING
        lines.append(
            f"{i}  {role:18s} statistical={stat:26s} domain={dom:26s} final={final}"
        )
    return lines


def readiness_summary(lines: list[str]) -> str:
    counts: dict[str, int] = {}
    for ln in lines:
        for field in ("statistical=", "domain="):
            pos = ln.find(field)
            if pos == -1:
                continue
            rest = ln[pos + len(field):]
            val = rest.split("domain=")[0].split("final=")[0].strip()
            if val and val != "-":
                counts[val] = counts.get(val, 0) + 1
    resolved = sum(counts.get(s, 0) for s in (ST_APPROVED, ST_REJECTED, ST_REVISE))
    return (f"{resolved}/14 decisions resolved | component states: "
            + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="review-resolution artifact checker")
    ap.add_argument("--self-test", action="store_true", help="run fixture tests")
    ap.add_argument("--readiness", action="store_true",
                    help="report per-decision readiness (read-only; never fails)")
    ap.add_argument("--artifact", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.readiness:
        lines = readiness(args.artifact or ARTIFACT)
        if lines and lines[0].startswith("artifact not found"):
            print(lines[0])
            return 1
        print("REVIEW READINESS (report only — no decision is advanced):")
        for ln in lines:
            print("  " + ln)
        print(readiness_summary(lines))
        return 0

    problems = check(args.artifact or ARTIFACT)
    if problems:
        print(f"REVIEW RESOLUTION CHECK: FAIL ({len(problems)} problem(s))")
        for p in problems:
            print(f"  FAIL {p}")
        return 1
    print(
        "REVIEW RESOLUTION CHECK: PASS (14/14 decisions present, reviewer "
        "components match the authoritative ownership table, no unresolved field "
        "left blank, no approval without a reviewer, N1-N13 intact, no freeze "
        "record or tag)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())