# PS-14 — Credential Rotation and PII Re-Encryption (Phase 4A)

**Document status:** final report for Phase 4A.
**Type:** operational security remediation.
**Bounded conclusion (phase brief §24 wording):**

> The previously exposed local credentials have been rotated and the affected PII encryption material migrated to the versioned replacement key. Historical repository exposure remains part of the project's security history.

Stated precisely: that sentence covers the **JWT signing secret**, the **internal service token** and the **PII data
key**. It does **not** cover the Supabase `service_role` credential, whose rotation requires the account owner
(see §10 and §19). Git history was **not** rewritten and is not claimed to be erased.

**Final decision (phase brief §21):** `REQUIRES OWNER ACTION` — detail: `OWNER ACTION REQUIRED — PROVIDER CREDENTIAL ROTATION`

Machine-readable companion: [PHASE4A_CREDENTIAL_ROTATION_EVIDENCE.json](PHASE4A_CREDENTIAL_ROTATION_EVIDENCE.json).
Closeout: [PHASE_CREDENTIAL_ROTATION_CLOSEOUT.md](PHASE_CREDENTIAL_ROTATION_CLOSEOUT.md).

---

## 1. Exposure inventory

Values are never reproduced here. "Historical exposure" means the value was committed to the repository at some
point (and, for the JWT secret, is still recoverable from history).

| Credential | Location (current) | Historical exposure | Current use | Rotation mechanism | Status |
|---|---|---|---|---|---|
| Supabase `service_role` credential + project URL | `.env` (gitignored); literals removed from `backend/scripts/smoke_test_supabase.py` in Phase 4 | **YES** — tracked file, commit `6e0f9f8` (2026-09-12) | optional remote Postgres smoke test only; **no local service reads it** (§3) | provider dashboard/API by the account owner | **PENDING — OWNER ACTION** |
| JWT signing secret | `.env` (gitignored; newly generated) | **YES** — `backend/scripts/start_risk_loadtest.py` (Phase 4 remediation) | identity-service HS256 sign/verify | application secret rotation (done) | **ROTATED** |
| Internal service token | `.env` (gitignored; newly generated) | **YES** — a *different* literal (`loadtest-prod-token-2026`) was tracked; the live value once reached a gitignored log during a Phase-4 probe and that log has since been truncated | service-to-service auth (`X-Internal-Token`) | application rotation (done) | **ROTATED** |
| PII data key material | `.env`: `PII_KEY_CURRENT` (new), `PII_KEY_LEGACY` + `PII_ENCRYPTION_KEY` (previous, migration-read only) | **NO** — never committed | PII field encryption at rest | versioned key rotation + re-encryption (done) | **ROTATED + MIGRATED** |
| Export signing key, compliance token, admin passphrase | `.env` (gitignored) | **NO** | export signatures, compliance gate, admin session bootstrap | not in scope (not exposed) | UNCHANGED |
| `ACCESS_DOC_KEY` | `.env` (gitignored) | **NO** | `docs/ACCESS.md.enc` | separate key, unaffected | UNCHANGED |

---

## 2. Root cause

Two utility scripts in the tracked tree carried live credentials as literals, and the repository had no
CI-enforced guard against that until Phase 4:

1. `backend/scripts/smoke_test_supabase.py` — `service_role` JWT + project URL literals (F-04).
2. `backend/scripts/start_risk_loadtest.py` — the live `JWT_SECRET` value and an internal-token literal (F-05).

`PII_KEY_*` / `PII_ENCRYPTION_KEY` were never committed; the PII key rotation in this phase is therefore not a
*response to exposure* but the establishment of the capability the phase requires — a PII key that rotates
independently of authentication, with a real migration executed and verified.

---

## 3. Credential dependencies (verified, not assumed)

| Secret | Depends on | Verified how |
|---|---|---|
| JWT signing secret | identity tokens only (`jwt.encode`/`jwt.decode` in `identity_service/security.py`) | source map + live proof: a token signed with the old secret now returns 401 while the same claims signed with the new secret return 200 |
| PII data key | PII ciphertext at rest; `settings.fernet_key`; admin TOTP derivation; a non-secret key fingerprint in `monitoring/security_hardening.py` | candidate-key probe over **every** stored token (below); admin TOTP secret absent on this host; fingerprint is derived at runtime and not persisted |
| Internal token | all five services' `/internal/*` routes | rotated and probed on two services (risk evaluate, privacy ingest): old → 401, new → 200 |
| Supabase credential | `backend/scripts/smoke_test_supabase.py` (env-driven since Phase 4), `backend/docker-compose.yml` (prod profile), `.env` | `grep` over `backend/src`: the front service only *strips* these variables from child environments — **no local runtime dependency** |

**Corrected claim.** The repository's working notes asserted that `settings.jwt_secret` also derives the PII
`fernet_key`. That is **false for data at rest** and this phase verified it rather than trusting it:

| Candidate key | Tokens decrypted (db/identity.db) |
|---|---|
| `base64(sha256(PII_ENCRYPTION_KEY))` — the real `settings.fernet_key` | **2487 / 2487** |
| `base64(sha256(JWT_SECRET))` | 0 / 2487 |
| `base64(sha256("dek:" + PII_ENCRYPTION_KEY))` (`key_hierarchy.py` DEK) | 0 / 2487 |
| `base64(sha256("dek:" + JWT_SECRET))` | 0 / 2487 |

`jwt_secret` **cannot** decrypt any stored PII, and the `key_hierarchy` DEK/KEK path has no consumers in `src/` or
`scripts/`. Rotating either secret therefore cannot silently break the other — which is what makes this phase's
ordering (migrate PII, then rotate JWT) safe rather than lucky. The stale note is superseded by this measurement.

---

## 4. PII encryption architecture (before → after)

**Before**

```
PII_ENCRYPTION_KEY ──sha256──► Fernet key ──► users.full_name / phone_encrypted / email_encrypted
                                             (untagged Fernet tokens, no version marker)
JWT_SECRET ────────────────────────────────► identity tokens (independent, verified above)
```

**After (versioned, independently rotatable)**

```
PII_KEY_CURRENT ──sha256──► Fernet(v2) ──► new writes: "v2:" + token
PII_KEY_LEGACY  ──sha256──► Fernet(v1) ──► read-only for untagged legacy rows until migration completes
JWT_SECRET ─────────────────────────────► identity tokens      (no PII coupling)
```

Implementation: `backend/src/pii_crypto.py` (new) — `PiiCipher` holds a version→key map, writes tagged tokens when
more than one key version is present, dispatches reads by tag, and raises `UnknownKeyVersion` rather than guessing.
`settings.py` gains `pii_key_current`, `pii_key_legacy`, `pii_key_version` (env: `PII_KEY_CURRENT`,
`PII_KEY_LEGACY`, `PII_KEY_VERSION`), with full backward compatibility: when the new variables are absent the
historical single-key/untagged behaviour is byte-identical to before (CI and hermetic suites are unaffected, since
they do not set them). `identity_service/security.py` now delegates to the module, keeping its public API
(`encrypt_pii`, `decrypt_pii`, `encrypt_opt`, `decrypt_opt`).

Call sites re-encrypted by the migration: `users.full_name`, `users.phone_encrypted`, `users.email_encrypted`
(and `address_encrypted`, `account_number`, `account_type` when populated — all NULL on this host).

---

## 5. Migration design

* **Dry run by default.** `--apply` is required for any write; the dry run opens stores read-only.
* **Backup first.** Per store: a byte copy under `db/_pii_rotation_backups/` (gitignored), hashed (sha256)
  before any write, with `matches_source` recorded in the report. **Defect found after the live run (fixed in the
  tool):** the filename was derived from the store's *basename*, and all three stores are named `identity.db`, so
  the copies collided and only the last one written (the `wt-scenarios` fixture store) survived on disk
  (49152 B, sha256 `189ee94f…`). The surviving file is a valid pre-migration snapshot **of that store only**; the
  two live stores have no `.bak` file. The name now slugs the full relative path
  (`db__identity.db.pre-pii-rotation-<UTC>.bak`), and the primary rollback path below does not depend on a backup
  file at all.
* **In-memory verification.** Each row is decrypted with the legacy key, re-encrypted with the current key, and the
  new ciphertext is decrypted again and compared to the same in-memory plaintext **before** the UPDATE is committed.
  No plaintext or key material is ever written to disk, logged, or printed.
* **Quarantine, never overwrite.** A token that no available key can read is counted, listed by rowid and left
  byte-identical; it is never deleted and never replaced with new ciphertext.
* **Atomicity.** Rows are updated inside a single transaction; a failed commit rolls back, leaving the store on the
  legacy key (still readable). The tool returns a structured error instead of raising.
* **Post-commit verification** on a fresh read-only connection: same row and token counts, unchanged undecryptable
  set, and `needs_migration == undecryptable` (i.e. nothing readable is left on the old key).
* **Idempotent.** A second run migrates nothing and reports the rows as already current.
* **Retirement condition for the old key** (§6): remove `PII_KEY_LEGACY` **and** `PII_ENCRYPTION_KEY` once every
  store reports `needs_migration == 0` and the rollback artifact is no longer needed.

Tool: `backend/scripts/rotate_pii_key.py` (dry run by default; `--apply` to migrate, `--revert` for the documented
reverse migration). Tests: `backend/scripts/pii_key_rotation_test.py` (31 checks, hermetic, now part of the CI fast
regression list).

---

## 6. Old key handling

* The previous PII key material lives **only** in the gitignored `.env` as `PII_KEY_LEGACY` (plus the historical
  `PII_ENCRYPTION_KEY` entry it was copied from) and in the gitignored rollback copy.
* It is not hardcoded, not committed, not printed, not in logs, not in any report or evidence bundle (verified by
  the containment scan in §17).
* It is **read-only**: the tool never writes v1-ciphertext, and the live service writes v2 only.
* It has a documented retirement condition (§5) and a recorded fingerprint for audit
  (`v1: 4101a113988fad4c`, `v2: 248f5b3d3a0f00af` — sha256 prefixes of derived keys, not key material).

---

## 7. Dry run (before any write)

`backend/scripts/rotate_pii_key.py` with the rotated configuration (current `v2`, keys `v1`+`v2`):

| Metric | Value |
|---|---|
| Stores surveyed | 3 (`db/identity.db`, `db/walkthrough/identity.db`, `db/wt-scenarios/identity.db`) |
| Records | 835 |
| Fernet tokens discovered | 2500 |
| Tokens needing migration | 2500 |
| Already on the current key | 0 |
| Successfully decryptable | 2490 |
| Undecryptable (pre-existing) | **10** (rows 1–5 of the `wt-scenarios` fixture store) |
| Unknown key version | 0 |
| Migration attempted | 0 (dry run) |
| Exit code | 0 |

The 10 undecryptable tokens pre-date this phase: they sit in a **hermetic scenario fixture store** whose rows were
encrypted under a per-process random dev key (the documented `Settings` behaviour when no `.env` reaches the
process). They are readable by no key in this repository, were left untouched, and are quarantined in every later
run. They are not part of the live DB-1 and not covered by any evidence record.

---

## 8. Migration results (`--apply`)

| Store | Records | Tokens | Migrated | Quarantined | Failed | Verified | Pre-write copy |
|---|---|---|---|---|---|---|---|
| `db/identity.db` | 829 | 2487 | **2487** | 0 | 0 | ✅ | written, then overwritten by the next store's copy (filename collision, §5) |
| `db/walkthrough/identity.db` | 1 | 3 | **3** | 0 | 0 | ✅ | same |
| `db/wt-scenarios/identity.db` | 5 | 10 | 0 | **10** | 0 | ✅ | survives: `identity.db.pre-pii-rotation-20261006T045525Z.bak` (sha256 `189ee94f…`) |
| **Total** | 835 | 2500 | **2490** | 10 | **0** | 3/3 | 1 of 3 on disk |

Post-commit verification per store: identical row/token counts, unchanged undecryptable set, and
`needs_migration == undecryptable` (0 for the two live stores). Exit code 0, `problems: 0`.

Rollback strategy. The **primary** path is the tool's own reverse migration: `rotate_pii_key.py --revert`
(implemented and covered by test block 19) decrypts each v2 token with the current key and re-encrypts it with
`PII_KEY_LEGACY`, restoring the store to a state readable by a legacy-only configuration — no committed secret and
no hand-editing required. Because the old key material is still present in the gitignored `.env`
(`PII_KEY_LEGACY` + `PII_ENCRYPTION_KEY`), this works even though the file copies for the two live stores were lost
to the naming defect: the migration was verified in memory per row before commit, and a revert is a forward
transformation from verified plaintext, not a restore of unknown bytes. The surviving
`wt-scenarios` copy was kept only as a byte-level comparison; the fixture store was never modified (0 migrated).
No rollback was required in this phase.

---

## 9. Post-migration verification (live, 15/15 PASS)

`.freebuff/p4a_verify.json` — stack restarted with the rotated `.env`:

| ID | Check | Result |
|---|---|---|
| R1 | login works with the rotated JWT secret | 200 |
| R2 | authenticated PII read (`GET /me/profile`) decrypts under the **new** PII key, masked, no plaintext email | 200, masked |
| R2b | second authenticated route accepts the new token | 200 |
| R3 | token signed with the **old** JWT secret | **401** |
| R3b | same claims signed with the **new** secret (control) | 200 |
| R4 | **old** internal token on risk `/internal/evaluate` | **401** |
| R5 | **new** internal token on risk `/internal/evaluate` | 200 |
| R5b | **old** internal token on privacy `/internal/ingest-transaction` | **401** |
| R5c | **new** internal token on privacy ingest | 200 |
| R6 / R6b | admin passphrase login and an admin API call | 200 / 200 |
| R7 | **old PII key reads 0 of 2487** migrated rows | 0/2487 |
| R7b | new PII key reads **every** migrated row | 2487/2487 |

**Artifact-level re-check** (`.freebuff/p4a_db_verify.json`, read-only, run against the live stores after the final
code change): every PII token in every store was decrypt-tested with the current key and with a legacy-only cipher.

| Store | Rows | Tokens | `v2:`-tagged | Untagged legacy | Read by current key | Read by legacy key alone |
|---|---|---|---|---|---|---|
| `db/identity.db` | 865 | 2595 | **2595** | **0** | 2595 | 0 |
| `db/walkthrough/identity.db` | 1 | 3 | **3** | **0** | 3 | 0 |
| `db/wt-scenarios/identity.db` | 5 | 10 | 0 | 10 (quarantined) | 0 | 0 |

`db/identity.db` held 829 rows / 2487 tokens at migration time and 865 / 2595 at this re-check: the running stack
created new users in between, and **every one of the new tokens is already `v2:`-tagged** — independent proof that
live writes use the new key rather than the legacy one. Off these two live stores, the count of ciphertext
readable **only** by the retired key is **zero**.
| R8 | quarantined fixture rows byte-identical | 5 rows untouched |
| R9 | no tracebacks, crypto errors (`InvalidToken`, `Unhandled exception`) in six service logs | clean |

The authenticated PII read is the end-to-end proof: the identity service decrypted real DB-1 ciphertext with the
new key and returned masked fields, and the same route refused a token signed with the retired secret.

---

## 10. Supabase credential

* **Application configuration**: the value is read from `.env`/environment only; the literals were removed from
  `backend/scripts/smoke_test_supabase.py` in Phase 4 and the script now fails fast when the variables are absent.
* **Verification of dependency**: no component under `backend/src` reads it (the only references strip it from
  child environments), the local launcher removes `SUPABASE_*` before starting the stack, and the tracked-file
  containment scan finds no copy of the value in the repository.
* **Provider rotation**: **not performed.** Rotating or revoking a Supabase `service_role` key requires
  account-owner access through the Supabase dashboard/management API, and this phase does not act on external
  systems without the owner present. No workaround was attempted and no completion is claimed.
* **Interim state**: the credential is not committed (current tree), is not used by the running stack, and the only
  remaining exposure is the historical commit plus the local `.env`. The owner should rotate it in the Supabase
  dashboard and update `.env`; nothing else needs to change.

**`OWNER ACTION REQUIRED — PROVIDER CREDENTIAL ROTATION`**

---

## 11. Secret scanning and containment (§17)

| Scan | Result |
|---|---|
| `backend/scripts/secret_hygiene_test.py` (repo-wide) | **16/16 PASS** — no live `.env` value in any tracked file; no credential-shaped literal outside self-declared detector files; remediated scripts hold no literals; Bandit medium+ gate exits 0 |
| Targeted search for the **old and new** values (16 keys, tracked files) | **NONE** (the only hit is the non-secret `CORS_ORIGINS` list) |
| Bandit (`-r backend/src --severity-level medium`, the CI command) | exit 0 |
| `backend/scripts/security_scan.py` / CI security job | unchanged; the CI job additionally runs trufflehog `--only-verified` (which cannot verify a Supabase service key) |
| Worktree scan (tracked + untracked, excluding `.git`/`.venv`) | no secret value outside `.env` and the gitignored rollback artifact |
| Rollback artifact | `db/_pii_rotation_backups/env.p4a-rollback-*.bak` — inside the gitignored `db/` tree (`git check-ignore` verified); contains the three previous values for rollback only, delete after verification |

A scanner returning zero findings does not prove the historical exposure never happened; the historical commits
remain part of the project's security history.

**Exposure classification (§16/§24 of the phase brief).** Stated with the brief's own labels so that current and
historical exposure can never be confused:

| Classification | Applies to |
|---|---|
| `CURRENT SECRET EXPOSURE` | **none** — every secret active today exists only in the gitignored `.env` (or the deployment secret store); nothing in the worktree is a live value |
| `HISTORICAL SECRET EXPOSURE` | the previously committed JWT signing secret and internal-token literal, and the Supabase `service_role` key + project URL; all still recoverable from git history, which was deliberately **not** rewritten and is not claimed to be erased |
| `HISTORICALLY EXPOSED — ROTATED` | JWT signing secret, internal service token — the values that were actually committed and are now replaced and demonstrably rejected |
| `HISTORICALLY EXPOSED — OWNER ACTION PENDING` | Supabase `service_role` key + project URL — source-remediated and unused locally, but the provider credential itself is not yet rolled |
| `NOT EXPOSED — ROTATED BY DESIGN` | PII data key material — never committed; rotated onto a dedicated versioned key because the phase requires independent rotation, not because it leaked |

---

## 12. Regression and failure testing (§18/§19)

`backend/scripts/pii_key_rotation_test.py` — **31 checks, 0 failures** (hermetic: own key material, temp stores):

* round-trip encryption against the current key; legacy token readable during rotation; migration + read-back;
  duplicate migration idempotent; corrupted ciphertext rejected; unknown key version refused; missing current key
  fails closed; legacy token unreadable without the legacy key; old key cannot read v2 ciphertext;
* no plaintext in logs, no key material in logs, errors or the migration report;
* quarantined/corrupt rows left byte-identical while good rows migrate around them;
* write failure (blocked UPDATE) rolls back and leaves every row readable on the old key;
* JWT rotation does not break PII decryption; PII rotation does not break JWT authentication; the retired JWT
  secret no longer validates a current token;
* reverse migration (the documented rollback path): rows moved back onto the legacy key, readable by a
  legacy-only configuration and **not** readable by the new key alone.

The suite caught two real defects in the migration tool during development (tagged tokens were not recognised as
ciphertext; the tag was not stripped before decryption) — fixed before the live migration ran — and, after the live
run, the backup-filename collision described in §5, whose fix is covered by the reverse-migration block.

One further defect surfaced by the repository battery: the supply-chain suite flagged the new hygiene test for
containing detector patterns. Cause fixed by extending that suite's existing, documented exclusion list of
detector/fixture files (the rule and its patterns are unchanged) — the supply-chain suite then reports
117/117.

---

## 13. Verification battery and CI (§23)

| Check | Result |
|---|---|
| Repository battery (`.freebuff/p114_battery.sh`) | first run: 1 FAIL (`phase78_supply_chain_security_test`, cause fixed above) → **final run: 84/84 PASS, 0 FAIL** |
| Fast regression list (`regression_suite.py --fast`, includes the new PII rotation suite) | PASS (see closeout) |
| `claim_evidence_check` | PASS (22 claims) |
| `eval_record_test` | PASS |
| `review_resolution_check` | PASS |
| `review_package_check` | PASS (13/13 sections) |
| `check_freeze.py` | **rc 1 / 78 failures — expected by design**, unchanged by this phase |
| Bandit | exit 0 |
| Live post-rotation verification | 15/15 PASS |
| CI on the final commit (`28d0935`) | **both workflows green** — CI/CD run `37420876940` (5/5 jobs, including the secret scanner and the Bandit gate) and Security Scan run `37420876927`; detail in the closeout |

Scientific artifacts untouched: no change to the Research Plan, preregistration, review decisions, datasets,
benchmark, model artifacts, thresholds, or the fraud-detection methodology.

---

## 14. Unresolved items

| Item | State |
|---|---|
| Supabase `service_role` provider rotation | **Owner action required** (account ownership); no local dependency remains |
| 10 quarantined fixture tokens (`db/wt-scenarios/identity.db`) | Unreadable by any key in this repository; pre-existing; left untouched. If that fixture store is ever needed, regenerate it rather than attempting recovery |
| Pre-write `.bak` copies for the two live stores | Lost to the backup-filename collision (§5); the defect is fixed in the tool and the copies are not the rollback path — the documented rollback is `rotate_pii_key.py --revert` (reverse migration, test block 19), possible because the legacy key is still in `.env` |
| Retirement of `PII_KEY_LEGACY` / `PII_ENCRYPTION_KEY` and the rollback artifact | Pending by design: remove both after the rollback window closes and every store reports `needs_migration == 0` |
| Git history | Contains the two historically exposed values; not rewritten (separate operational decision, out of scope) |
| Processes that import settings without `.env` | Pre-existing dev behaviour: `Settings` generates a random per-process PII key when no environment is present. The launcher supplies `.env`; scripts must call `load_dotenv_and_patch()` (the numeric probe in this report briefly hit exactly this trap and it is now also mitigated by a cipher-cache reset in that loader) |

## 15. Limitations

1. Rotation covers the **local/staging** stack only; the optional Supabase path is not exercised here.
2. The Supabase credential is the one item this phase cannot complete alone.
3. "Old credential invalid" is proven for the three local credentials by live probes; the provider credential is
   proven only to be *unused locally*, not invalid.
4. The migration's integrity check is in-memory equality of plaintext, not an external attestation.
5. Assertions about PII are counts and status codes; no plaintext PII was read out during verification.
6. The pre-migration byte copies of the two live stores were overwritten by the filename defect (§5). Rollback rests
   on the per-row in-memory decrypt/re-encrypt equality checked before each commit plus the tested `--revert` path,
   not on those files.

---

## 16. Final decision

> ## `REQUIRES OWNER ACTION`
>
> `OWNER ACTION REQUIRED — PROVIDER CREDENTIAL ROTATION`

With that single exception, the phase's stop conditions are met: the JWT↔PII coupling is documented and
disproved, the versioned PII key works, the dry run succeeded, PII was migrated with verification and rollback in
place, JWT and internal-token rotations are complete and their predecessors demonstrably rejected, the secret
scans pass, the regression and PII tests pass, and the batch verification is green. The safe interim state for the
Supabase credential is recorded in §10.
