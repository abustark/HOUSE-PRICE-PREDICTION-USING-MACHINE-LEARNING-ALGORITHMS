# <Project Name> — Bugs & Fixes Registry

> **Purpose:** the living record of every confirmed bug, where it lived (file + approximate line
> range), what fixed it, and in which commit. Used for regression review after every update
> (see §Re-verification Policy).
>
> **Maintenance rules:**
> 1. **Every** confirmed bug gets an entry — with file, approx line range (guess if code moved),
>    fix summary, and the version + commit ref that fixed it.
> 2. **Re-verification policy (MANDATORY):** whenever a new major or intermediate update ships,
>    re-run the checks in §Re-verification Checklist and spot-check the Fix Lines of every entry
>    below. If a bug's symptom reappears, its Fix Line region regressed — diff that region against
>    the fix commit and restore.
> 3. Regressions are **new entries** (marked REGRESSION, linked to the original). History is
>    append-only — never edit or delete a shipped entry.
> 4. *(Optional)* If you use multiple branches, keep one section per branch — branches have
>    independent version numbers; compare commits by hash, never by version.
>
> Protocol reference: see `BUG-FIX-PROTOCOL.md` (or: ABot repo → `bug and fixes summary/`).

---

## §Bug & Fix History

| # | Bug | File(s) + approx Fix Lines | Fix | Version / Commit |
|---|---|---|---|---|
| B-01 | **<short symptom the user saw>** — <root cause once known> | `<file>` `<function>` (~L<a>–<b>) | <the fix mechanism, not just the file> | <version> · `<commit hash>` |
| B-02 | **EXAMPLE (delete me):** "Reply hidden behind composer on mobile" — fixed-height padding vs a dock whose height grows with the keyboard | `js/app.js` `syncScrollPadding` (~L374–388) | Padding synced to the dock's measured rect via ResizeObserver; floor of 170px; re-synced on keyboard open/close | 2.0.21 · `1d95ba5` |
| B-03 | **EXAMPLE REGRESSION (delete me):** B-02 returned after v2.0.26 — cause: `src/input.css` edited but `dist/output.css` never rebuilt, so the rule never shipped | `dist/output.css` (committed rebuild) | Rebuilt + committed the CSS; CI now fails if the bundle doesn't match the rebuild | 2.0.27 · `87a04d4` |

<!-- New bugs append BELOW. If a fixed bug's symptom returns, add a NEW entry marked
     REGRESSION (link the old one) — never edit history. Guard each fix with an automated
     test that names its B-id in a comment wherever a silent regression is possible. -->

---

## §Re-verification Checklist (run after EVERY major/intermediate update)

**Automated pass (replace with this repo's own commands):**
```bash
npm run lint && npm run test
npm run build            # if the repo builds assets, rebuild them
git diff --exit-code -- <built-assets>   # fail if a build product is stale
```

**Manual spot-checks (5 min, real browser, both themes/viewports) — each item names the entries it protects:**
1. <User-facing check 1, e.g. "reload → build id matches the new deployment"> (B-xx)
2. <User-facing check 2, e.g. "sign out → sign back in → toggle still single-fires"> (B-xx)
3. <User-facing check 3, e.g. "long reply on mobile: nothing hidden behind the composer"> (B-xx)

---

*Started <YYYY-MM-DD>. Convention: `bug and fixes summary/BUG-FIX-PROTOCOL.md`.*
