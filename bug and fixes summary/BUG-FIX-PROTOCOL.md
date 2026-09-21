# Bug & Fix Registry Protocol — the reference description

> Adopt this in any project by copying `BUGS-AND-FIXES.template.md` to the repo root as
> `BUGS-AND-FIXES.md` and following the rules below. This is the convention ABot's registry
> (46 entries) runs on; it has repeatedly caught shipped-but-inert fixes and silent regressions.

---

## 1. What the file is

One living markdown document at the repo root: **`BUGS-AND-FIXES.md`**. It is the memory of every
confirmed bug — so that (a) nothing gets fixed twice, (b) a regression is recognized instantly,
and (c) any agent/maintainer can audit "is every past fix still actually live?" in minutes.

## 2. The entry format — one table row per bug

```
| # | Bug | File(s) + approx Fix Lines | Fix | Version / Commit |
|---|---|---|---|---|
```

| Column | Rule |
|---|---|
| **#** | Stable ID (`B-01`, `B-02`, …). **Never reused, never renumbered.** A regression of an old bug gets a **new** B-id that links back to the original. |
| **Bug** | **Bold symptom** first (what the user sees), then the root cause once known. Quote the owner's words when they sharpen the repro. |
| **File(s) + approx Fix Lines** | Where the fix lives **now** — file path + approximate line range (e.g. `` `js/chat.js` `sendMessage` head (~L543–620) ``). Line numbers drift; mark them approximate and refresh whenever you touch that region. |
| **Fix** | The **mechanism**, not just the file (e.g. "guard set before any await; `isStreaming` first-line", not "fixed chat.js"). |
| **Version / Commit** | The release version + commit hash that shipped the fix (`2.1.0 · e019a9b`). This is what lets anyone diff exactly the fix later. |

## 3. Maintenance rules

1. **Every** confirmed bug gets an entry — symptom, file + approx line range, fix, version + commit.
2. **Regressions are new entries**, marked `REGRESSION`, naming the entry they regressed — they force
   the question "what keeps killing this fix?" instead of another blind patch.
3. **History is append-only.** Never edit or delete an entry once shipped; if it was wrong, add a
   correcting entry. (Entries for "not a bug" or "owner action required" are allowed — mark them
   clearly; they stop future re-investigation.)
4. **Guard the fix with a test where possible.** Any fix that can regress silently gets an automated
   check that names the B-id in a comment. Tests are the proof the row's fix is still live — without
   them the registry is just a diary.
5. **The header carries the rules.** The file starts with its own purpose + maintenance rules so the
   protocol travels with the file.

## 4. The re-verification policy (the point of the file)

After **every** major or intermediate update ships:

1. Run the project's automated gate suite (build, lint, tests — whatever that repo has).
2. **Spot-check the Fix Lines of every entry** — grep/read the region and confirm the fix code is
   still present and wired. (This is where ABot caught "fixed in the registry, absent in the shipped
   bundle" cases — e.g. a rebuilt-but-uncommitted CSS file made two shipped fixes inert.)
3. If a symptom reappears → its fix region regressed → diff that region against the fix commit and
   restore it; then log a new `REGRESSION` entry.
4. Multi-branch repos: keep one section per branch's own history (`§Arena Branch`, `§Main Repo`) and
   note that branches have independent version numbers — compare commits by hash, never by version.

## 5. The re-verification checklist section

Keep a ready-to-run checklist at the bottom of the registry: the automated pass (commands) plus a
short manual spot-check list (real browser, both themes/viewports) where **each item names the B-ids
it protects** — so the checklist is self-explaining and audit-able.

## 6. Why it works

- **File + line region** turns regression review from archaeology into a 5-minute diff.
- **Version + commit ref** makes "show me exactly what fixed it" one command away.
- **New-entry-for-regression** kills the fix-revert-fix loop: it forces root-cause work.
- **Test references to B-ids** make the registry machine-checkable instead of faith-based.
- **The file documents itself** — paste it anywhere and the protocol is on page one.
