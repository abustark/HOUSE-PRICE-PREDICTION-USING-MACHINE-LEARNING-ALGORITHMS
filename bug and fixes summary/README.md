# bug and fixes summary — paste-kit for the Bug & Fix Registry protocol

> **Purpose:** a self-contained kit you can hand to any other project (or its AI agent) so it adopts
> the same bug-tracking discipline as this repo's `BUGS-AND-FIXES.md`.
>
> **Source of truth:** ABot repo — root `BUGS-AND-FIXES.md` (the live registry, 46 entries) + the
> `§Re-verification Checklist` inside it. Extracted as a protocol at v2.1.2 (commit `090bdaf`-era docs).

## Files

| File | What it is |
|---|---|
| `BUG-FIX-PROTOCOL.md` | **The reference description** — the rules, the entry format, the re-verification policy, why it works. Read/share this one. |
| `BUGS-AND-FIXES.template.md` | **Copy-paste starter** — drop it into another repo's root, rename to `BUGS-AND-FIXES.md`, and start logging. |

## How to use (2 steps)

1. Read `BUG-FIX-PROTOCOL.md` — it's the convention: one row per bug (ID, description, file + approx
   fix lines, fix description, version + commit), maintenance rules, and the mandatory
   re-verification policy.
2. Copy `BUGS-AND-FIXES.template.md` → `<other-repo>/BUGS-AND-FIXES.md`, fill in the example rows,
   and wire its re-verification checklist to that repo's own gate commands.

## The one-line summary of the protocol

> Every confirmed bug gets a row: **what it was · which file + line region fixed it · how ·
> which version + commit shipped the fix** — and after every update, you spot-check those fix
> lines again. Regressions become NEW entries; history is never edited.
