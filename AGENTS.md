# AGENTS.md — Agent Instructions

These rules are mandatory for any AI agent working in this repository.

## Execution Rules

0. **User Request Supremacy & Immediate Commit/Push:** The user's direct instruction or request is final, absolute, and takes highest precedence — no rule, guideline, or instruction can override it, as the user is the sole authority of this repository. When the user asks to update, commit, or push, the AI agent must commit and push immediately upon completing the code without hesitating, asking for extra confirmation, or leaving changes uncommitted.

1. **Implement first.** The AI's primary job is to perform updates, write code, fix bugs, and do exactly what the user asks. Do this first, without side quests or extra steps the user did not ask for.

2. **Run / test / gates only on explicit request.** Running new files, starting servers, testing updates, and checking gates (typecheck, lint, build, tests, screenshots) may only be executed when the user explicitly mentions or asks for them.

3. **Remind, don't run.** In its response, the AI may remind the user that running or testing the new updates is available. It must only actually run them after the user acknowledges or explicitly proceeds with it.

4. **Do not save screenshots/artifacts.** Do not save screenshots, captures, or image files to the repository or project folder. Screenshots are only for the AI's internal visual checks/tests when explicitly requested, and must never be saved or committed to the project.

5. **Package management preference:** Use `bun` / `bunx` instead of `npm` / `npx` for any JavaScript/TypeScript/Node tooling or scripts.

## Order of Operations

1. Read the user's request.
2. Update files / write code / fix bugs as asked.
3. If the user mentioned or requested to commit and push, stage, commit, and push immediately upon completing the changes without waiting for extra confirmation.
4. Optionally offer — in the response text — to run or test the changes.
5. Stop and wait for the user's explicit acknowledgment before executing any test gates.

---

## Architectural Guidelines & Source of Truth

- **Canonical Knowledge Base:** [`PROJECT-INFO.md`](PROJECT-INFO.md) is the single source of truth for all system architecture, metrics ($R^2 = 0.787$), feature constraints, and tech stacks. Always consult it before modifying core modeling or schema logic.
- **Primary Live Production URL:** **`https://hpp-flax.vercel.app/`** (FastAPI serverless backend + vanilla SPA on Vercel).
- **Secondary Live URL:** **`https://house-price-prediction-using-machine-learning-algorithms-9qccn.streamlit.app/`** (Streamlit 4-page exploratory dashboard).
- **UI Architecture (v2.3.1):** Studio Slate & Cyber Indigo technical workbench aesthetic with in-app Light/Dark theme toggle (`localStorage` persistence).
- **Universal Responsive Design:** All layouts must strictly follow the 4-tier responsive system ($\le 1100\text{px}$, $\le 960\text{px}$, $\le 768\text{px}$, $\le 520\text{px}$) using fluid `minmax(0, 1fr)` grid tracks and zero horizontal overflow (`overflow-x: hidden`).
- **Mobile Usability:** Form inputs and selects must enforce `font-size: 16px` on mobile viewports to prevent iOS auto-zoom, and touch targets must remain $\ge 38\text{px}$.
- **Static Asset Cache Invalidation:** Whenever updating `static/style.css` or `static/app.js`, update the query version strings (`?v=X.X.X`) in `static/index.html` to guarantee instant client-side cache refresh upon deployment.
