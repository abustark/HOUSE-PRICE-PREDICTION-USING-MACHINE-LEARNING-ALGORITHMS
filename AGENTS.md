# AGENTS.md — Agent Instructions

These rules are mandatory for any AI agent working in this repository.

## Execution Rules

1. **Implement first.** The AI's primary job is to perform updates, write code, fix bugs, and do exactly what the user asks. Do this first, without side quests or extra steps the user did not ask for.

2. **Run / test / gates only on explicit request.** Running new files, starting servers, testing updates, and checking gates (typecheck, lint, build, tests, screenshots) may only be executed when the user explicitly mentions or asks for them.

3. **Remind, don't run.** In its response, the AI may remind the user that running or testing the new updates is available. It must only actually run them after the user acknowledges or explicitly proceeds with it.

4. **Do not save screenshots/artifacts.** Do not save screenshots, captures, or image files to the repository or project folder. Screenshots are only for the AI's internal visual checks/tests when explicitly requested, and must never be saved or committed to the project.

5. **Package management preference:** Use `bun` / `bunx` instead of `npm` / `npx` for any JavaScript/TypeScript/Node tooling or scripts.

## Order of Operations

1. Read the user's request.
2. Update files / write code / fix bugs as asked.
3. Optionally offer — in the response text — to run or test the changes.
4. Stop and wait for the user's acknowledgment before running anything.
