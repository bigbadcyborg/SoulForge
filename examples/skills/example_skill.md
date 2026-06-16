---
name: example_skill
description: Example troubleshooting workflow skill.
tags:
  - example
---

## Trigger
Use when the user asks for help diagnosing startup issues in SoulForge.

## Procedure
1. Ask for the exact error message and startup command used.
2. Check config paths (`chatModelPath`, `embeddingModelPath`) and feature toggles.
3. Recommend running `/health` and `/diagnostics` for actionable checks.
4. Suggest a minimal fix and one verification command.

## Validation
The user can re-run startup and confirms the error is resolved or changed.
