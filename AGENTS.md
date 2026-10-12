# Agent Execution Policy

## Prime Directive

Do not proceed with work when there are unanswered questions.

If you have questions, stop as soon as you have them and ask before proceeding. Do not guess, do not fill in missing requirements silently, and do not continue implementation while a decision is unclear.

## Testing

Use test-driven development.

- Use pytest. Keep tests focused on one behavior each. Use fixtures to reduce setup duplication.
- Write the test first and confirm it FAILS before writing implementation code.
- Derive expected behavior from the spec/requirements, not from reading or running the implementation
- Write tests using only the public interface. Treat internal helper methods, private state, and intermediate variables as invisible.
- State & Result Verification First: Prefer assert outcome == expected over verifying that specific internal functions were executed.
- Restrict Mocking to System Boundaries: Only mock external network calls, file system I/O, or non-deterministic inputs (like dates/randomness). Do not mock internal classes, utility functions, or peer domain objects.
- Sanity check: the test must fail against a stub implementation (`pass` / `return None`). If it wouldn't, rewrite it.

## File Operations

- Write all temporary/patch/debug files to `.tmp/` only
- Before completing any task: check `.tmp/` for files created this session and delete them

## Code Verification (required after changing code)

Run in order:

1. `ruff check . --fix`
2. `ruff format .`
3. `python tools/verify.py`

Task is not complete until `verify.py` passes.

### Ruff Error B905 (zip() missing strict=True)

Do NOT auto-apply `strict=True`. First confirm the zipped iterables are guaranteed equal length, then add it.

## Completion Report

Reply with a structured block:

- **Files modified:**  list of files
- **Ruff:** pass/fail + any warnings
- **verify.py:** pass/fail + output
