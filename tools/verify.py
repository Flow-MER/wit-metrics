from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}


def make_verify_env(project_root: Path) -> tuple[dict[str, str], Path]:
    """Create an isolated temp folder and env for verification subprocesses."""
    tmp_root = project_root / ".tmp"
    tmp_root.mkdir(exist_ok=True)

    run_tmp = Path(
        tempfile.mkdtemp(
            prefix="verify-",
            dir=tmp_root,
        )
    )

    env = os.environ.copy()
    env["TMP"] = str(run_tmp)
    env["TEMP"] = str(run_tmp)
    env["TMPDIR"] = str(run_tmp)

    return env, run_tmp


def iter_python_files(project_root: Path):
    """Yield Python files, excluding common generated/cache folders."""
    for path in project_root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def run_syntax_check(project_root: Path) -> bool:
    """Check Python syntax without writing .pyc files."""
    print("\n" + "=" * 70)
    print("Python Syntax Check")
    print("=" * 70)

    errors = []
    checked = 0

    for path in iter_python_files(project_root):
        checked += 1

        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")

        except SyntaxError as ex:
            errors.append((path, ex))

        except UnicodeDecodeError as ex:
            errors.append((path, ex))

    print(f"Files checked: {checked}")

    if errors:
        print(f"Syntax errors: {len(errors)}")

        for path, ex in errors:
            print(f"\n{path}")
            print(ex)

        return False

    print("Syntax check passed.")
    return True


def run_step(name: str, command: list[str], cwd: Path, env: dict[str, str]) -> bool:
    """Run an external verification command."""
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(" ".join(command))

    result = subprocess.run(command, cwd=cwd, env=env)
    if name == "Pytest":
        return result.returncode in {0, 5}
    return result.returncode == 0


def main() -> int:
    """Run project verification."""
    project_root = Path(__file__).resolve().parent.parent

    env, run_tmp = make_verify_env(project_root)

    test_folder = project_root / "tests"
    test_folder.mkdir(exist_ok=True)

    failures = 0
    try:
        if not run_syntax_check(project_root):
            failures += 1

        if not run_step("Ruff", ["ruff", "check", "--no-cache", "."], project_root, env):
            failures += 1

        if not run_step(
            "Pytest",
            [sys.executable, "-m", "pytest", "-s", "-p", "no:cacheprovider"],
            project_root,
            env,
        ):
            failures += 1
    finally:
        shutil.rmtree(run_tmp, ignore_errors=True)

    print("\n" + "=" * 70)

    if failures:
        print(f"VERIFICATION FAILED ({failures} failing step(s))")
        return 1

    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
