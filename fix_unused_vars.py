#!/usr/bin/env python3
"""
Script to automatically fix unused variable violations in browser-use package.
This script adds underscores to unused variables as prefixes.
"""

import re
import subprocess


def run_ruff_check():
    """Run ruff to get list of unused variable violations."""
    try:
        result = subprocess.run(
            ["ruff", "check", "--select", "F841", ".", "--output-format", "json"],
            capture_output=True,
            text=True,
            cwd="/Users/jeremyparker/Desktop/Claude Coding Projects/AIgent/browser-use",
        )

        if result.returncode == 0:
            return []  # No violations

        # Parse the JSON output
        import json

        try:
            violations_data = json.loads(result.stdout)
            violations = [v for v in violations_data if v.get("code") == "F841"]
            return violations
        except json.JSONDecodeError:
            print("Error parsing JSON output from ruff")
            return []

    except Exception as e:
        print(f"Error running ruff: {e}")
        return []


def fix_unused_variable(file_path, line_num, variable_name):
    """Fix a single unused variable by adding underscore prefix."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if line_num > len(lines):
            return False

        line = lines[line_num - 1]  # Convert to 0-based indexing

        # Common patterns to fix
        patterns = [
            # except Exception as e:
            (
                rf"except\s+\w+\s+as\s+{re.escape(variable_name)}:",
                lambda m: m.group(0).replace(
                    f" {variable_name}:", f" _{variable_name}:"
                ),
            ),
            # variable = ...
            (
                rf"\b{re.escape(variable_name)}\s*=\s*",
                lambda m: m.group(0).replace(
                    f"{variable_name} =", f"_{variable_name} ="
                ),
            ),
        ]

        modified = False
        for pattern, replacement in patterns:
            if re.search(pattern, line):
                new_line = re.sub(pattern, replacement, line)
                if new_line != line:
                    lines[line_num - 1] = new_line
                    modified = True
                    break

        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
            return True

    except Exception as e:
        print(f"Error fixing {file_path}:{line_num}: {e}")

    return False


def main():
    """Main function to fix all unused variables."""
    print("Running ruff to find unused variable violations...")

    violations = run_ruff_check()
    print(f"Found {len(violations)} F841 violations")

    fixed_count = 0
    for violation in violations:
        file_path = violation.get("filename")
        line_num = violation.get("location", {}).get("row")
        message = violation.get("message", "")

        if not file_path or not line_num:
            continue

        # Extract variable name from message like "Local variable `e` is assigned to but never used"
        match = re.search(
            r"Local variable `([^`]+)` is assigned to but never used", message
        )
        if match:
            variable_name = match.group(1)
            if fix_unused_variable(file_path, line_num, variable_name):
                print(f"Fixed {file_path}:{line_num} - {variable_name}")
                fixed_count += 1
            else:
                print(f"Could not fix {file_path}:{line_num} - {variable_name}")

    print(f"Fixed {fixed_count} violations")

    # Run ruff again to see remaining count
    print("\nChecking remaining violations...")
    remaining = run_ruff_check()
    print(f"Remaining F841 violations: {len(remaining)}")


if __name__ == "__main__":
    main()
