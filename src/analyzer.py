from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Finding:
    line: int
    severity: str
    category: str
    message: str
    confidence: float


class SecurityAnalyzer(ast.NodeVisitor):
    """
    Lightweight static Python security analyzer.

    V0.1 intentionally performs no code execution.
    Source files are parsed into Python's AST and inspected statically.
    """

    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def add_finding(
        self,
        node: ast.AST,
        severity: str,
        category: str,
        message: str,
        confidence: float,
    ) -> None:
        self.findings.append(
            Finding(
                line=getattr(node, "lineno", 0),
                severity=severity,
                category=category,
                message=message,
                confidence=confidence,
            )
        )

    def visit_Call(self, node: ast.Call) -> None:
        function_name = self._get_call_name(node.func)

        checks = {
            "eval": (
                "HIGH",
                "UNSAFE_EXECUTION",
                "Use of eval() can execute dynamically supplied Python code.",
                0.96,
            ),
            "exec": (
                "HIGH",
                "UNSAFE_EXECUTION",
                "Use of exec() can execute dynamically supplied Python code.",
                0.96,
            ),
            "pickle.loads": (
                "HIGH",
                "UNSAFE_DESERIALIZATION",
                "Deserializing untrusted pickle data can execute code.",
                0.95,
            ),
            "pickle.load": (
                "HIGH",
                "UNSAFE_DESERIALIZATION",
                "Loading untrusted pickle data can execute code.",
                0.95,
            ),
        }

        finding = checks.get(function_name)

        if finding:
            severity, category, message, confidence = finding
            self.add_finding(
                node,
                severity,
                category,
                message,
                confidence,
            )

        self._check_subprocess(node, function_name)

        self.generic_visit(node)

    def _check_subprocess(
        self,
        node: ast.Call,
        function_name: str,
    ) -> None:
        subprocess_functions = {
            "subprocess.run",
            "subprocess.call",
            "subprocess.Popen",
            "subprocess.check_call",
            "subprocess.check_output",
        }

        if function_name not in subprocess_functions:
            return

        for keyword in node.keywords:
            if (
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                self.add_finding(
                    node,
                    "MEDIUM",
                    "SHELL_EXECUTION",
                    "Subprocess is executed with shell=True. "
                    "Review whether untrusted input can reach this call.",
                    0.88,
                )

    @staticmethod
    def _get_call_name(node: ast.AST) -> str:
        parts: list[str] = []

        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            parts.append(node.id)

        return ".".join(reversed(parts))


def analyze_source(source: str) -> list[Finding]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(
            f"Cannot parse Python source: {exc}"
        ) from exc

    analyzer = SecurityAnalyzer()
    analyzer.visit(tree)

    return sorted(
        analyzer.findings,
        key=lambda finding: finding.line,
    )


def analyze_file(path: Path) -> list[Finding]:
    if path.suffix.lower() != ".py":
        raise ValueError("V0.1 only supports Python files.")

    source = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    return analyze_source(source)


def print_report(path: Path, findings: list[Finding]) -> None:
    print()
    print("SentryLite v0.1")
    print("=" * 50)
    print(f"File: {path}")
    print()

    if not findings:
        print("No findings detected.")
        return

    for finding in findings:
        print(
            f"[{finding.severity}] "
            f"Line {finding.line} | "
            f"{finding.category}"
        )
        print(f"  {finding.message}")
        print(f"  Confidence: {finding.confidence:.0%}")
        print()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python src/analyzer.py <file.py>")
        return 1

    path = Path(sys.argv[1])

    if not path.is_file():
        print(f"File not found: {path}")
        return 1

    try:
        findings = analyze_file(path)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print_report(path, findings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())