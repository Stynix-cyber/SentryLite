from __future__ import annotations

import ast
from dataclasses import dataclass

from src.dataflow import analyze_dataflow


@dataclass(frozen=True)
class CodeFeatures:
    function_count: int
    call_count: int
    branch_count: int
    loop_count: int
    import_count: int

    input_source_count: int
    dynamic_execution_count: int
    subprocess_count: int
    shell_true_count: int
    deserialization_count: int

    tainted_flow_count: int
    tainted_eval_count: int
    tainted_exec_count: int
    tainted_subprocess_count: int
    tainted_deserialization_count: int

    def as_vector(self) -> list[float]:
        return [
            float(self.function_count),
            float(self.call_count),
            float(self.branch_count),
            float(self.loop_count),
            float(self.import_count),
            float(self.input_source_count),
            float(self.dynamic_execution_count),
            float(self.subprocess_count),
            float(self.shell_true_count),
            float(self.deserialization_count),
            float(self.tainted_flow_count),
            float(self.tainted_eval_count),
            float(self.tainted_exec_count),
            float(self.tainted_subprocess_count),
            float(self.tainted_deserialization_count),
        ]


class FeatureExtractor(ast.NodeVisitor):

    SUBPROCESS_FUNCTIONS = {
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
    }

    def __init__(self) -> None:
        self.function_count = 0
        self.call_count = 0
        self.branch_count = 0
        self.loop_count = 0
        self.import_count = 0

        self.input_source_count = 0
        self.dynamic_execution_count = 0
        self.subprocess_count = 0
        self.shell_true_count = 0
        self.deserialization_count = 0

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:
        self.function_count += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self.function_count += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.branch_count += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loop_count += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.loop_count += 1
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.import_count += len(node.names)
        self.generic_visit(node)

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:
        self.import_count += len(node.names)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.call_count += 1

        name = self._call_name(node.func)

        if name in {
            "input",
            "sys.stdin.readline",
        }:
            self.input_source_count += 1

        if name in {
            "eval",
            "exec",
        }:
            self.dynamic_execution_count += 1

        if name in self.SUBPROCESS_FUNCTIONS:
            self.subprocess_count += 1

            if self._has_shell_true(node):
                self.shell_true_count += 1

        if name in {
            "pickle.load",
            "pickle.loads",
        }:
            self.deserialization_count += 1

        self.generic_visit(node)

    @staticmethod
    def _has_shell_true(node: ast.Call) -> bool:
        for keyword in node.keywords:
            if keyword.arg != "shell":
                continue

            if (
                isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                return True

        return False

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        parts: list[str] = []

        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            parts.append(node.id)

        return ".".join(reversed(parts))


def extract_features(source: str) -> CodeFeatures:
    tree = ast.parse(source)

    extractor = FeatureExtractor()
    extractor.visit(tree)

    flows = analyze_dataflow(source)

    tainted_eval = 0
    tainted_exec = 0
    tainted_subprocess = 0
    tainted_deserialization = 0

    for flow in flows:

        if flow.sink == "eval":
            tainted_eval += 1

        elif flow.sink == "exec":
            tainted_exec += 1

        elif flow.sink.startswith("subprocess."):
            tainted_subprocess += 1

        elif flow.sink in {
            "pickle.load",
            "pickle.loads",
        }:
            tainted_deserialization += 1

    return CodeFeatures(
        function_count=extractor.function_count,
        call_count=extractor.call_count,
        branch_count=extractor.branch_count,
        loop_count=extractor.loop_count,
        import_count=extractor.import_count,

        input_source_count=extractor.input_source_count,
        dynamic_execution_count=extractor.dynamic_execution_count,
        subprocess_count=extractor.subprocess_count,
        shell_true_count=extractor.shell_true_count,
        deserialization_count=extractor.deserialization_count,

        tainted_flow_count=len(flows),
        tainted_eval_count=tainted_eval,
        tainted_exec_count=tainted_exec,
        tainted_subprocess_count=tainted_subprocess,
        tainted_deserialization_count=tainted_deserialization,
    )