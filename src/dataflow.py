from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class DataFlowFinding:
    source_line: int
    sink_line: int
    variable: str
    source: str
    sink: str


@dataclass(frozen=True)
class FunctionSummary:
    """
    Compact description of how tainted data can flow through a function.

    source_return:
        The function itself creates tainted data and returns it.

    parameter_to_return:
        Indices of parameters whose values can flow into the return value.
    """

    name: str
    source_return: bool
    source_name: str | None
    source_line: int
    parameter_to_return: frozenset[int]


@dataclass(frozen=True)
class Taint:
    source: str
    source_line: int


class FunctionSummaryBuilder:
    """
    Builds small interprocedural summaries before the main analysis.
    """

    SOURCES = {
        "input",
        "sys.stdin.readline",
    }

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree
        self.summaries: dict[str, FunctionSummary] = {}

    def build(self) -> dict[str, FunctionSummary]:
        functions = [
            node
            for node in ast.walk(self.tree)
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            )
        ]

        # Multiple passes allow summaries to depend on summaries
        # discovered during earlier passes.
        for _ in range(max(1, len(functions) + 1)):
            changed = False

            for function in functions:
                summary = self._summarize(function)

                if self.summaries.get(function.name) != summary:
                    self.summaries[function.name] = summary
                    changed = True

            if not changed:
                break

        return self.summaries

    def _summarize(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> FunctionSummary:

        parameters = [
            argument.arg
            for argument in function.args.args
        ]

        # Variable -> set of parameter indices influencing it.
        parameter_taint: dict[str, set[int]] = {
            name: {index}
            for index, name in enumerate(parameters)
        }

        # Variable -> direct source information.
        source_taint: dict[str, Taint] = {}

        source_return = False
        source_name: str | None = None
        source_line = 0

        parameter_to_return: set[int] = set()

        for node in ast.walk(function):

            if isinstance(node, ast.Assign):
                parameter_sources = self._parameter_sources(
                    node.value,
                    parameter_taint,
                )

                direct_source = self._direct_source(node.value)

                summary_source = self._summary_source(node.value)

                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue

                    if parameter_sources:
                        parameter_taint[target.id] = set(
                            parameter_sources
                        )

                    if direct_source is not None:
                        source_taint[target.id] = direct_source

                    elif summary_source is not None:
                        source_taint[target.id] = summary_source

            elif isinstance(node, ast.AnnAssign):
                if (
                    node.value is None
                    or not isinstance(node.target, ast.Name)
                ):
                    continue

                parameter_sources = self._parameter_sources(
                    node.value,
                    parameter_taint,
                )

                if parameter_sources:
                    parameter_taint[node.target.id] = set(
                        parameter_sources
                    )

                direct_source = self._direct_source(node.value)

                if direct_source is not None:
                    source_taint[node.target.id] = direct_source

                else:
                    summary_source = self._summary_source(
                        node.value
                    )

                    if summary_source is not None:
                        source_taint[node.target.id] = (
                            summary_source
                        )

            elif isinstance(node, ast.Return):
                if node.value is None:
                    continue

                direct_source = self._direct_source(node.value)

                if direct_source is not None:
                    source_return = True
                    source_name = direct_source.source
                    source_line = direct_source.source_line

                summary_source = self._summary_source(
                    node.value
                )

                if summary_source is not None:
                    source_return = True
                    source_name = summary_source.source
                    source_line = summary_source.source_line

                for variable in self._variables(node.value):
                    taint = source_taint.get(variable)

                    if taint is not None:
                        source_return = True
                        source_name = taint.source
                        source_line = taint.source_line

                parameter_to_return.update(
                    self._parameter_sources(
                        node.value,
                        parameter_taint,
                    )
                )

                if isinstance(node.value, ast.Call):
                    called_name = self._call_name(
                        node.value.func
                    )

                    called_summary = self.summaries.get(
                        called_name
                    )

                    if called_summary is not None:
                        for parameter_index in (
                            called_summary.parameter_to_return
                        ):
                            if parameter_index >= len(
                                node.value.args
                            ):
                                continue

                            argument = node.value.args[
                                parameter_index
                            ]

                            parameter_to_return.update(
                                self._parameter_sources(
                                    argument,
                                    parameter_taint,
                                )
                            )

        return FunctionSummary(
            name=function.name,
            source_return=source_return,
            source_name=source_name,
            source_line=source_line,
            parameter_to_return=frozenset(
                parameter_to_return
            ),
        )

    def _direct_source(
        self,
        expression: ast.AST,
    ) -> Taint | None:

        if not isinstance(expression, ast.Call):
            return None

        name = self._call_name(expression.func)

        if name not in self.SOURCES:
            return None

        return Taint(
            source=name,
            source_line=getattr(expression, "lineno", 0),
        )

    def _summary_source(
        self,
        expression: ast.AST,
    ) -> Taint | None:

        if not isinstance(expression, ast.Call):
            return None

        name = self._call_name(expression.func)

        summary = self.summaries.get(name)

        if summary is None:
            return None

        if not summary.source_return:
            return None

        return Taint(
            source=summary.source_name or name,
            source_line=summary.source_line,
        )

    @staticmethod
    def _parameter_sources(
        expression: ast.AST,
        parameter_taint: dict[str, set[int]],
    ) -> set[int]:

        result: set[int] = set()

        for variable in FunctionSummaryBuilder._variables(
            expression
        ):
            result.update(
                parameter_taint.get(variable, set())
            )

        return result

    @staticmethod
    def _variables(expression: ast.AST) -> set[str]:
        return {
            node.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Name)
        }

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        parts: list[str] = []

        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            parts.append(node.id)

        return ".".join(reversed(parts))


class DataFlowAnalyzer(ast.NodeVisitor):

    SOURCES = {
        "input",
        "sys.stdin.readline",
    }

    SINKS = {
        "eval",
        "exec",
        "pickle.load",
        "pickle.loads",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
    }

    def __init__(
        self,
        summaries: dict[str, FunctionSummary],
    ) -> None:

        self.summaries = summaries

        self.tainted: dict[str, Taint] = {}

        self.findings: list[DataFlowFinding] = []

    def visit_Assign(self, node: ast.Assign) -> None:
        taint = self._taint_from_expression(node.value)

        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue

            if taint is not None:
                self.tainted[target.id] = taint
            else:
                self.tainted.pop(target.id, None)

        self.generic_visit(node)

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign,
    ) -> None:

        if (
            node.value is None
            or not isinstance(node.target, ast.Name)
        ):
            self.generic_visit(node)
            return

        taint = self._taint_from_expression(node.value)

        if taint is not None:
            self.tainted[node.target.id] = taint
        else:
            self.tainted.pop(node.target.id, None)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function_name = self._call_name(node.func)

        if function_name in self.SINKS:
            self._check_sink(
                node,
                function_name,
            )

        self.generic_visit(node)

    def _check_sink(
        self,
        node: ast.Call,
        sink: str,
    ) -> None:

        for argument in node.args:
            taint = self._taint_from_expression(argument)

            if taint is None:
                continue

            variables = self._variables(argument)

            variable = (
                sorted(variables)[0]
                if variables
                else "<expression>"
            )

            self.findings.append(
                DataFlowFinding(
                    source_line=taint.source_line,
                    sink_line=getattr(node, "lineno", 0),
                    variable=variable,
                    source=taint.source,
                    sink=sink,
                )
            )

    def _taint_from_expression(
        self,
        expression: ast.AST,
    ) -> Taint | None:

        # Direct source:
        #
        # x = input()
        if isinstance(expression, ast.Call):
            function_name = self._call_name(
                expression.func
            )

            if function_name in self.SOURCES:
                return Taint(
                    source=function_name,
                    source_line=getattr(
                        expression,
                        "lineno",
                        0,
                    ),
                )

            # Function known to return source data:
            #
            # x = read_expression()
            summary = self.summaries.get(
                function_name
            )

            if (
                summary is not None
                and summary.source_return
            ):
                return Taint(
                    source=(
                        summary.source_name
                        or function_name
                    ),
                    source_line=summary.source_line,
                )

            # Function propagates one of its parameters:
            #
            # x = prepare(tainted_value)
            if summary is not None:
                for index in (
                    summary.parameter_to_return
                ):
                    if index >= len(expression.args):
                        continue

                    argument = expression.args[index]

                    taint = self._taint_from_expression(
                        argument
                    )

                    if taint is not None:
                        return taint

        # Ordinary propagation:
        #
        # b = a
        # b = a.strip()
        # b = f"{a}"
        for variable in self._variables(expression):
            taint = self.tainted.get(variable)

            if taint is not None:
                return taint

        return None

    @staticmethod
    def _variables(expression: ast.AST) -> set[str]:
        return {
            node.id
            for node in ast.walk(expression)
            if isinstance(node, ast.Name)
        }

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        parts: list[str] = []

        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value

        if isinstance(node, ast.Name):
            parts.append(node.id)

        return ".".join(reversed(parts))


def analyze_dataflow(
    source: str,
) -> list[DataFlowFinding]:

    tree = ast.parse(source)

    summaries = FunctionSummaryBuilder(
        tree
    ).build()

    analyzer = DataFlowAnalyzer(
        summaries=summaries
    )

    analyzer.visit(tree)

    return analyzer.findings