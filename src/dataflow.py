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
class Taint:
    source: str
    source_line: int


@dataclass(frozen=True)
class FunctionSummary:
    name: str
    source_return: bool
    source_name: str | None
    source_line: int
    parameter_to_return: frozenset[int]


def call_name(node: ast.AST) -> str:
    parts: list[str] = []

    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value

    if isinstance(node, ast.Name):
        parts.append(node.id)

    return ".".join(reversed(parts))


def attribute_name(node: ast.AST) -> str | None:
    """
    Convert a simple attribute chain into a stable name.

    Examples:
        request.value       -> "request.value"
        config.user.name    -> "config.user.name"

    More complex expressions such as get_request().value are
    intentionally not handled yet.
    """
    parts: list[str] = []

    current = node

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if not isinstance(current, ast.Name):
        return None

    parts.append(current.id)

    return ".".join(reversed(parts))


def variables(expression: ast.AST) -> set[str]:
    return {
        node.id
        for node in ast.walk(expression)
        if isinstance(node, ast.Name)
    }


class FunctionSummaryBuilder:

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

        parameter_taint: dict[str, set[int]] = {
            name: {index}
            for index, name in enumerate(parameters)
        }

        source_taint: dict[str, Taint] = {}

        source_return = False
        source_name: str | None = None
        source_line = 0

        parameter_to_return: set[int] = set()

        for node in ast.walk(function):

            if isinstance(node, (ast.Assign, ast.AnnAssign)):

                if isinstance(node, ast.Assign):
                    targets = node.targets
                    value = node.value
                else:
                    targets = [node.target]
                    value = node.value

                if value is None:
                    continue

                parameter_sources = self._parameter_sources(
                    value,
                    parameter_taint,
                )

                direct_source = self._source_from_expression(
                    value
                )

                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue

                    if parameter_sources:
                        parameter_taint[target.id] = set(
                            parameter_sources
                        )
                    else:
                        parameter_taint.pop(
                            target.id,
                            None,
                        )

                    if direct_source is not None:
                        source_taint[target.id] = direct_source
                    else:
                        source_taint.pop(
                            target.id,
                            None,
                        )

            elif isinstance(node, ast.Return):

                if node.value is None:
                    continue

                direct_source = self._source_from_expression(
                    node.value
                )

                if direct_source is not None:
                    source_return = True
                    source_name = direct_source.source
                    source_line = direct_source.source_line

                for variable in variables(node.value):
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
                    name = call_name(node.value.func)

                    summary = self.summaries.get(name)

                    if summary is not None:

                        if summary.source_return:
                            source_return = True
                            source_name = (
                                summary.source_name
                                or name
                            )
                            source_line = (
                                summary.source_line
                            )

                        for index in (
                            summary.parameter_to_return
                        ):
                            if index >= len(
                                node.value.args
                            ):
                                continue

                            parameter_to_return.update(
                                self._parameter_sources(
                                    node.value.args[index],
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

    def _source_from_expression(
        self,
        expression: ast.AST,
    ) -> Taint | None:

        if not isinstance(expression, ast.Call):
            return None

        name = call_name(expression.func)

        if name in self.SOURCES:
            return Taint(
                source=name,
                source_line=getattr(
                    expression,
                    "lineno",
                    0,
                ),
            )

        summary = self.summaries.get(name)

        if (
            summary is not None
            and summary.source_return
        ):
            return Taint(
                source=summary.source_name or name,
                source_line=summary.source_line,
            )

        return None

    @staticmethod
    def _parameter_sources(
        expression: ast.AST,
        parameter_taint: dict[str, set[int]],
    ) -> set[int]:

        result: set[int] = set()

        for variable in variables(expression):
            result.update(
                parameter_taint.get(
                    variable,
                    set(),
                )
            )

        return result


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

        # Ordinary variable taint:
        #
        # x = input()
        self.tainted: dict[str, Taint] = {}

        # Container taint:
        #
        # values.append(input())
        # data["x"] = input()
        self.tainted_containers: dict[str, Taint] = {}

        # Attribute taint:
        #
        # request.value = input()
        # config.user.name = input()
        self.tainted_attributes: dict[str, Taint] = {}

        self.findings: list[DataFlowFinding] = []

    # ---------------------------------------------------------
    # Assignments
    # ---------------------------------------------------------

    def visit_Assign(
        self,
        node: ast.Assign,
    ) -> None:

        taint = self._taint_from_expression(
            node.value
        )

        for target in node.targets:

            # x = ...
            if isinstance(target, ast.Name):
                self._assign_name(
                    target.id,
                    node.value,
                    taint,
                )

            # values[0] = ...
            # data["key"] = ...
            elif isinstance(target, ast.Subscript):
                self._assign_subscript(
                    target,
                    taint,
                )

            # request.value = ...
            # config.user.name = ...
            elif isinstance(target, ast.Attribute):
                self._assign_attribute(
                    target,
                    taint,
                )

        self.generic_visit(node)

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign,
    ) -> None:

        if node.value is None:
            self.generic_visit(node)
            return

        taint = self._taint_from_expression(
            node.value
        )

        if isinstance(node.target, ast.Name):
            self._assign_name(
                node.target.id,
                node.value,
                taint,
            )

        elif isinstance(
            node.target,
            ast.Subscript,
        ):
            self._assign_subscript(
                node.target,
                taint,
            )

        elif isinstance(
            node.target,
            ast.Attribute,
        ):
            self._assign_attribute(
                node.target,
                taint,
            )

        self.generic_visit(node)

    def _assign_name(
        self,
        name: str,
        value: ast.AST,
        taint: Taint | None,
    ) -> None:

        if taint is not None:
            self.tainted[name] = taint
        else:
            self.tainted.pop(
                name,
                None,
            )

        # Reassigning a container creates/replaces its state.
        #
        # values = []
        # data = {}
        if isinstance(
            value,
            (ast.List, ast.Dict, ast.Set, ast.Tuple),
        ):
            self.tainted_containers.pop(
                name,
                None,
            )

            literal_taint = (
                self._taint_inside_container_literal(
                    value
                )
            )

            if literal_taint is not None:
                self.tainted_containers[
                    name
                ] = literal_taint

        # Alias:
        #
        # other = values
        elif isinstance(value, ast.Name):

            container_taint = (
                self.tainted_containers.get(
                    value.id
                )
            )

            if container_taint is not None:
                self.tainted_containers[
                    name
                ] = container_taint
            else:
                self.tainted_containers.pop(
                    name,
                    None,
                )

        else:
            self.tainted_containers.pop(
                name,
                None,
            )

        # If an object variable is completely replaced, attributes
        # belonging to the old object should not stay tainted.
        #
        # request = RequestData()
        self._clear_attributes_for_root(name)

    def _assign_subscript(
        self,
        target: ast.Subscript,
        taint: Taint | None,
    ) -> None:

        container_name = self._container_name(
            target
        )

        if container_name is None:
            return

        if taint is not None:
            self.tainted_containers[
                container_name
            ] = taint

    def _assign_attribute(
        self,
        target: ast.Attribute,
        taint: Taint | None,
    ) -> None:

        name = attribute_name(target)

        if name is None:
            return

        if taint is not None:
            self.tainted_attributes[name] = taint
        else:
            # Important:
            #
            # request.value = input()
            # request.value = "fixed"
            #
            # The second assignment removes the old taint.
            self.tainted_attributes.pop(
                name,
                None,
            )

    def _clear_attributes_for_root(
        self,
        root: str,
    ) -> None:

        prefix = root + "."

        stale = [
            name
            for name in self.tainted_attributes
            if name.startswith(prefix)
        ]

        for name in stale:
            self.tainted_attributes.pop(
                name,
                None,
            )

    # ---------------------------------------------------------
    # Calls
    # ---------------------------------------------------------

    def visit_Call(
        self,
        node: ast.Call,
    ) -> None:

        name = call_name(node.func)

        # values.append(tainted_value)
        # values.extend(tainted_values)
        # values.insert(0, tainted_value)
        self._handle_container_method(
            node,
            name,
        )

        if name in self.SINKS:
            self._check_sink(
                node,
                name,
            )

        self.generic_visit(node)

    def _handle_container_method(
        self,
        node: ast.Call,
        name: str,
    ) -> None:

        if not isinstance(
            node.func,
            ast.Attribute,
        ):
            return

        if not isinstance(
            node.func.value,
            ast.Name,
        ):
            return

        container = node.func.value.id
        method = node.func.attr

        relevant_arguments: list[ast.AST] = []

        if method == "append":
            if node.args:
                relevant_arguments.append(
                    node.args[0]
                )

        elif method == "extend":
            if node.args:
                relevant_arguments.append(
                    node.args[0]
                )

        elif method == "insert":
            if len(node.args) >= 2:
                relevant_arguments.append(
                    node.args[1]
                )

        elif method == "update":
            if node.args:
                relevant_arguments.append(
                    node.args[0]
                )

        elif method == "setdefault":
            if len(node.args) >= 2:
                relevant_arguments.append(
                    node.args[1]
                )

        else:
            return

        for argument in relevant_arguments:

            taint = self._taint_from_expression(
                argument
            )

            if taint is not None:
                self.tainted_containers[
                    container
                ] = taint
                return

    # ---------------------------------------------------------
    # Sink detection
    # ---------------------------------------------------------

    def _check_sink(
        self,
        node: ast.Call,
        sink: str,
    ) -> None:

        for argument in node.args:

            taint = self._taint_from_expression(
                argument
            )

            if taint is None:
                continue

            variable = self._expression_label(
                argument
            )

            self.findings.append(
                DataFlowFinding(
                    source_line=taint.source_line,
                    sink_line=getattr(
                        node,
                        "lineno",
                        0,
                    ),
                    variable=variable,
                    source=taint.source,
                    sink=sink,
                )
            )

    @staticmethod
    def _expression_label(
        expression: ast.AST,
    ) -> str:

        if isinstance(expression, ast.Name):
            return expression.id

        if isinstance(expression, ast.Attribute):
            name = attribute_name(expression)

            if name is not None:
                return name

        if isinstance(expression, ast.Subscript):
            if isinstance(
                expression.value,
                ast.Name,
            ):
                return expression.value.id

        used_variables = variables(expression)

        if used_variables:
            return sorted(used_variables)[0]

        return "<expression>"

    # ---------------------------------------------------------
    # Taint resolution
    # ---------------------------------------------------------

    def _taint_from_expression(
        self,
        expression: ast.AST,
    ) -> Taint | None:

        # Direct source:
        #
        # input()
        if isinstance(expression, ast.Call):

            name = call_name(expression.func)

            if name in self.SOURCES:
                return Taint(
                    source=name,
                    source_line=getattr(
                        expression,
                        "lineno",
                        0,
                    ),
                )

            summary = self.summaries.get(name)

            if summary is not None:

                if summary.source_return:
                    return Taint(
                        source=(
                            summary.source_name
                            or name
                        ),
                        source_line=(
                            summary.source_line
                        ),
                    )

                for index in (
                    summary.parameter_to_return
                ):

                    if index >= len(
                        expression.args
                    ):
                        continue

                    taint = (
                        self._taint_from_expression(
                            expression.args[index]
                        )
                    )

                    if taint is not None:
                        return taint

        # Attribute lookup:
        #
        # request.value
        # config.user.name
        if isinstance(
            expression,
            ast.Attribute,
        ):

            name = attribute_name(
                expression
            )

            if name is not None:
                taint = (
                    self.tainted_attributes.get(
                        name
                    )
                )

                if taint is not None:
                    return taint

        # Container lookup:
        #
        # values[0]
        # data["value"]
        if isinstance(
            expression,
            ast.Subscript,
        ):

            container = self._container_name(
                expression
            )

            if container is not None:

                taint = (
                    self.tainted_containers.get(
                        container
                    )
                )

                if taint is not None:
                    return taint

        # Search attributes nested inside a larger expression:
        #
        # f"{request.value}"
        # request.value + "!"
        for node in ast.walk(expression):

            if not isinstance(
                node,
                ast.Attribute,
            ):
                continue

            name = attribute_name(node)

            if name is None:
                continue

            taint = self.tainted_attributes.get(
                name
            )

            if taint is not None:
                return taint

        # Ordinary variables / nested expressions.
        for variable in variables(expression):

            taint = self.tainted.get(variable)

            if taint is not None:
                return taint

        return None

    def _taint_inside_container_literal(
        self,
        expression: ast.AST,
    ) -> Taint | None:

        if isinstance(
            expression,
            (ast.List, ast.Tuple, ast.Set),
        ):

            for element in expression.elts:
                taint = self._taint_from_expression(
                    element
                )

                if taint is not None:
                    return taint

        elif isinstance(expression, ast.Dict):

            for value in expression.values:

                if value is None:
                    continue

                taint = self._taint_from_expression(
                    value
                )

                if taint is not None:
                    return taint

        return None

    @staticmethod
    def _container_name(
        expression: ast.Subscript,
    ) -> str | None:

        value = expression.value

        if isinstance(value, ast.Name):
            return value.id

        return None


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