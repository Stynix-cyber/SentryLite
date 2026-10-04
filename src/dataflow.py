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


class AliasResolver(ast.NodeVisitor):
    """
    Collect simple callable/module aliases.

    Examples:

        reader = input
            reader -> input

        import pickle as p
            p -> pickle

        from pickle import loads as deserialize
            deserialize -> pickle.loads
    """

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}

    def visit_Import(
        self,
        node: ast.Import,
    ) -> None:

        for alias in node.names:
            if alias.asname:
                self.aliases[alias.asname] = alias.name

        self.generic_visit(node)

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:

        if node.module is None:
            self.generic_visit(node)
            return

        for alias in node.names:
            local_name = alias.asname or alias.name
            full_name = f"{node.module}.{alias.name}"

            self.aliases[local_name] = full_name

        self.generic_visit(node)

    def visit_Assign(
        self,
        node: ast.Assign,
    ) -> None:

        if isinstance(node.value, ast.Name):
            source_name = self.resolve(node.value.id)

            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = source_name

        elif isinstance(node.value, ast.Attribute):
            source_name = self.resolve(
                call_name(node.value)
            )

            if source_name:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.aliases[target.id] = source_name

        self.generic_visit(node)

    def resolve(
        self,
        name: str,
    ) -> str:

        if not name:
            return name

        seen: set[str] = set()
        current = name

        while current not in seen:
            seen.add(current)

            direct = self.aliases.get(current)

            if direct is not None:
                current = direct
                continue

            parts = current.split(".")

            if not parts:
                break

            root = parts[0]
            root_alias = self.aliases.get(root)

            if root_alias is None:
                break

            if len(parts) == 1:
                current = root_alias
            else:
                current = ".".join(
                    [root_alias, *parts[1:]]
                )

        return current


class FunctionSummaryBuilder:

    SOURCES = {
        "input",
        "sys.stdin.readline",
    }

    def __init__(
        self,
        tree: ast.AST,
        aliases: AliasResolver,
    ) -> None:

        self.tree = tree
        self.aliases = aliases
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
                    name = self._resolved_call_name(
                        node.value.func
                    )

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
                            argument = self._argument_for_parameter(
                                node.value,
                                summary,
                                index,
                            )

                            if argument is None:
                                continue

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

    def _source_from_expression(
        self,
        expression: ast.AST,
    ) -> Taint | None:

        if not isinstance(expression, ast.Call):
            return None

        name = self._resolved_call_name(
            expression.func
        )

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

    def _resolved_call_name(
        self,
        node: ast.AST,
    ) -> str:

        return self.aliases.resolve(
            call_name(node)
        )

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

    @staticmethod
    def _argument_for_parameter(
        call: ast.Call,
        summary: FunctionSummary,
        index: int,
    ) -> ast.AST | None:

        if index < len(call.args):
            return call.args[index]

        # FunctionSummary currently stores parameter indexes,
        # not their names. Keyword arguments are still handled
        # conservatively elsewhere through variables(expression).
        return None


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
        aliases: AliasResolver,
    ) -> None:

        self.summaries = summaries
        self.aliases = aliases

        # Ordinary variables:
        #
        # value = input()
        self.tainted: dict[str, Taint] = {}

        # Containers:
        #
        # values.append(input())
        # data["value"] = input()
        self.tainted_containers: dict[str, Taint] = {}

        # Attributes:
        #
        # request.value = input()
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

            if isinstance(target, ast.Name):
                self._assign_name(
                    target.id,
                    node.value,
                    taint,
                )

            elif isinstance(target, ast.Subscript):
                self._assign_subscript(
                    target,
                    taint,
                )

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

        elif isinstance(node.target, ast.Subscript):
            self._assign_subscript(
                node.target,
                taint,
            )

        elif isinstance(node.target, ast.Attribute):
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

        raw_name = call_name(node.func)

        name = self.aliases.resolve(
            raw_name
        )

        self._handle_container_method(
            node,
            raw_name,
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

        arguments = list(node.args)

        arguments.extend(
            keyword.value
            for keyword in node.keywords
        )

        for argument in arguments:

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

        # Direct call:
        #
        # input()
        # reader()
        # p.loads(...)
        # deserialize(...)
        if isinstance(expression, ast.Call):

            raw_name = call_name(
                expression.func
            )

            name = self.aliases.resolve(
                raw_name
            )

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

                    if index < len(
                        expression.args
                    ):
                        taint = (
                            self._taint_from_expression(
                                expression.args[index]
                            )
                        )

                        if taint is not None:
                            return taint

            # Conservative propagation through keyword arguments.
            for keyword in expression.keywords:
                taint = self._taint_from_expression(
                    keyword.value
                )

                if taint is not None:
                    return taint

        # Attribute:
        #
        # request.value
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

        # Container:
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

        # Attributes nested in larger expressions.
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

    alias_resolver = AliasResolver()
    alias_resolver.visit(tree)

    summaries = FunctionSummaryBuilder(
        tree,
        aliases=alias_resolver,
    ).build()

    analyzer = DataFlowAnalyzer(
        summaries=summaries,
        aliases=alias_resolver,
    )

    analyzer.visit(tree)

    return analyzer.findings