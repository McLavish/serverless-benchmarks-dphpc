import json
from typing import Dict, List, Set, Tuple

try:
    import yaml
except ImportError:
    yaml = None


class UnsupportedWorkflowError(Exception):
    pass


def _collect_functions(funcs: Set[str], func_name: str):
    if func_name:
        funcs.add(func_name)


def _task_state(name: str, state: Dict, funcs: Set[str]) -> Dict:
    func_name = state["func_name"]
    _collect_functions(funcs, func_name)
    node = {
        "name": name,
        "type": "operation",
        "actions": [{"name": func_name, "functionRef": func_name}],
    }
    if "next" in state:
        node["transition"] = state["next"]
    else:
        node["end"] = True
    return node


def _map_state(name: str, state: Dict, funcs: Set[str]) -> Dict:
    nested_states: Dict = state["states"]
    root = state["root"]
    if root not in nested_states:
        raise UnsupportedWorkflowError(f"Map state {name} missing root state {root}")
    root_state = nested_states[root]
    func_name = root_state["func_name"]
    _collect_functions(funcs, func_name)
    node = {
        "name": name,
        "type": "foreach",
        "inputCollection": f"$.{state['array']}",
        "iterationParam": "item",
        "actions": [{"name": func_name, "functionRef": func_name}],
    }
    if "next" in state:
        node["transition"] = state["next"]
    else:
        node["end"] = True
    return node


def _loop_state(name: str, state: Dict, funcs: Set[str]) -> Dict:
    func_name = state["func_name"]
    _collect_functions(funcs, func_name)
    node = {
        "name": name,
        "type": "operation",
        "actions": [{"name": func_name, "functionRef": func_name}],
    }
    if "next" in state:
        node["transition"] = state["next"]
    else:
        node["end"] = True
    return node


def _linearize(definition: Dict) -> List[Tuple[str, Dict]]:
    states = definition["states"]
    cursor = definition["root"]
    ordered: List[Tuple[str, Dict]] = []
    visited = set()
    while cursor:
        if cursor in visited:
            raise UnsupportedWorkflowError(f"Cycle detected at state {cursor}")
        visited.add(cursor)
        state = states[cursor]
        ordered.append((cursor, state))
        cursor = state.get("next")
    return ordered


def definition_to_sonataflow(definition: Dict, name: str = "workflow") -> Tuple[Dict, Set[str]]:
    """
    Convert SEBS workflow definition.json into a minimal SonataFlow (CNCF Serverless Workflow)
    spec dictionary. Supports task/map/loop constructs used by the SEBS workflows.
    Returns (spec, functions_used).
    """
    if "root" not in definition or "states" not in definition:
        raise UnsupportedWorkflowError("Invalid workflow definition")

    ordered_states = _linearize(definition)
    funcs: Set[str] = set()
    sonata_states: List[Dict] = []

    builders = {"task": _task_state, "map": _map_state, "loop": _loop_state}

    for state_name, state in ordered_states:
        stype = state["type"]
        if stype not in builders:
            raise UnsupportedWorkflowError(f"Unsupported state type {stype}")
        sonata_states.append(builders[stype](state_name, state, funcs))

    functions_block = [
        {"name": fn, "type": "custom", "operation": f"fn://{fn}"} for fn in sorted(funcs)
    ]
    spec = {
        "id": name,
        "name": name,
        "version": "0.1",
        "specVersion": "0.8",
        "functions": functions_block,
        "states": sonata_states,
    }
    return spec, funcs


def sonataflow_to_yaml(spec: Dict) -> str:
    if yaml is None:
        raise ImportError("PyYAML is required for YAML export")
    return yaml.safe_dump(spec, sort_keys=False)


def definition_to_sonataflow_yaml(definition: Dict, name: str = "workflow") -> str:
    spec, _ = definition_to_sonataflow(definition, name=name)
    if yaml:
        return sonataflow_to_yaml(spec)
    return json.dumps(spec, indent=2)
