from typing import Dict, List, Optional, Set

from sebs.faas.function import FunctionConfig, Workflow


class SonataFlowWorkflow(Workflow):
    """
    Lightweight wrapper that keeps track of the generated SonataFlow (CNCF Serverless Workflow)
    spec for a SEBS benchmark workflow. Mirrors the structure of other providers' Workflow
    classes so it can be cached/serialized the same way.
    """

    def __init__(
        self,
        name: str,
        benchmark: str,
        code_package_hash: str,
        cfg: FunctionConfig,
        sonataflow_spec: Dict,
        functions: Optional[Set[str]] = None,
    ):
        super().__init__(benchmark, name, code_package_hash, cfg)
        self.sonataflow_spec = sonataflow_spec
        self.functions = list(functions or [])

    @staticmethod
    def typename() -> str:
        return "SonataFlow.Workflow"

    def serialize(self) -> dict:
        return {
            **super().serialize(),
            "sonataflow_spec": self.sonataflow_spec,
            "functions": self.functions,
        }

    @staticmethod
    def deserialize(cached_config: dict) -> "SonataFlowWorkflow":
        cfg = FunctionConfig.deserialize(cached_config["config"])
        return SonataFlowWorkflow(
            name=cached_config["name"],
            benchmark=cached_config["benchmark"],
            code_package_hash=cached_config["hash"],
            cfg=cfg,
            sonataflow_spec=cached_config["sonataflow_spec"],
            functions=set(cached_config.get("functions", [])),
        )
