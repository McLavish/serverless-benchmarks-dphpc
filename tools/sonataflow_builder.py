#!/usr/bin/env python3
"""
Generate SonataFlow (CNCF Serverless Workflow) YAML along with Knative Service
manifests for SEBS workflows. This is a thin helper so users can go from
`definition.json` + function image mapping to runnable manifests under a
Quarkus SonataFlow operator on Knative.

Usage:
  python tools/sonataflow_builder.py \
    --definition benchmarks/600.workflows/6xx.OCR-pipeline/definition.json \
    --name ocr-pipeline \
    --namespace sebs \
    --image split=ghcr.io/example/split:latest \
    --image detect=ghcr.io/example/detect:latest \
    --image recognize=ghcr.io/example/recognize:latest \
    --image merge=ghcr.io/example/merge:latest \
    --out manifests.yaml
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from sebs.sonataflow.translator import definition_to_sonataflow_yaml, definition_to_sonataflow


KNATIVE_TEMPLATE = """\
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {name}
  namespace: {namespace}
spec:
  template:
    spec:
      containers:
        - image: {image}
          env:
            - name: FUNCTION_NAME
              value: {name}
"""

SONATAFLOW_CR_TEMPLATE = """\
apiVersion: sonataflow.org/v1alpha08
kind: SonataFlow
metadata:
  name: {name}
  namespace: {namespace}
spec:
  flow: |
{workflow_yaml}
"""


def parse_images(image_args: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in image_args:
        if "=" not in item:
            raise ValueError(f"Image argument must be key=value, got: {item}")
        fn, img = item.split("=", 1)
        mapping[fn.strip()] = img.strip()
    return mapping


def build_knative_services(namespace: str, images: Dict[str, str], functions: List[str]) -> List[str]:
    yamls = []
    for fn in functions:
        image = images.get(fn)
        if not image:
            raise ValueError(f"Missing image mapping for function: {fn}")
        yamls.append(KNATIVE_TEMPLATE.format(name=fn, namespace=namespace, image=image))
    return yamls


def main():
    parser = argparse.ArgumentParser(description="Generate SonataFlow + Knative YAML for SEBS workflow.")
    parser.add_argument("--definition", required=True, help="Path to SEBS workflow definition.json")
    parser.add_argument("--name", required=True, help="Workflow name/id")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace for resources")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Mapping of function=image (repeat for each function)",
    )
    parser.add_argument("--out", default="-", help="Output file path or '-' for stdout")

    args = parser.parse_args()

    definition = json.loads(Path(args.definition).read_text())
    spec, functions = definition_to_sonataflow(definition, name=args.name)
    images = parse_images(args.image)

    workflow_yaml = definition_to_sonataflow_yaml(definition, name=args.name)
    knative_yamls = build_knative_services(args.namespace, images, sorted(functions))

    pieces = []
    pieces.extend(knative_yamls)
    pieces.append(SONATAFLOW_CR_TEMPLATE.format(name=args.name, namespace=args.namespace, workflow_yaml="\n".join("    " + line for line in workflow_yaml.splitlines())))
    output = "\n---\n".join(pieces)

    if args.out == "-" or args.out == "/dev/stdout":
        sys.stdout.write(output)
    else:
        Path(args.out).write_text(output)
        print(f"Wrote manifests to {args.out}")


if __name__ == "__main__":
    main()
