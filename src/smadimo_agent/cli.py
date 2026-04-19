from __future__ import annotations

import argparse
import json
from typing import Optional

from dotenv import load_dotenv

from smadimo_agent.config import AgentConfig, ensure_lm_studio_server
from smadimo_agent.workflow import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autonomous ML agent built on LangChain and LangGraph.",
    )
    parser.add_argument("--dataset", required=True, help="Path to the input dataset.")
    parser.add_argument(
        "--business-task",
        required=True,
        help="Business task description passed directly into the agent prompt.",
    )
    parser.add_argument("--model", default=None, help="Primary LLM model name.")
    parser.add_argument("--review-model", default=None, help="Optional reviewer model name.")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory where runs, reports and memory will be stored.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature for the primary model.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    config = AgentConfig.from_runtime(
        model_name=args.model,
        review_model_name=args.review_model,
        output_root=args.output_root,
        temperature=args.temperature,
    )
    ensure_lm_studio_server(config)

    result = run_pipeline(
        dataset_path=args.dataset,
        business_task=args.business_task,
        config=config,
    )

    payload = {
        "run_id": result.get("run_id"),
        "task_type": result.get("task_type"),
        "target_column": result.get("target_column"),
        "selected_models": result.get("selected_models", []),
        "best_model_name": result.get("best_model_name"),
        "report_path": result.get("report_path"),
        "workspace_dir": result.get("workspace_dir"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
