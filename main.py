from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smadimo_agent.config import AgentConfig, ensure_lm_studio_server
from smadimo_agent.workflow import run_pipeline


def run_agent(
    business_task: str,
    csv_path: str,
    model_name: Optional[str] = None,
    output_root: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    load_dotenv()

    if not csv_path.lower().endswith(".csv"):
        raise ValueError("run_agent expects a path to a .csv file.")

    config = AgentConfig.from_runtime(
        model_name=model_name,
        output_root=output_root,
        temperature=temperature,
    )
    ensure_lm_studio_server(config)

    result = run_pipeline(
        dataset_path=csv_path,
        business_task=business_task,
        config=config,
    )

    return {
        "run_id": result.get("run_id"),
        "task_type": result.get("task_type"),
        "target_column": result.get("target_column"),
        "selected_models": result.get("selected_models", []),
        "best_model_name": result.get("best_model_name"),
        "report_path": result.get("report_path"),
        "workspace_dir": result.get("workspace_dir"),
    }


if __name__ == "__main__":
    BUSINESS_TASK = "У нас есть датасет с данными об аренде квартир. Помоги нам понять, какие факторы влияют на стоимость аренды, и предскажи стоимость аренды для новых объявлений."
    CSV_PATH = "/Users/leonidprokopev/projects/SMADIMO-GP-3/data/apartments_for_rent_classified_10K.csv"
    MODEL_NAME = "qwen/qwen3.5-9b"

    summary = run_agent(
        business_task=BUSINESS_TASK,
        csv_path=CSV_PATH,
        model_name=MODEL_NAME,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
