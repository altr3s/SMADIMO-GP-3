from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

try:
    from langchain.agents import AgentState
except ImportError:
    class AgentState(TypedDict, total=False):
        messages: List[Any]


PHASE_ANALYZE = "analyze"
PHASE_EDA = "eda"
PHASE_FEATURES = "feature_engineering"
PHASE_MODEL_SELECTION = "model_selection"
PHASE_SPLIT = "split"
PHASE_TRAIN_CLASSIFICATION = "train_classification"
PHASE_TRAIN_REGRESSION = "train_regression"
PHASE_TRAIN_CLUSTERING = "train_clustering"
PHASE_EVALUATE = "evaluate"
PHASE_PERSIST = "persist"
PHASE_REPORT = "report"

TRAINING_PHASE_BY_TASK = {
    "classification": PHASE_TRAIN_CLASSIFICATION,
    "regression": PHASE_TRAIN_REGRESSION,
    "clustering": PHASE_TRAIN_CLUSTERING,
}


class WorkflowState(TypedDict, total=False):
    business_task: str
    dataset_path: str
    workspace_dir: str
    run_id: str
    thread_id: str
    phase: str
    task_type: Optional[str]
    target_column: Optional[str]
    selected_models: List[str]
    best_model_name: Optional[str]
    best_model_path: Optional[str]
    report_path: Optional[str]
    schema_summary: Optional[str]
    history_comparison: Dict[str, Any]
    artifacts: Dict[str, str]
    phase_outputs: Dict[str, str]
    execution_log: List[str]
    errors: List[str]


class StageAgentState(AgentState, total=False):
    phase: str
    business_task: str
    dataset_path: str
    workspace_dir: str
    task_type: Optional[str]
    target_column: Optional[str]
    schema_summary: Optional[str]


def append_log(state: WorkflowState, message: str) -> List[str]:
    log = list(state.get("execution_log", []))
    log.append(message)
    return log


def merge_phase_output(
    state: WorkflowState,
    phase: str,
    summary: str,
) -> Dict[str, str]:
    outputs = dict(state.get("phase_outputs", {}))
    outputs[phase] = summary
    return outputs
