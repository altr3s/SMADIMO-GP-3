from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from openai import RateLimitError

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from smadimo_agent.config import AgentConfig
from smadimo_agent.io_utils import (
    copy_dataset_to_workspace,
    ensure_dir,
    now_run_id,
    read_json,
    safe_resolve,
    slugify,
    write_json,
)
from smadimo_agent.ml_tools import build_ml_tools, collect_artifacts
from smadimo_agent.prompts import build_phase_user_message, build_system_prompt
from smadimo_agent.state import (
    PHASE_ANALYZE,
    PHASE_EDA,
    PHASE_EVALUATE,
    PHASE_FEATURES,
    PHASE_MODEL_SELECTION,
    PHASE_PERSIST,
    PHASE_REPORT,
    PHASE_SPLIT,
    PHASE_TRAIN_CLASSIFICATION,
    PHASE_TRAIN_CLUSTERING,
    PHASE_TRAIN_REGRESSION,
    StageAgentState,
    TRAINING_PHASE_BY_TASK,
    WorkflowState,
    append_log,
    merge_phase_output,
)


PHASE_TOOL_NAMES: Dict[str, List[str]] = {
    PHASE_ANALYZE: ["profile_dataset", "get_dataset_schema", "set_modeling_goal", "clean_dataset"],
    PHASE_EDA: ["profile_dataset", "get_dataset_schema", "run_eda"],
    PHASE_FEATURES: ["profile_dataset", "get_dataset_schema", "run_eda", "engineer_features"],
    PHASE_MODEL_SELECTION: ["profile_dataset", "get_dataset_schema", "run_eda", "select_candidate_models"],
    PHASE_SPLIT: ["prepare_splits"],
    PHASE_TRAIN_CLASSIFICATION: ["train_models"],
    PHASE_TRAIN_REGRESSION: ["train_models"],
    PHASE_TRAIN_CLUSTERING: ["train_models"],
    PHASE_EVALUATE: ["evaluate_models"],
    PHASE_PERSIST: ["load_long_term_memory", "save_best_model"],
    PHASE_REPORT: ["write_report"],
}


def _required_paths(workspace_dir: Path, phase: str) -> List[Path]:
    if phase == PHASE_ANALYZE:
        return [
            workspace_dir / "analysis" / "dataset_profile.json",
            workspace_dir / "analysis" / "schema_snapshot.json",
            workspace_dir / "analysis" / "modeling_goal.json",
            workspace_dir / "analysis" / "cleaning_report.json",
            workspace_dir / "data" / "cleaned" / "cleaned_dataset.csv",
        ]
    if phase == PHASE_EDA:
        return [workspace_dir / "analysis" / "eda_report.json"]
    if phase == PHASE_FEATURES:
        return [
            workspace_dir / "analysis" / "schema_snapshot.json",
            workspace_dir / "analysis" / "feature_report.json",
            workspace_dir / "data" / "featured" / "featured_dataset.csv",
        ]
    if phase == PHASE_MODEL_SELECTION:
        return [workspace_dir / "modeling" / "model_plan.json"]
    if phase == PHASE_SPLIT:
        return [
            workspace_dir / "modeling" / "splits" / "train.csv",
            workspace_dir / "modeling" / "splits" / "val.csv",
            workspace_dir / "modeling" / "splits" / "test.csv",
        ]
    if phase in {
        PHASE_TRAIN_CLASSIFICATION,
        PHASE_TRAIN_REGRESSION,
        PHASE_TRAIN_CLUSTERING,
    }:
        return [workspace_dir / "modeling" / "leaderboard.json"]
    if phase == PHASE_EVALUATE:
        return [
            workspace_dir / "modeling" / "evaluation.json",
            workspace_dir / "models" / "best_current_model.joblib",
        ]
    if phase == PHASE_PERSIST:
        return [workspace_dir.parent / "memory" / "best_registry.json"]
    if phase == PHASE_REPORT:
        return [workspace_dir / "reports" / "run_report.md"]
    return []


def _extract_summary(result: Dict[str, Any]) -> str:
    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_chunks = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_chunks.append(block.get("text", ""))
            joined = "\n".join(chunk for chunk in text_chunks if chunk)
            if joined.strip():
                return joined.strip()
    return "Stage finished."


def _refresh_state_from_workspace(state: WorkflowState) -> WorkflowState:
    workspace_dir = Path(state["workspace_dir"])
    artifacts = collect_artifacts(workspace_dir)
    goal = read_json(workspace_dir / "analysis" / "modeling_goal.json")
    schema_snapshot = read_json(workspace_dir / "analysis" / "schema_snapshot.json")
    model_plan = read_json(workspace_dir / "modeling" / "model_plan.json")
    evaluation = read_json(workspace_dir / "modeling" / "evaluation.json")
    history = read_json(workspace_dir.parent / "memory" / "best_registry.json", default={"best_run": None})

    refreshed = dict(state)
    refreshed["artifacts"] = artifacts
    refreshed["task_type"] = goal.get("task_type", state.get("task_type"))
    refreshed["target_column"] = goal.get("target_column", state.get("target_column"))
    refreshed["selected_models"] = model_plan.get("selected_models", state.get("selected_models", []))
    refreshed["best_model_name"] = evaluation.get("best_model_name", state.get("best_model_name"))
    refreshed["best_model_path"] = evaluation.get(
        "current_best_model_path",
        state.get("best_model_path"),
    )
    refreshed["schema_summary"] = schema_snapshot.get("summary", state.get("schema_summary"))
    if "report_md" in artifacts:
        refreshed["report_path"] = artifacts["report_md"]
    refreshed["history_comparison"] = history
    return refreshed  # type: ignore[return-value]


def _filter_tools(all_tools: Iterable[Any], phase: str) -> List[Any]:
    allowed = set(PHASE_TOOL_NAMES[phase])
    return [tool for tool in all_tools if getattr(tool, "name", "") in allowed]


def _extract_rate_limit_delay(error: RateLimitError, default_seconds: int = 65) -> int:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}

    reset_header = (
        headers.get("x-ratelimit-reset")
        or headers.get("X-RateLimit-Reset")
    )
    if reset_header:
        try:
            reset_ms = int(reset_header)
            delay = max(1, int((reset_ms / 1000) - time.time()) + 1)
            return min(delay, 300)
        except (TypeError, ValueError):
            pass

    body = getattr(error, "body", None) or {}
    metadata = body.get("error", {}).get("metadata", {})
    nested_headers = metadata.get("headers", {})
    nested_reset = nested_headers.get("X-RateLimit-Reset") or nested_headers.get("x-ratelimit-reset")
    if nested_reset:
        try:
            reset_ms = int(nested_reset)
            delay = max(1, int((reset_ms / 1000) - time.time()) + 1)
            return min(delay, 300)
        except (TypeError, ValueError):
            pass

    return default_seconds


def _invoke_agent_with_rate_limit_retry(
    agent: Any,
    payload: Dict[str, Any],
    thread_id: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            return agent.invoke(
                payload,
                config={"configurable": {"thread_id": thread_id}},
            )
        except RateLimitError as error:
            if attempt >= max_retries:
                raise
            delay = _extract_rate_limit_delay(error)
            time.sleep(delay)

    raise RuntimeError("Agent invocation failed after rate-limit retries.")


def _invoke_phase_agent(
    state: WorkflowState,
    phase: str,
    config: AgentConfig,
    saver: InMemorySaver,
    all_tools: List[Any],
) -> WorkflowState:
    workspace_dir = Path(state["workspace_dir"])
    tools = _filter_tools(all_tools, phase)

    reminder = ""
    for attempt in range(2):
        system_prompt = build_system_prompt(
            {
                "phase": phase,
                "business_task": state["business_task"],
                "task_type": state.get("task_type"),
                "target_column": state.get("target_column"),
                "schema_summary": state.get("schema_summary"),
            }
        )

        agent = create_agent(
            model=config.build_primary_model(),
            tools=tools,
            system_prompt=system_prompt,
            state_schema=StageAgentState,
            checkpointer=saver,
        )

        result = _invoke_agent_with_rate_limit_retry(
            agent=agent,
            payload={
                "messages": [
                    {
                        "role": "user",
                        "content": build_phase_user_message(state, phase) + reminder,
                    }
                ],
                "phase": phase,
                "business_task": state["business_task"],
                "dataset_path": state["dataset_path"],
                "workspace_dir": state["workspace_dir"],
                "task_type": state.get("task_type"),
                "target_column": state.get("target_column"),
                "schema_summary": state.get("schema_summary"),
            },
            thread_id=state["thread_id"],
        )

        missing = [path for path in _required_paths(workspace_dir, phase) if not path.exists()]
        if not missing:
            refreshed = _refresh_state_from_workspace(state)
            summary = _extract_summary(result)
            refreshed["phase"] = phase
            refreshed["phase_outputs"] = merge_phase_output(state, phase, summary)
            refreshed["execution_log"] = append_log(
                state,
                f"Phase `{phase}` completed with artifacts in {workspace_dir}.",
            )
            return refreshed

        reminder = (
            "\n\nПовтори шаг еще раз. На предыдущей попытке не были созданы обязательные артефакты:\n"
            + "\n".join(f"- {path}" for path in missing)
        )

    raise RuntimeError(f"Phase `{phase}` finished without required artifacts.")


def _bootstrap_node(config: AgentConfig, dataset_path: Path, business_task: str):
    def node(_: WorkflowState) -> WorkflowState:
        run_id = now_run_id() + "-" + slugify(dataset_path.stem)
        workspace_dir = ensure_dir(config.output_root.resolve() / "runs" / run_id)
        ensure_dir(workspace_dir / "analysis")
        ensure_dir(workspace_dir / "data")
        ensure_dir(workspace_dir / "modeling")
        ensure_dir(workspace_dir / "models")
        ensure_dir(workspace_dir / "reports")
        ensure_dir(workspace_dir.parent / "memory")
        raw_copy = copy_dataset_to_workspace(dataset_path, workspace_dir)

        workflow_manifest = {
            "nodes": [
                "bootstrap",
                PHASE_ANALYZE,
                PHASE_EDA,
                PHASE_FEATURES,
                PHASE_MODEL_SELECTION,
                PHASE_SPLIT,
                PHASE_TRAIN_CLASSIFICATION,
                PHASE_TRAIN_REGRESSION,
                PHASE_TRAIN_CLUSTERING,
                PHASE_EVALUATE,
                PHASE_PERSIST,
                PHASE_REPORT,
            ],
            "edges": [
                ["bootstrap", PHASE_ANALYZE],
                [PHASE_ANALYZE, PHASE_EDA],
                [PHASE_EDA, PHASE_FEATURES],
                [PHASE_FEATURES, PHASE_MODEL_SELECTION],
                [PHASE_MODEL_SELECTION, PHASE_SPLIT],
                [PHASE_SPLIT, "conditional_training_phase"],
                [PHASE_TRAIN_CLASSIFICATION, PHASE_EVALUATE],
                [PHASE_TRAIN_REGRESSION, PHASE_EVALUATE],
                [PHASE_TRAIN_CLUSTERING, PHASE_EVALUATE],
                [PHASE_EVALUATE, PHASE_PERSIST],
                [PHASE_PERSIST, PHASE_REPORT],
            ],
        }
        write_json(workspace_dir / "workflow_spec.json", workflow_manifest)

        return {
            "business_task": business_task,
            "dataset_path": str(raw_copy),
            "workspace_dir": str(workspace_dir),
            "run_id": run_id,
            "thread_id": run_id,
            "phase": "bootstrap",
            "artifacts": {"workflow_spec": str(workspace_dir / "workflow_spec.json")},
            "phase_outputs": {},
            "execution_log": [f"Bootstrap finished for run `{run_id}`."],
            "errors": [],
            "selected_models": [],
        }

    return node


def _make_phase_node(
    phase: str,
    config: AgentConfig,
    saver: InMemorySaver,
    all_tools: List[Any],
):
    def node(state: WorkflowState) -> WorkflowState:
        return _invoke_phase_agent(state, phase, config, saver, all_tools)

    return node


def _route_training_phase(state: WorkflowState) -> str:
    task_type = state.get("task_type")
    if task_type not in TRAINING_PHASE_BY_TASK:
        raise ValueError(
            "Task type was not defined after analysis. Expected one of "
            f"{sorted(TRAINING_PHASE_BY_TASK.keys())}, got {task_type!r}."
        )
    return TRAINING_PHASE_BY_TASK[task_type]


def build_workflow(
    config: AgentConfig,
    dataset_path: str,
    business_task: str,
):
    dataset = safe_resolve(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    saver = InMemorySaver()
    all_tools = build_ml_tools()
    graph = StateGraph(WorkflowState)

    graph.add_node("bootstrap", _bootstrap_node(config, dataset, business_task))
    graph.add_node(PHASE_ANALYZE, _make_phase_node(PHASE_ANALYZE, config, saver, all_tools))
    graph.add_node(PHASE_EDA, _make_phase_node(PHASE_EDA, config, saver, all_tools))
    graph.add_node(PHASE_FEATURES, _make_phase_node(PHASE_FEATURES, config, saver, all_tools))
    graph.add_node(
        PHASE_MODEL_SELECTION,
        _make_phase_node(PHASE_MODEL_SELECTION, config, saver, all_tools),
    )
    graph.add_node(PHASE_SPLIT, _make_phase_node(PHASE_SPLIT, config, saver, all_tools))
    graph.add_node(
        PHASE_TRAIN_CLASSIFICATION,
        _make_phase_node(PHASE_TRAIN_CLASSIFICATION, config, saver, all_tools),
    )
    graph.add_node(
        PHASE_TRAIN_REGRESSION,
        _make_phase_node(PHASE_TRAIN_REGRESSION, config, saver, all_tools),
    )
    graph.add_node(
        PHASE_TRAIN_CLUSTERING,
        _make_phase_node(PHASE_TRAIN_CLUSTERING, config, saver, all_tools),
    )
    graph.add_node(PHASE_EVALUATE, _make_phase_node(PHASE_EVALUATE, config, saver, all_tools))
    graph.add_node(PHASE_PERSIST, _make_phase_node(PHASE_PERSIST, config, saver, all_tools))
    graph.add_node(PHASE_REPORT, _make_phase_node(PHASE_REPORT, config, saver, all_tools))

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", PHASE_ANALYZE)
    graph.add_edge(PHASE_ANALYZE, PHASE_EDA)
    graph.add_edge(PHASE_EDA, PHASE_FEATURES)
    graph.add_edge(PHASE_FEATURES, PHASE_MODEL_SELECTION)
    graph.add_edge(PHASE_MODEL_SELECTION, PHASE_SPLIT)
    graph.add_conditional_edges(
        PHASE_SPLIT,
        _route_training_phase,
        {
            PHASE_TRAIN_CLASSIFICATION: PHASE_TRAIN_CLASSIFICATION,
            PHASE_TRAIN_REGRESSION: PHASE_TRAIN_REGRESSION,
            PHASE_TRAIN_CLUSTERING: PHASE_TRAIN_CLUSTERING,
        },
    )
    graph.add_edge(PHASE_TRAIN_CLASSIFICATION, PHASE_EVALUATE)
    graph.add_edge(PHASE_TRAIN_REGRESSION, PHASE_EVALUATE)
    graph.add_edge(PHASE_TRAIN_CLUSTERING, PHASE_EVALUATE)
    graph.add_edge(PHASE_EVALUATE, PHASE_PERSIST)
    graph.add_edge(PHASE_PERSIST, PHASE_REPORT)
    graph.add_edge(PHASE_REPORT, END)

    return graph.compile()


def run_pipeline(
    dataset_path: str,
    business_task: str,
    config: AgentConfig,
) -> WorkflowState:
    app = build_workflow(config, dataset_path=dataset_path, business_task=business_task)
    result = app.invoke({})
    return result
