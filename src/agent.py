import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import NotRequired, Required

from tools import get_all_tools, read_json
from prompts import PHASE_TOOLS, get_system_prompt, get_phase_message
from user_feedback import (
    print_json_summary_block,
    print_phase_intro,
    print_phase_outro,
    print_phase_transcript,
    print_run_footer,
    print_run_header,
)


class MyAgentState(AgentState[Any]):
    workspace_dir: Required[str]
    dataset_path: Required[str]
    business_task: Required[str]
    task_type: NotRequired[Any]
    target_column: NotRequired[Any]
    schema_summary: NotRequired[Any]


def build_graph_input(state, user_message: str) -> dict:
    return {
        "messages": [{"role": "user", "content": user_message}],
        "workspace_dir": state["workspace_dir"],
        "dataset_path": state["dataset_path"],
        "business_task": state["business_task"],
        "task_type": state.get("task_type"),
        "target_column": state.get("target_column"),
        "schema_summary": state.get("schema_summary"),
    }


WORKSPACE_FOLDERS = [
    "analysis", "data/cleaned", "data/featured",
    "modeling/splits", "models", "reports", "errors",
]

PHASES_BEFORE_TRAIN = [
    "analyze", "eda", "feature_engineering",
    "model_selection", "split", "tune_models",
]

PHASES_AFTER_TRAIN = ["evaluate", "persist", "report", "business_interpretation"]


def run_pipeline(dataset_path, business_task, output_root="artifacts"):
    load_dotenv()

    llm = ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=4000,
    )

    dataset = os.path.abspath(os.path.expanduser(str(dataset_path)))
    if not os.path.isfile(dataset):
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = os.path.abspath(output_root)
    workspace = os.path.join(root, "runs", run_id)

    for folder in WORKSPACE_FOLDERS:
        os.makedirs(os.path.join(workspace, folder), exist_ok=True)
    os.makedirs(os.path.join(root, "runs", "memory"), exist_ok=True)

    total_phases = len(PHASES_BEFORE_TRAIN) + 1 + len(PHASES_AFTER_TRAIN)
    print_run_header(run_id, dataset, workspace, business_task)

    state = {
        "business_task": business_task,
        "dataset_path": str(dataset),
        "workspace_dir": str(workspace),
        "run_id": run_id,
        "task_type": None,
        "target_column": None,
        "schema_summary": None,
        "selected_models": [],
        "best_model_name": None,
        "phase_outputs": {},
    }

    all_tools = get_all_tools()
    saver = MemorySaver()

    phase_no = 0
    for phase in PHASES_BEFORE_TRAIN:
        phase_no += 1
        print_phase_intro(phase, phase_no, total_phases)
        state = run_phase(state, phase, llm, all_tools, saver)

    task_type = state.get("task_type") or "regression"
    train_phase = f"train_{task_type}"
    phase_no += 1
    print_phase_intro(train_phase, phase_no, total_phases)
    state = run_phase(state, train_phase, llm, all_tools, saver)

    for phase in PHASES_AFTER_TRAIN:
        phase_no += 1
        print_phase_intro(phase, phase_no, total_phases)
        state = run_phase(state, phase, llm, all_tools, saver)

    report_path = os.path.join(workspace, "reports", "run_report.md")
    if os.path.isfile(report_path):
        state["report_path"] = report_path
    biz_path = os.path.join(workspace, "reports", "business_interpretation.md")
    if os.path.isfile(biz_path):
        state["business_report_path"] = biz_path

    print_run_footer(state)
    print_json_summary_block({
        "run_id": state.get("run_id"),
        "task_type": state.get("task_type"),
        "target_column": state.get("target_column"),
        "best_model": state.get("best_model_name"),
        "report": state.get("report_path"),
        "business_interpretation": state.get("business_report_path"),
        "workspace_dir": state.get("workspace_dir"),
    })
    return state


def run_phase(state, phase, llm, all_tools, saver):
    allowed = set(PHASE_TOOLS.get(phase, []))
    tools = [t for t in all_tools if getattr(t, "name", "") in allowed]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=get_system_prompt(state, phase),
        checkpointer=saver,
        state_schema=MyAgentState,
    )

    result = agent.invoke(
        build_graph_input(state, get_phase_message(state, phase)),
        config={"configurable": {"thread_id": state["run_id"] + "_" + phase}},
    )

    print_phase_transcript(result.get("messages", []))

    state = refresh_state(state)

    summary = ""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            summary = content.strip()
            break

    state["phase_outputs"][phase] = summary
    print_phase_outro(phase, summary, state)
    return state


def refresh_state(state):
    ws = state["workspace_dir"]
    state = dict(state)

    goal = read_json(os.path.join(ws, "analysis", "modeling_goal.json"), default={})
    schema = read_json(os.path.join(ws, "analysis", "schema_snapshot.json"), default={})
    model_plan = read_json(os.path.join(ws, "modeling", "model_plan.json"), default={})
    evaluation = read_json(os.path.join(ws, "modeling", "evaluation.json"), default={})

    state["task_type"] = goal.get("task_type") or state.get("task_type")
    state["target_column"] = goal.get("target_column") or state.get("target_column")
    state["selected_models"] = model_plan.get("selected_models") or state.get("selected_models", [])
    state["best_model_name"] = evaluation.get("best_model_name") or state.get("best_model_name")
    state["schema_summary"] = schema.get("summary") or state.get("schema_summary")

    return state
