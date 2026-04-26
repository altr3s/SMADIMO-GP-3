import time
import traceback
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from openai import RateLimitError

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from smadimo_agent.io_utils import now_run_id, read_json, slugify, write_json
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
    PHASE_TUNE_MODELS,
    PHASE_TRAIN_CLASSIFICATION,
    PHASE_TRAIN_CLUSTERING,
    PHASE_TRAIN_REGRESSION,
    StageAgentState,
    TRAINING_PHASE_BY_TASK,
    WorkflowState,
    append_log,
    merge_phase_output,
)


def log(message):
    print(f"[agent] {message}", flush=True)


PHASE_TOOL_NAMES = {
    PHASE_ANALYZE: ["profile_dataset", "get_dataset_schema", "set_modeling_goal", "analyze_distributions", "clean_dataset"],
    PHASE_EDA: ["profile_dataset", "get_dataset_schema", "analyze_distributions", "run_eda"],
    PHASE_FEATURES: ["profile_dataset", "get_dataset_schema", "run_eda", "engineer_features"],
    PHASE_MODEL_SELECTION: ["profile_dataset", "get_dataset_schema", "run_eda", "select_candidate_models"],
    PHASE_SPLIT: ["prepare_splits"],
    PHASE_TUNE_MODELS: ["tune_models"],
    PHASE_TRAIN_CLASSIFICATION: ["train_models"],
    PHASE_TRAIN_REGRESSION: ["train_models"],
    PHASE_TRAIN_CLUSTERING: ["train_models"],
    PHASE_EVALUATE: ["evaluate_models"],
    PHASE_PERSIST: ["load_long_term_memory", "load_best_model_from_memory", "save_best_model"],
    PHASE_REPORT: ["write_report"],
}


TRAINING_PHASES = [
    PHASE_TRAIN_CLASSIFICATION,
    PHASE_TRAIN_REGRESSION,
    PHASE_TRAIN_CLUSTERING,
]


WORKSPACE_FOLDERS = [
    "analysis",
    "data",
    "data/cleaned",
    "data/featured",
    "modeling",
    "modeling/splits",
    "models",
    "reports",
]


def _required_paths(workspace_dir, phase):
    if phase == PHASE_ANALYZE:
        return [
            workspace_dir / "analysis" / "dataset_profile.json",
            workspace_dir / "analysis" / "schema_snapshot.json",
            workspace_dir / "analysis" / "modeling_goal.json",
            workspace_dir / "analysis" / "distribution_report.json",
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
    if phase == PHASE_TUNE_MODELS:
        return [workspace_dir / "modeling" / "hyperparameter_tuning.json"]
    if phase in TRAINING_PHASES:
        return [workspace_dir / "modeling" / "leaderboard.json"]
    if phase == PHASE_EVALUATE:
        evaluation_path = workspace_dir / "modeling" / "evaluation.json"
        evaluation = read_json(evaluation_path, default={})
        model_path = evaluation.get("current_best_model_path")
        if model_path:
            return [evaluation_path, Path(model_path)]
        return [evaluation_path, workspace_dir / "models" / "best_current_model.pkl"]
    if phase == PHASE_PERSIST:
        return [workspace_dir.parent / "memory" / "best_registry.json"]
    if phase == PHASE_REPORT:
        return [workspace_dir / "reports" / "run_report.md"]
    return []


def _extract_summary(result):
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


def _refresh_state_from_workspace(state):
    workspace_dir = Path(state["workspace_dir"])
    artifacts = collect_artifacts(workspace_dir)
    goal = read_json(workspace_dir / "analysis" / "modeling_goal.json", default={})
    schema_snapshot = read_json(workspace_dir / "analysis" / "schema_snapshot.json", default={})
    model_plan = read_json(workspace_dir / "modeling" / "model_plan.json", default={})
    evaluation = read_json(workspace_dir / "modeling" / "evaluation.json", default={})
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
    return refreshed


def _filter_tools(all_tools, phase):
    allowed = set(PHASE_TOOL_NAMES[phase])
    return [tool for tool in all_tools if getattr(tool, "name", "") in allowed]


def _phase_payload(state, phase, reminder):
    return {
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
    }


def _phase_prompt_state(state, phase):
    return {
        "phase": phase,
        "business_task": state["business_task"],
        "task_type": state.get("task_type"),
        "target_column": state.get("target_column"),
        "schema_summary": state.get("schema_summary"),
    }


def _short_trace(error):
    lines = traceback.format_exception(type(error), error, error.__traceback__)
    return "".join(lines[-8:])


def _guess_error_reason(error):
    message = str(error)
    lowered = message.lower()
    if "not found" in lowered and "column" in lowered:
        return "Возможная причина: агент сослался на столбец, которого нет в текущем датасете."
    if "unsupported" in lowered and "model" in lowered:
        return "Возможная причина: выбран неподдерживаемый алгоритм или имя модели записано не в ожидаемом формате."
    if "target column" in lowered:
        return "Возможная причина: target не выбран, выбран неверно или отсутствует в датасете."
    if "required artifacts" in lowered:
        return "Возможная причина: этап завершился без обязательных файлов, значит нужный tool не был вызван или упал внутри."
    if "no selected models" in lowered:
        return "Возможная причина: этап выбора моделей не сформировал список моделей."
    return "Причина не определена эвристически. Нужно сверить фазу, доступные tool'ы, схему датасета и последний traceback."


def _llm_error_explanation(config, phase, state, error):
    available_tools = ", ".join(PHASE_TOOL_NAMES.get(phase, []))
    prompt = (
        "Проанализируй ошибку ML-агента. Ответь кратко на русском: "
        "1) где возникла ошибка; 2) вероятная причина; 3) какой следующий tool или действие нужно выполнить. "
        "Не придумывай названия столбцов и tool'ов. "
        "Предлагай только tool из списка доступных для текущей фазы.\n\n"
        f"Фаза: {phase}\n"
        f"Доступные tools: {available_tools}\n"
        f"Бизнес-задача: {state.get('business_task')}\n"
        f"Датасет: {state.get('dataset_path')}\n"
        f"Target: {state.get('target_column')}\n"
        f"Тип задачи: {state.get('task_type')}\n"
        f"Схема: {state.get('schema_summary')}\n"
        f"Ошибка: {type(error).__name__}: {error}\n"
        f"Traceback:\n{_short_trace(error)}"
    )
    try:
        response = config.build_review_model().invoke(
            [
                SystemMessage(content="Ты помогаешь диагностировать ошибки пайплайна ML-агента."),
                HumanMessage(content=prompt),
            ]
        )
        content = getattr(response, "content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        pass
    return _guess_error_reason(error)


def _save_phase_error(workspace_dir, phase, error, explanation):
    error_dir = workspace_dir / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    path = error_dir / f"{phase}_error.json"
    write_json(
        path,
        {
            "phase": phase,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "diagnosis": explanation,
            "traceback": _short_trace(error),
        },
    )
    return path


def _invoke_agent_with_rate_limit_retry(agent, payload, thread_id, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            return agent.invoke(
                payload,
                config={"configurable": {"thread_id": thread_id}},
            )
        except RateLimitError as error:
            if attempt >= max_retries:
                raise
            time.sleep(65)

    raise RuntimeError("Agent invocation failed after rate-limit retries.")


def _run_phase_attempt(state, phase, config, saver, tools, reminder):
    agent = create_agent(
        model=config.build_primary_model(),
        tools=tools,
        system_prompt=build_system_prompt(_phase_prompt_state(state, phase)),
        state_schema=StageAgentState,
        checkpointer=saver,
    )
    return _invoke_agent_with_rate_limit_retry(
        agent=agent,
        payload=_phase_payload(state, phase, reminder),
        thread_id=state["thread_id"],
    )


def _retry_message_from_error(config, workspace_dir, phase, state, error):
    explanation = _llm_error_explanation(config, phase, state, error)
    error_path = _save_phase_error(workspace_dir, phase, error, explanation)
    message = (
        "\n\nНа предыдущей попытке возникла ошибка. "
        "Проанализируй диагностику, не повторяй неверный вызов tool и исправь действие.\n"
        f"Диагностика: {explanation}\n"
        f"Файл диагностики: {error_path}\n"
    )
    return message, explanation, error_path


def _retry_message_from_missing_files(phase, missing):
    guidance = {
        PHASE_ANALYZE: (
            "На этой фазе нельзя завершать работу текстом. "
            "Если нет modeling_goal.json, вызови `set_modeling_goal`. "
            "Если нет distribution_report.json, вызови `analyze_distributions`. "
            "Если нет cleaning_report.json или cleaned_dataset.csv, вызови `clean_dataset` "
            "с простыми аргументами без вложенного `plan`."
        ),
        PHASE_EDA: "Вызови `analyze_distributions`, затем `run_eda`.",
        PHASE_FEATURES: "Вызови `engineer_features`.",
        PHASE_MODEL_SELECTION: "Вызови `select_candidate_models`.",
        PHASE_SPLIT: "Вызови `prepare_splits`.",
        PHASE_TUNE_MODELS: (
            "Вызови `tune_models` с прямыми аргументами `n_iter`, `model_spaces`, `reasoning`. "
            "`model_spaces` передавай строкой с JSON по шаблону: "
            "'{\"ridge_regression\":{\"alpha\":[0.1,1.0,10.0]}}'. "
            "Не используй вложенный ключ `plan`, markdown или Python `None`."
        ),
        PHASE_EVALUATE: "Вызови `evaluate_models`.",
        PHASE_PERSIST: "Вызови `load_long_term_memory`, `load_best_model_from_memory`, затем `save_best_model`.",
        PHASE_REPORT: "Вызови `write_report`.",
    }
    return (
        "\n\nПовтори шаг еще раз. Не отвечай итоговым текстом, пока не вызовешь недостающие tools.\n"
        f"{guidance.get(phase, 'Вызови обязательный tool текущей фазы.')}\n"
        "На предыдущей попытке не были созданы обязательные артефакты:\n"
        + "\n".join(f"- {path}" for path in missing)
    )


def _finish_phase(state, phase, workspace_dir, result):
    refreshed = _refresh_state_from_workspace(state)
    summary = _extract_summary(result)
    log(f"Фаза `{phase}` завершена.")
    log(f"Краткий результат: {summary}")
    refreshed["phase"] = phase
    refreshed["phase_outputs"] = merge_phase_output(state, phase, summary)
    refreshed["execution_log"] = append_log(
        state,
        f"Phase `{phase}` completed with artifacts in {workspace_dir}.",
    )
    return refreshed


def _invoke_phase_agent(state, phase, config, saver, all_tools):
    workspace_dir = Path(state["workspace_dir"])
    tools = _filter_tools(all_tools, phase)
    tool_names = ", ".join(getattr(tool, "name", "") for tool in tools)

    reminder = ""
    max_attempts = 4
    for attempt in range(max_attempts):
        log(f"Фаза `{phase}`: попытка {attempt + 1}. Доступные tools: {tool_names}")
        try:
            result = _run_phase_attempt(state, phase, config, saver, tools, reminder)
        except Exception as error:
            reminder, explanation, error_path = _retry_message_from_error(
                config,
                workspace_dir,
                phase,
                state,
                error,
            )
            log(f"Фаза `{phase}` упала: {type(error).__name__}: {error}")
            log(f"Диагностика сохранена: {error_path}")
            if attempt >= 1:
                raise RuntimeError(
                    f"Phase `{phase}` failed after retry. Diagnosis saved to {error_path}. "
                    f"{explanation}"
                ) from error
            continue

        missing = [path for path in _required_paths(workspace_dir, phase) if not path.exists()]
        if not missing:
            return _finish_phase(state, phase, workspace_dir, result)

        log(f"Фаза `{phase}` не создала обязательные артефакты, повторяю.")
        state = _refresh_state_from_workspace(state)
        reminder = _retry_message_from_missing_files(phase, missing)

    error = RuntimeError(f"Phase `{phase}` finished without required artifacts.")
    explanation = _guess_error_reason(error)
    error_path = _save_phase_error(workspace_dir, phase, error, explanation)
    raise RuntimeError(
        f"Phase `{phase}` finished without required artifacts. Diagnosis saved to {error_path}. "
        f"{explanation}"
    )


def _bootstrap_node(config, dataset_path, business_task):
    def node(_):
        run_id = now_run_id() + "-" + slugify(dataset_path.stem)
        workspace_dir = config.output_root.resolve() / "runs" / run_id
        log(f"Старт запуска `{run_id}`.")
        log(f"Датасет: {dataset_path}")
        log(f"LLM: {config.model_name} ({config.base_url})")
        log(f"Артефакты: {workspace_dir}")
        folders = [workspace_dir / folder for folder in WORKSPACE_FOLDERS]
        folders.append(workspace_dir.parent / "memory")
        for directory in folders:
            directory.mkdir(parents=True, exist_ok=True)

        workflow_manifest = {
            "nodes": [
                "bootstrap",
                PHASE_ANALYZE,
                PHASE_EDA,
                PHASE_FEATURES,
                PHASE_MODEL_SELECTION,
                PHASE_SPLIT,
                PHASE_TUNE_MODELS,
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
                [PHASE_SPLIT, PHASE_TUNE_MODELS],
                [PHASE_TUNE_MODELS, "conditional_training_phase"],
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
            "dataset_path": str(dataset_path),
            "workspace_dir": str(workspace_dir),
            "run_id": run_id,
            "thread_id": run_id,
            "llm_model": config.model_name,
            "llm_base_url": config.base_url,
            "phase": "bootstrap",
            "artifacts": {"workflow_spec": str(workspace_dir / "workflow_spec.json")},
            "phase_outputs": {},
            "execution_log": [f"Bootstrap finished for run `{run_id}`."],
            "errors": [],
            "selected_models": [],
        }

    return node


def _make_phase_node(phase, config, saver, all_tools):
    def node(state):
        return _invoke_phase_agent(state, phase, config, saver, all_tools)

    return node


def _route_training_phase(state):
    task_type = state.get("task_type")
    if task_type not in TRAINING_PHASE_BY_TASK:
        raise ValueError(
            "Task type was not defined after analysis. Expected one of "
            f"{sorted(TRAINING_PHASE_BY_TASK.keys())}, got {task_type!r}."
        )
    return TRAINING_PHASE_BY_TASK[task_type]


def build_workflow(config, dataset_path, business_task):
    dataset = Path(dataset_path).expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    saver = InMemorySaver()
    all_tools = build_ml_tools()
    graph = StateGraph(WorkflowState)

    graph.add_node("bootstrap", _bootstrap_node(config, dataset, business_task))
    for phase in PHASE_TOOL_NAMES:
        graph.add_node(phase, _make_phase_node(phase, config, saver, all_tools))

    graph.add_edge(START, "bootstrap")
    graph.add_edge("bootstrap", PHASE_ANALYZE)
    graph.add_edge(PHASE_ANALYZE, PHASE_EDA)
    graph.add_edge(PHASE_EDA, PHASE_FEATURES)
    graph.add_edge(PHASE_FEATURES, PHASE_MODEL_SELECTION)
    graph.add_edge(PHASE_MODEL_SELECTION, PHASE_SPLIT)
    graph.add_edge(PHASE_SPLIT, PHASE_TUNE_MODELS)
    graph.add_conditional_edges(
        PHASE_TUNE_MODELS,
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


def run_pipeline(dataset_path, business_task, config):
    app = build_workflow(config, dataset_path=dataset_path, business_task=business_task)
    result = app.invoke({})
    log(f"Pipeline завершён. Отчёт: {result.get('report_path')}")
    return result
