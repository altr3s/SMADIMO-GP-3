import textwrap
from typing import Any, Sequence

from prompts import PHASE_TOOLS

W = 78

PHASE_USER_RU: dict[str, tuple[str, str]] = {
    "analyze": (
        "Разбор данных и подготовка",
        "Агент читает датасет, фиксирует схему и роли колонок, задаёт цель (таргет и тип задачи), "
        "смотрит распределения и выполняет очистку: пропуски, выбросы, лишние колонки.",
    ),
    "eda": (
        "Исследовательский анализ (EDA)",
        "Строится отчёт о пропусках, корреляциях с таргетом, выбросах и возможных утечках данных.",
    ),
    "feature_engineering": (
        "Инженерия признаков",
        "Поверх очищенных данных создаются новые признаки (отношения, агрегаты по тексту, даты и т.д.).",
    ),
    "model_selection": (
        "Выбор моделей",
        "Формируется пул алгоритмов под тип задачи (классификация / регрессия / кластеризация).",
    ),
    "split": (
        "Разбиение на train / validation / test",
        "Данные делятся на обучающую, валидационную и тестовую выборки для честной оценки.",
    ),
    "tune_models": (
        "Подбор гиперпараметров",
        "Для каждой выбранной модели перебираются конфигурации; лучшие параметры сохраняются.",
    ),
    "train_classification": (
        "Обучение моделей (классификация)",
        "Модели обучаются на train, качество смотрится на validation; результаты ранжируются.",
    ),
    "train_regression": (
        "Обучение моделей (регрессия)",
        "Модели обучаются на train, качество смотрится на validation; результаты ранжируются.",
    ),
    "train_clustering": (
        "Обучение моделей (кластеризация)",
        "Кластерные модели подгоняются под данные; метрики кластерного качества считаются на train.",
    ),
    "evaluate": (
        "Best: оценка лучшей модели на тесте",
        "Выбирается best-модель по validation; она переобучается на train+val и проверяется на отложенном test — финальные метрики.",
    ),
    "persist": (
        "Сравнение с прошлыми запусками",
        "Текущий результат сравнивается с «долговременной памятью»; при улучшении модель сохраняется.",
    ),
    "report": (
        "Итоговый отчёт и бизнес-документ",
        "Технический run_report, затем факты из прогона и отдельный markdown для ЛПР (выводы, риски, рекомендации простым языком).",
    ),
}


def _line(ch: str = "─") -> str:
    return ch * (W - 2)


def _box_top(title: str = "") -> None:
    if title:
        inner = f" {title.strip()} "
        max_inner = W - 4
        if len(inner) > max_inner:
            inner = inner[: max_inner - 1] + "… "
        pad = max(0, W - 4 - len(inner))
        left, right = pad // 2, pad - pad // 2
        print(f"┌{'─' * left}{inner}{'─' * right}┐")
    else:
        print(f"┌{_line()}┐")


def _box_mid(text: str = "") -> None:
    if not text:
        print(f"│{' ' * (W - 2)}│")
        return
    for raw in text.splitlines() or [""]:
        for chunk in textwrap.wrap(raw, width=W - 4) or [""]:
            pad = W - 4 - len(chunk)
            print(f"│ {chunk}{' ' * pad} │")


def _box_bottom() -> None:
    print(f"└{_line()}┘")


def _section(title: str, body: str = "") -> None:
    _box_top(title)
    if body.strip():
        _box_mid(body.rstrip())
    _box_bottom()
    print()


def _truncate(s: Any, limit: int = 600) -> str:
    t = s if isinstance(s, str) else repr(s)
    t = t.replace("\r\n", "\n").strip()
    if len(t) <= limit:
        return t
    return t[: limit - 3] + "..."


def print_run_header(
    run_id: str,
    dataset_path: str,
    workspace: str,
    business_task: str,
) -> None:
    print()
    _box_top(" Запуск ML-агента ")
    _box_mid("")
    _box_mid(f"Идентификатор запуска: {run_id}")
    _box_mid(f"Датасет: {dataset_path}")
    _box_mid(f"Рабочая папка (все артефакты): {workspace}")
    _box_bottom()
    print()
    _section("Бизнес-задача", business_task)
    _section(
        "Как читать вывод",
        "Ниже по шагам показано, на каком этапе пайплайна вы находитесь, какие инструменты доступны "
        "агенту и что он вызывает. После каждого этапа — краткая сводка от модели.",
    )


def print_phase_intro(phase: str, phase_index: int, total_phases: int) -> None:
    title_ru, desc = PHASE_USER_RU.get(phase, (phase, "Выполнение этапа пайплайна."))
    tools = PHASE_TOOLS.get(phase, [])
    tools_line = ", ".join(tools) if tools else "(инструментов нет — проверьте конфигурацию)"

    _box_top(f" Этап {phase_index}/{total_phases}: {title_ru} ")
    _box_mid("")
    _box_mid(f"Внутреннее имя фазы: {phase}")
    _box_mid("")
    _box_mid(desc)
    _box_mid("")
    _box_mid("Доступные инструменты на этом шаге:")
    for t in tools:
        _box_mid(f"  • {t}")
    if not tools:
        _box_mid(f"  • {tools_line}")
    _box_bottom()
    print()
    print("  ▸ Агент думает и вызывает инструменты…")
    print()


def print_phase_transcript(messages: Sequence[Any]) -> None:
    try:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    except ImportError:
        return

    blocks: list[tuple[str, str]] = []
    for msg in messages or []:
        if isinstance(msg, HumanMessage):
            c = _truncate(msg.content, 800)
            if c:
                blocks.append(("Запрос к агенту", c))
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, list):
                content = " ".join(str(x) for x in content)
            if isinstance(content, str) and content.strip():
                blocks.append(("Ответ агента (текст)", _truncate(content, 1200)))
            for tc in getattr(msg, "tool_calls", None) or []:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name", "?")
                args = tc.get("args", tc.get("arguments", {}))
                arg_s = _truncate(args if isinstance(args, str) else repr(args), 900)
                blocks.append((f"Вызов инструмента: {name}", arg_s))
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            blocks.append((f"Результат: {name}", _truncate(msg.content, 1200)))

    if not blocks:
        print("  (сообщений для отображения нет)")
        print()
        return

    print("  ─── Ход работы агента (сообщения шага) ───")
    print()
    for i, (head, body) in enumerate(blocks, 1):
        print(f"  [{i:02d}] {head}")
        for ln in body.splitlines() or [body]:
            print(f"       {ln}")
        print()
    print("  ─────────────────────────────────────────")
    print()


def print_phase_outro(phase: str, summary: str, state: dict[str, Any]) -> None:
    snap = []
    if state.get("target_column"):
        snap.append(f"Таргет: {state['target_column']}")
    if state.get("task_type"):
        snap.append(f"Тип задачи: {state['task_type']}")
    if state.get("schema_summary"):
        snap.append(f"Схема: {_truncate(state['schema_summary'], 200)}")
    if state.get("best_model_name"):
        snap.append(f"Лучшая модель (на данный момент): {state['best_model_name']}")

    body = "\n".join(snap) if snap else "Состояние обновится по мере появления артефактов."
    _section(f"Этап «{phase}» завершён", f"Краткая сводка от агента:\n{summary or '(пусто)'}\n\nТекущее состояние:\n{body}")


def print_run_footer(state: dict[str, Any]) -> None:
    ws = state.get("workspace_dir", "")
    report = state.get("report_path")
    lines = [
        f"Идентификатор: {state.get('run_id')}",
        f"Таргет: {state.get('target_column')}",
        f"Тип задачи: {state.get('task_type')}",
        f"Лучшая модель: {state.get('best_model_name')}",
        "",
        f"Папка запуска: {ws}",
    ]
    if report:
        lines.append(f"Отчёт: {report}")
    biz = state.get("business_report_path")
    if biz:
        lines.append(f"Бизнес-интерпретация: {biz}")
    if state.get("phase_outputs"):
        lines.append("")
        lines.append("Сводки по этапам:")
        for ph, s in state["phase_outputs"].items():
            lines.append(f"  • {ph}: {_truncate(s, 160)}")

    _section("Пайплайн завершён", "\n".join(lines))


def print_json_summary_block(payload: dict[str, Any]) -> None:
    import json

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    _box_top(" Итог (JSON для скриптов) ")
    for ln in text.splitlines():
        pad = W - 4 - len(ln)
        if pad < 0:
            for chunk in textwrap.wrap(ln, width=W - 4):
                print(f"│ {chunk}{' ' * (W - 4 - len(chunk))} │")
        else:
            print(f"│ {ln}{' ' * pad} │")
    _box_bottom()
    print()
