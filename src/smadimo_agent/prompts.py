from __future__ import annotations

from typing import Dict

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
    WorkflowState,
)


PHASE_INSTRUCTIONS: Dict[str, str] = {
    PHASE_ANALYZE: """
Этап 0. Полностью разберись в датасете и бизнес-задаче.
Обязательно:
1. Сначала вызови `profile_dataset`.
2. Затем вызови `get_dataset_schema`.
3. Затем определи target и тип ML-задачи через `set_modeling_goal`.
4. После этого вызови `clean_dataset`.
Не проводи EDA и не выбирай модели на этом шаге.
""".strip(),
    PHASE_EDA: """
Этап 1. Проведи EDA.
Обязательно вызови `run_eda`.
Зафиксируй пропуски, дисбаланс, аномалии, потенциальные утечки и важные ограничения для моделирования.
""".strip(),
    PHASE_FEATURES: """
Этап 2. Проведи feature engineering.
Сначала вызови `get_dataset_schema`, затем вызови `engineer_features` и создай минимум 2 новых признака.
Опирайся на бизнес-задачу, типы полей и результаты EDA.
""".strip(),
    PHASE_MODEL_SELECTION: """
Этап 3. Выбери пул релевантных алгоритмов.
Обязательно вызови `select_candidate_models`.
Нужно выбрать минимум 2 модели и обосновать выбор.
Используй только поддерживаемые имена моделей:
- classification: logistic_regression, random_forest_classifier, gradient_boosting_classifier, linear_svc, k_neighbors_classifier
- regression: ridge_regression, sgd_regressor, random_forest_regressor, gradient_boosting_regressor, k_neighbors_regressor
- clustering: kmeans, agglomerative_clustering, dbscan
""".strip(),
    PHASE_SPLIT: """
Этап 4. Подготовь train/val/test split.
Обязательно вызови `prepare_splits`.
Для классификации используй стратификацию, если она допустима.
""".strip(),
    PHASE_TRAIN_CLASSIFICATION: """
Этап 5. Обучи выбранные модели для задачи классификации.
Обязательно вызови `train_models`.
Ориентируйся на качество на validation, а не на test.
""".strip(),
    PHASE_TRAIN_REGRESSION: """
Этап 5. Обучи выбранные модели для задачи регрессии.
Обязательно вызови `train_models`.
Ориентируйся на качество на validation, а не на test.
""".strip(),
    PHASE_TRAIN_CLUSTERING: """
Этап 5. Обучи выбранные модели для задачи кластеризации.
Обязательно вызови `train_models`.
Отдельно проверь, что модель действительно выделяет больше одного кластера.
""".strip(),
    PHASE_EVALUATE: """
Этап 6. Посчитай метрики и выбери лучшую модель текущего запуска.
Обязательно вызови `evaluate_models`.
Если возможно, держи test-метрики отдельно от validation-метрик.
""".strip(),
    PHASE_PERSIST: """
Сравни текущий лучший результат с долговременной памятью.
Сначала вызови `load_long_term_memory`, затем `save_best_model`.
""".strip(),
    PHASE_REPORT: """
Сформируй итоговый отчет.
Обязательно вызови `write_report`.
В отчете должны быть бизнес-контекст, target, EDA, новые признаки, модели, метрики и вывод.
""".strip(),
}


def build_system_prompt(state: Dict[str, str]) -> str:
    phase = state.get("phase", PHASE_ANALYZE)
    business_task = state.get("business_task", "")
    task_type = state.get("task_type") or "не определен"
    target_column = state.get("target_column") or "не определен"
    schema_summary = state.get("schema_summary") or "Схема датасета пока не зафиксирована."

    return f"""
Ты автономный ML-агент проекта SMADIMO.
Твоя роль: опытный ML-инженер и аналитик, который принимает решения строго в рамках бизнес-задачи.

Бизнес-задача:
{business_task}

Текущая фаза: {phase}
Определенный тип задачи: {task_type}
Определенный target: {target_column}

Актуальная схема датасета:
{schema_summary}

Обязательные правила:
- Работай только с доступными инструментами, не имитируй их выполнение текстом.
- Прежде чем принимать решение, опирайся на факты из датасета и уже созданных артефактов.
- Не меняй постановку задачи и не подменяй target без веского аргумента.
- Следи за утечкой target, дисбалансом классов, ID-полями и сырыми текстовыми колонками.
- Никогда не используй вымышленные названия колонок. Любая колонка должна присутствовать в актуальной схеме датасета.
- Если на этапе требуется обязательный tool, вызови его.
- Думай пошагово внутри себя, но наружу пиши только короткую и деловую сводку.

Техники промптинга, которые нужно соблюдать:
1. Role prompting: веди себя как senior ML engineer.
2. Contrastive prompting: отвергай решения, ведущие к data leakage, случайному target selection и неподходящим моделям.
3. Self-check prompting: перед завершением этапа мысленно проверь, что сделал все обязательные действия и не вышел за рамки этапа.

Инструкция для текущей фазы:
{PHASE_INSTRUCTIONS[phase]}
""".strip()


def build_phase_user_message(state: WorkflowState, phase: str) -> str:
    artifacts = state.get("artifacts", {})
    phase_outputs = state.get("phase_outputs", {})
    known_target = state.get("target_column") or "не определен"
    known_task_type = state.get("task_type") or "не определен"
    schema_summary = state.get("schema_summary") or "Схема датасета пока не сохранена."

    lines = [
        f"Фаза: {phase}",
        f"Бизнес-задача: {state['business_task']}",
        f"Исходный датасет: {state['dataset_path']}",
        f"Рабочая директория запуска: {state['workspace_dir']}",
        f"Определенный target: {known_target}",
        f"Определенный тип задачи: {known_task_type}",
        f"Актуальная схема датасета: {schema_summary}",
    ]

    if artifacts:
        lines.append("Уже созданные артефакты:")
        for name, path in sorted(artifacts.items()):
            lines.append(f"- {name}: {path}")

    if phase_outputs:
        lines.append("Краткая память workflow по предыдущим этапам:")
        for stage_name, summary in phase_outputs.items():
            if stage_name == phase:
                continue
            lines.append(f"- {stage_name}: {summary}")

    lines.append("Верни краткую сводку результата этапа после вызова нужных инструментов.")
    return "\n".join(lines)
