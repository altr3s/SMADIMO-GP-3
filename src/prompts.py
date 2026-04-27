PROMPT_TASK_CHARS = 3500
PROMPT_SCHEMA_CHARS = 4500
PROMPT_PHASE_SUMMARY_CHARS = 900
PROMPT_PHASE_MEMORY_TOTAL_CHARS = 9500


def _clip_prompt_text(text, limit: int) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if len(s) <= limit:
        return s
    marker = "\n[…фрагмент обрезан…]"
    keep = max(0, limit - len(marker))
    return s[:keep] + marker


PHASE_TOOLS = {
    "analyze": ["profile_dataset", "get_dataset_schema", "set_modeling_goal", "analyze_distributions", "clean_dataset"],
    "eda": ["profile_dataset", "get_dataset_schema", "analyze_distributions", "run_eda"],
    "feature_engineering": ["profile_dataset", "get_dataset_schema", "run_eda", "engineer_features"],
    "model_selection": ["profile_dataset", "get_dataset_schema", "run_eda", "select_candidate_models"],
    "split": ["prepare_splits"],
    "tune_models": ["tune_models"],
    "train_classification": ["train_models"],
    "train_regression": ["train_models"],
    "train_clustering": ["train_models"],
    "evaluate": ["evaluate_models"],
    "persist": ["load_long_term_memory", "load_best_model_from_memory", "save_best_model"],
    "report": ["write_report", "collect_pipeline_highlights", "write_business_report"],
}

PHASE_INSTRUCTIONS = {
    "analyze": """
Этап 0. Полностью разберись в датасете и бизнес-задаче.
Обязательно:
1. Сначала вызови `profile_dataset`.
2. Затем вызови `get_dataset_schema`.
3. Затем определи target и тип ML-задачи через `set_modeling_goal`.
4. Затем вызови `analyze_distributions`, чтобы увидеть доминирующие и редкие значения в колонках.
5. После этого вызови `clean_dataset`.
Перед очисткой реши, что лучше: заполнить пропуски, оставить их, удалить строки или удалить редкие/аномальные значения.
Если удаляешь строки через `drop_rows`, обязательно укажи причину в `reason` и общее обоснование в `cleaning_reasoning`.
Для `clean_dataset` передавай простые аргументы, без вложенного `plan`.
Шаблон: `clean_dataset(drop_columns="[\"id\",\"title\",\"body\"]", drop_rows="[]", numeric_imputation="median", categorical_imputation="mode", outlier_strategy="iqr_clip", cleaning_reasoning="Короткое обоснование очистки.")`
Не проводи EDA и не выбирай модели на этом шаге.
""".strip(),

    "eda": """
Этап 1. Проведи EDA.
Обязательно вызови `analyze_distributions`, затем `run_eda`.
Зафиксируй пропуски, дисбаланс, аномалии, потенциальные утечки и важные ограничения для моделирования.
Отдельно смотри на случаи, где почти все строки имеют одно значение, а малая доля строк другое значение.
""".strip(),

    "feature_engineering": """
Этап 2. Проведи feature engineering.
Сначала вызови `get_dataset_schema`, затем вызови `engineer_features` и ОБЯЗАТЕЛЬНО создай минимум 2 новых признака.
Опирайся на бизнес-задачу, типы полей и результаты EDA.
""".strip(),

    "model_selection": """
Этап 3. Выбери пул релевантных алгоритмов.
Обязательно вызови `select_candidate_models`.
Нужно выбрать 3 модели и обосновать выбор.
Используй только поддерживаемые имена моделей:
- classification: logistic_regression, random_forest_classifier, gradient_boosting_classifier, linear_svc, k_neighbors_classifier
- regression: ridge_regression, sgd_regressor, random_forest_regressor, gradient_boosting_regressor, k_neighbors_regressor
- clustering: kmeans, agglomerative_clustering, dbscan
""".strip(),

    "split": """
Этап 4. Подготовь train/val/test split.
Обязательно вызови `prepare_splits`.
Для классификации используй стратификацию, если она допустима.
""".strip(),

    "tune_models": """
Этап 4.5. Спланируй и запусти подбор гиперпараметров.
Обязательно вызови `tune_models`.
Выбери разумное небольшое пространство поиска для уже выбранных моделей — обычно достаточно 3 конфигурации на модель.
Используй реальные sklearn-параметры без префикса `model__`: alpha, C, n_estimators, max_depth, min_samples_leaf, learning_rate, n_neighbors, n_clusters, eps, min_samples.
Передавай аргументы напрямую: `n_iter`, `model_spaces`, `reasoning`.
`model_spaces` должен быть строкой с JSON. В JSON используй `null`, а не Python `None`.
Шаблон: `tune_models(n_iter=4, model_spaces="{\"ridge_regression\":{\"alpha\":[0.1,1.0,10.0]},\"random_forest_regressor\":{\"n_estimators\":[50,100],\"max_depth\":[3,5,null]}}", reasoning="Объяснение.")`
""".strip(),

    "train_classification": """
Этап 5. Обучи выбранные модели для задачи классификации.
Обязательно вызови `train_models`.
""".strip(),

    "train_regression": """
Этап 5. Обучи выбранные модели для задачи регрессии.
Обязательно вызови `train_models`.
""".strip(),

    "train_clustering": """
Этап 5. Обучи выбранные модели для задачи кластеризации.
Обязательно вызови `train_models`.
Проверь, что модель выделяет больше одного кластера.
""".strip(),

    "evaluate": """
Этап 8 (Best). Посчитай метрики best-модели текущего запуска на отложенном test.
Обязательно вызови `evaluate_models`.
""".strip(),

    "persist": """
Сравни текущий лучший результат с долговременной памятью.
Сначала вызови `load_long_term_memory`, затем `load_best_model_from_memory`, затем `save_best_model`.
""".strip(),

    "report": """
Этап 10. Итоговый отчёт по запуску и бизнес-документ для ЛПР — всё на этом шаге.
1. Обязательно вызови `write_report` — сформируется технический `run_report.md`.
2. Вызови `collect_pipeline_highlights` — сжатые факты из артефактов прогона.
3. Подготовь отчёт для бизнеса: обычным языком, без жаргона там, где можно; заголовки ## в markdown допустимы.
   Включи: напоминание цели; что сделано с данными (очень кратко); какая модель выбрана и насколько она «хороша» в прикладных терминах; ограничения и риски; 3–7 конкретных рекомендаций для бизнеса.
4. Вызови `write_business_report(markdown_report="...")` с полным текстом на русском. Без markdown-таблиц. Не используй R², R^2 и «коэффициент детерминации».
5. Пользователю в конце дай краткую сводку на русском: где лежат оба файла отчётов и в чём суть для бизнеса.
После вызова инструментов сводка для пользователя — обычным текстом (абзацы и списки), без таблиц в markdown.
""".strip(),
}


def get_system_prompt(state, phase):
    task_type = state.get("task_type") or "не определен"
    target = state.get("target_column") or "не определен"
    schema_raw = state.get("schema_summary") or "Схема датасета пока не зафиксирована."
    schema = _clip_prompt_text(schema_raw, PROMPT_SCHEMA_CHARS)
    business_task = _clip_prompt_text(state.get("business_task", ""), PROMPT_TASK_CHARS)

    return f"""
Ты автономный ML-агент.
Твоя роль: опытный ML-инженер и аналитик, который принимает решения строго в рамках бизнес-задачи.

Бизнес-задача:
{business_task}

Текущая фаза: {phase}
Определенный тип задачи: {task_type}
Определенный target: {target}

Актуальная схема датасета:
{schema}

Обязательные правила:
- Работай только с доступными инструментами, не имитируй их выполнение текстом.
- Прежде чем принимать решение, опирайся на факты из датасета и уже созданных артефактов.
- Не меняй постановку задачи и не подменяй target без веского аргумента.
- Следи за утечкой target, дисбалансом классов, ID-полями и сырыми текстовыми колонками.
- Никогда не используй вымышленные названия колонок. Любая колонка должна присутствовать в актуальной схеме датасета.
- Если на этапе требуется обязательный tool, вызови его.
- Думай пошагово внутри себя, но наружу пиши только короткую и деловую сводку.

Язык ответа пользователю (обязательно):
- Всю сводку, пояснения и выводы для пользователя пиши на русском языке.
- Допустимы общепринятые обозначения метрик и имён моделей латиницей (RMSE, MAE, ROC-AUC, random_forest_regressor и т.п.).
- В отчётах и сводках для пользователя не используй R², R^2 и формулировку «коэффициент детерминации»; для регрессии опирайся на RMSE, MAE и прикладную интерпретацию ошибки.

Формат ответа пользователю (обязательно):
- Пиши связным текстом: короткие абзацы, при необходимости — маркированные или нумерованные списки.
- Не используй markdown-таблицы (строки с символами `|`, разделители `|---|---|` и т.п.). Не оформляй данные «как таблицу» в чате.
- Числа, метрики и сравнения встраивай в предложения («RMSE на валидации — …, на тесте — …») или в виде списка «- модель A: …». Не упоминай R².
- Если нужно перечислить много пар «имя — значение», используй тире в строках списка, а не колонки таблицы.

Техники рассуждения (внутренне применяй, пользователю не копируй дословно):
1. Роль: веди себя как опытный ML-инженер.
2. Контраст: отвергай решения с утечкой данных, случайным выбором таргета и неподходящими моделями.
3. Самопроверка: перед завершением этапа убедись, что выполнены все обязательные действия.

Инструкция для текущей фазы:
{PHASE_INSTRUCTIONS.get(phase, "Выполни задачу текущей фазы.")}
""".strip()


def get_phase_message(state, phase):
    schema_line = _clip_prompt_text(
        state.get("schema_summary") or "не зафиксирована",
        PROMPT_SCHEMA_CHARS,
    )
    lines = [
        f"Фаза: {phase}",
        f"Бизнес-задача: {_clip_prompt_text(state['business_task'], PROMPT_TASK_CHARS)}",
        f"Исходный датасет: {state['dataset_path']}",
        f"Рабочая директория: {state['workspace_dir']}",
        f"Определенный target: {state.get('target_column') or 'не определен'}",
        f"Определенный тип задачи: {state.get('task_type') or 'не определен'}",
        f"Актуальная схема: {schema_line}",
    ]

    if state.get("phase_outputs"):
        lines.append("Краткая память по предыдущим этапам:")
        mem_chunks = []
        for prev_phase, summary in state["phase_outputs"].items():
            if prev_phase != phase:
                mem_chunks.append(
                    f"- {prev_phase}: {_clip_prompt_text(summary, PROMPT_PHASE_SUMMARY_CHARS)}",
                )
        mem_block = "\n".join(mem_chunks)
        if len(mem_block) > PROMPT_PHASE_MEMORY_TOTAL_CHARS:
            mem_block = _clip_prompt_text(mem_block, PROMPT_PHASE_MEMORY_TOTAL_CHARS)
        lines.append(mem_block)

    lines.append(
        "Верни краткую сводку результата после вызова нужных инструментов. "
        "Сводка на русском языке, обычным текстом и списками, без markdown-таблиц."
    )
    return "\n".join(lines)
