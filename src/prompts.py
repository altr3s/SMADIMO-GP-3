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
    "report": ["write_report"],
    "business_interpretation": ["collect_pipeline_highlights", "write_business_report"],
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
Нужно выбрать минимум 2 модели и обосновать выбор.
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
Выбери разумное небольшое пространство поиска для уже выбранных моделей — обычно достаточно 3-8 конфигураций на модель.
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
Сформируй итоговый отчет.
Обязательно вызови `write_report`.
После вызова инструментов дай пользователю итог на русском обычным текстом (абзацы и маркированные списки), без таблиц в markdown.
""".strip(),

    "business_interpretation": """
Этап 11. Бизнес-интерпретация (финальный шаг для ЛПР).
1. Сначала вызови `collect_pipeline_highlights` — получишь сжатые факты из всех ключевых артефактов прогона.
2. На основе бизнес-задачи, этих фактов и своего понимания пайплайна подготовь отчёт для бизнеса: обычным языком, без жаргона там, где можно; структура с заголовками ## в markdown допустима.
   Включи: напоминание цели; что сделано с данными (очень кратко); какая модель выбрана и насколько она «хороша» в прикладных терминах; ограничения и риски; 3–7 конкретных рекомендаций для бизнеса.
3. Вызови `write_business_report(markdown_report="...")` с полным текстом отчёта на русском. Без markdown-таблиц.
4. Пользователю в конце дай краткую сводку на русском: где лежит файл и в чём суть для бизнеса.
""".strip(),
}


def get_system_prompt(state, phase):
    task_type = state.get("task_type") or "не определен"
    target = state.get("target_column") or "не определен"
    schema = state.get("schema_summary") or "Схема датасета пока не зафиксирована."

    return f"""
Ты автономный ML-агент.
Твоя роль: опытный ML-инженер и аналитик, который принимает решения строго в рамках бизнес-задачи.

Бизнес-задача:
{state.get("business_task", "")}

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
- Допустимы общепринятые обозначения метрик и имён моделей латиницей (RMSE, R², ROC-AUC, random_forest_regressor и т.п.).

Формат ответа пользователю (обязательно):
- Пиши связным текстом: короткие абзацы, при необходимости — маркированные или нумерованные списки.
- Не используй markdown-таблицы (строки с символами `|`, разделители `|---|---|` и т.п.). Не оформляй данные «как таблицу» в чате.
- Числа, метрики и сравнения встраивай в предложения («RMSE на валидации — …, на тесте — …») или в виде списка «- модель A: …».
- Если нужно перечислить много пар «имя — значение», используй тире в строках списка, а не колонки таблицы.

Техники рассуждения (внутренне применяй, пользователю не копируй дословно):
1. Роль: веди себя как опытный ML-инженер.
2. Контраст: отвергай решения с утечкой данных, случайным выбором таргета и неподходящими моделями.
3. Самопроверка: перед завершением этапа убедись, что выполнены все обязательные действия.

Инструкция для текущей фазы:
{PHASE_INSTRUCTIONS.get(phase, "Выполни задачу текущей фазы.")}
""".strip()


def get_phase_message(state, phase):
    lines = [
        f"Фаза: {phase}",
        f"Бизнес-задача: {state['business_task']}",
        f"Исходный датасет: {state['dataset_path']}",
        f"Рабочая директория: {state['workspace_dir']}",
        f"Определенный target: {state.get('target_column') or 'не определен'}",
        f"Определенный тип задачи: {state.get('task_type') or 'не определен'}",
        f"Актуальная схема: {state.get('schema_summary') or 'не зафиксирована'}",
    ]

    if state.get("phase_outputs"):
        lines.append("Краткая память по предыдущим этапам:")
        for prev_phase, summary in state["phase_outputs"].items():
            if prev_phase != phase:
                lines.append(f"- {prev_phase}: {summary}")

    lines.append(
        "Верни краткую сводку результата после вызова нужных инструментов. "
        "Сводка на русском языке, обычным текстом и списками, без markdown-таблиц."
    )
    return "\n".join(lines)
