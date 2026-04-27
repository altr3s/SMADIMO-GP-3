from pathlib import Path

from agent import run_pipeline


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    dataset = root / "data" / "apartments_for_rent_classified_10K.csv"
    run_pipeline(
        dataset_path=str(dataset),
        business_task=(
            "У нас есть датасет с данными об аренде квартир. Помоги нам понять, какие факторы влияют на стоимость аренды, "
            "и предскажи стоимость аренды для новых объявлений."
        ),
    )


if __name__ == "__main__":
    main()
