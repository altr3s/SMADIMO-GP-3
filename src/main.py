from agent import run_pipeline


def main() -> None:
    run_pipeline(
        dataset_path = "/Users/leonidprokopev/projects/SMADIMO-GP-3/data/apartments_for_rent_classified_10K.csv",
        business_task="У нас есть датасет с данными об аренде квартир. Помоги нам понять, какие факторы влияют на стоимость аренды, и предскажи стоимость аренды для новых объявлений."
    )

if __name__ == "__main__":
    main()