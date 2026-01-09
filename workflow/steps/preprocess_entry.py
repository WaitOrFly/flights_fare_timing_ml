from preprocess import preprocess


def preprocess_entry(
    input_data_s3_uri: str,
    output_data_s3_uri: str,
    experiment_name: str = "main_experiment",
    run_name: str = "run-01",
):
    return preprocess(
        input_data_s3_uri=input_data_s3_uri,
        output_data_s3_uri=output_data_s3_uri,
        experiment_name=experiment_name,
        run_name=run_name,
    )
