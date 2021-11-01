CREATE DATABASE mlflow;

\c mlflow;

CREATE TABLE IF NOT EXISTS load_curves(
	run_id VARCHAR PRIMARY KEY NOT NULL,
	test_data_points JSON NOT NULL,
	load_curve JSON NOT NULL
);
