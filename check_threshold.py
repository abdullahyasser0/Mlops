import sys
import mlflow
import dagshub


def main():
    dagshub.init(repo_owner="abdullahyasser0", repo_name="Mlops", mlflow=True)

    run_id = open("model_info.txt").read().strip()
    print(f"Checking accuracy for Run ID: {run_id}")

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    accuracy = run.data.metrics.get("val_acc")

    if accuracy is None:
        print("ERROR: 'val_acc' metric not found in run.")
        sys.exit(1)

    print(f"Model accuracy: {accuracy:.4f}")

    if accuracy < 0.85:
        print(f"FAILED: Accuracy {accuracy:.4f} is below threshold 0.85")
        sys.exit(1)
    else:
        print(f"PASSED: Accuracy {accuracy:.4f} meets threshold 0.85")


if __name__ == "__main__":
    main()
