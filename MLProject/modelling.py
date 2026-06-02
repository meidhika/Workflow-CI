import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_training():
    print("Memuat data...")
    train_df = pd.read_csv("credit_scoring_preprocessing/train.csv")
    test_df = pd.read_csv("credit_scoring_preprocessing/test.csv")
    
    X_train = train_df.drop(columns=['default'])
    y_train = train_df['default']
    X_test = test_df.drop(columns=['default'])
    y_test = test_df['default']
    
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42, n_estimators=50)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Akurasi: {acc}")
        
        # MEMAKSA MLFLOW MENGGUNAKAN PYTHON 3.12.7 UNTUK DOCKER
        custom_env = {
            "name": "credit-scoring-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.12.7",  # Kunci utamanya di sini
                "pip",
                {"pip": ["pandas", "scikit-learn", "mlflow==2.19.0"]}
            ]
        }
        
        # Menyimpan model beserta environment yang sudah ditentukan
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model", 
            conda_env=custom_env
        )

if __name__ == "__main__":
    run_training()