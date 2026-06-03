import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_training():
    print("Memuat data...")
    # Mengubah ke folder Banknote
    train_df = pd.read_csv("banknote_preprocessing/train.csv")
    test_df = pd.read_csv("banknote_preprocessing/test.csv")
    
    # Mengubah target dari 'default' menjadi 'class'
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']
    
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42, n_estimators=50)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Akurasi: {acc}")
        
        # MEMAKSA MLFLOW MENGGUNAKAN PYTHON 3.12.7 UNTUK DOCKER
        custom_env = {
            "name": "banknote-env",  # Diubah menjadi banknote
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.12.7",  # Kunci utamanya di sini
                "pip",
                {"pip": ["pandas", "scikit-learn", "mlflow==2.19.0"]}
            ]
        }
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            conda_env=custom_env
        )
        print("Model berhasil disimpan ke dalam MLflow.")

if __name__ == "__main__":
    run_training()