import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Autolog untuk mencatat metrik dan artefak model secara lokal di runner CI
mlflow.sklearn.autolog()

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

if __name__ == "__main__":
    run_training()