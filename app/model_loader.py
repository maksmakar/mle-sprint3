# импортируйте необходимую библиотеку
# ваш код здесь
from fastapi import FastAPI
from catboost import CatBoostClassifier

def load_churn_model(model_path: str):
    """Загружаем обученную модель оттока.
    Args:
        model_path (str): Путь до модели.
    """
    try:
        model = CatBoostClassifier()
        model.load_model(model_path)

        print("Model loaded successfully")

    except Exception as e:

        print(f"Failed to load model: {e}")

    return model

if __name__ == "__main__":

    model = load_churn_model('models/catboost_churn_model.bin')
    print(f"Model parameter names: {model.feature_names_}") 