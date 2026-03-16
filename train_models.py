import sys
import pandas as pd
import numpy as np

# Vizualizacijoms ir modeliams
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Teksto apdorojimui
import nltk
from nltk.corpus import stopwords


def ensure_utf8_stdout() -> None:
    """
    Užtikrina, kad Windows konsolė teisingai rodytų UTF-8 simbolius.
    """
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            # Jei reconfigure nepavyksta, tiesiog tęsiame be klaidos kėlimo
            pass


def ensure_nltk_resources() -> None:
    """
    Atsisiunčia reikalingus NLTK išteklius, jei jų dar nėra.
    """
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")


def load_and_clean_data(csv_path: str = "McDonald_s_Reviews_prepared.csv") -> pd.DataFrame:
    """
    Nuskaito paruoštą CSV ir atlieka papildomą teksto valymą:
    - paverčia tekstą į mažąsias raides
    - pašalina anglų kalbos stop-žodžius iš `review_cleaned`
    """
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Užtikriname, kad sentiment stulpelis būtų sveikasis tipas (0 / 1)
    df["sentiment"] = df["sentiment"].astype(int)

    # Papildomas valymas: mažosios raidės ir stop-žodžių pašalinimas
    ensure_nltk_resources()
    eng_stopwords = set(stopwords.words("english"))

    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        tokens = text.split()
        tokens = [t for t in tokens if t not in eng_stopwords]
        return " ".join(tokens)

    df["review_cleaned_tokens"] = df["review_cleaned"].astype(str).apply(clean_text)

    return df


def vectorize_text(df: pd.DataFrame):
    """
    Vektorizuoja tekstą naudodama TF-IDF (max_features = 5000).
    Grąžina X (vektorius), y (sentiment) ir pačią vektorizatorių.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["review_cleaned_tokens"])
    y = df["sentiment"].values
    return X, y, vectorizer


def train_models(X_train, y_train):
    """
    Apmoko tris modelius:
    - Logistinę regresiją
    - Atsitiktinių miškų klasifikatorių
    - MLP (dirbtinį neuroninį tinklą)
    """
    models = {}

    # Logistinė regresija (dažnai geras bazinis modelis tekstui)
    log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
    log_reg.fit(X_train, y_train)
    models["LogisticRegression"] = log_reg

    # Atsitiktinių miškų klasifikatorius
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # Dirbtinis neuroninis tinklas
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    models["MLPClassifier"] = mlp

    return models


def plot_confusion_matrix(y_true, y_pred, title: str) -> None:
    """
    Nubraižo Confusion Matrix naudodama seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Prognozuota 0 (neigiama)", "Prognozuota 1 (teigiama)"],
        yticklabels=["Tikra 0 (neigiama)", "Tikra 1 (teigiama)"],
    )
    plt.title(title)
    plt.xlabel("Prognozė")
    plt.ylabel("Tikra reikšmė")
    plt.tight_layout()
    plt.show()


def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Įvertina visus pateiktus modelius:
    - išveda Accuracy ir F1-score
    - nubraižo Confusion Matrix kiekvienam modeliui
    - grąžina palyginimo lentelę kaip DataFrame
    """
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"\n===== {name} =====")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred, digits=4))

        # Nubraižome Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - {name}")

        results.append(
            {
                "model": name,
                "accuracy": acc,
                "f1_score": f1,
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False).reset_index(drop=True)
    print("\n===== Modelių palyginimo lentelė (pagal F1-score) =====")
    print(results_df)

    return results_df


def add_needs_urgent_response_column(df: pd.DataFrame, model, vectorizer) -> pd.DataFrame:
    """
    Prideda stulpelį `needs_urgent_response` pagal verslo logiką:
    - reikšmė 1, jei:
        * modelis prognozuoja neigiamą sentimentą (0)
        IR
        * tekste yra bent vienas iš raktažodžių: 'poison', 'sick', 'rude', 'dirty'
      kitu atveju 0.
    """
    # Naudojame tą patį išvalytą tekstą, kuris buvo naudotas vektorizavimui
    X_all = vectorizer.transform(df["review_cleaned_tokens"])
    y_pred_all = model.predict(X_all)

    keywords = ["poison", "sick", "rude", "dirty"]

    def has_urgent_keyword(text: str) -> bool:
        if not isinstance(text, str):
            return False
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    urgent_flags = []
    for text, pred in zip(df["review_cleaned_tokens"], y_pred_all):
        if pred == 0 and has_urgent_keyword(text):
            urgent_flags.append(1)
        else:
            urgent_flags.append(0)

    df = df.copy()
    df["needs_urgent_response"] = urgent_flags
    return df


def main():
    """
    Pagrindinė vykdymo funkcija:
    - užtikrina UTF-8 išvedimą
    - įkelia ir išvalo duomenis
    - vektorizuoja tekstą
    - padalina duomenis į mokymo ir testavimo rinkinius
    - apmoko 3 modelius
    - įvertina jų veikimą ir išveda palyginimo lentelę
    - prideda `needs_urgent_response` stulpelį naudodama logistinės regresijos modelį
    - išsaugo naują CSV su verslo logikos stulpeliu
    """
    ensure_utf8_stdout()

    print("=== Duomenų įkėlimas ir papildomas valymas ===")
    df = load_and_clean_data("McDonald_s_Reviews_prepared.csv")
    print(f"Eilučių skaičius po valymo: {len(df)}")

    print("\n=== TF-IDF vektorizacija ===")
    X, y, vectorizer = vectorize_text(df)
    print(f"TF-IDF matrica: {X.shape[0]} eilučių, {X.shape[1]} požymių")

    print("\n=== Duomenų skaidymas (train/test 80/20) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Mokymo rinkinys: {X_train.shape[0]} eilučių")
    print(f"Testavimo rinkinys: {X_test.shape[0]} eilučių")

    print("\n=== Modelių mokymas ===")
    models = train_models(X_train, y_train)

    print("\n=== Modelių vertinimas ===")
    results_df = evaluate_models(models, X_train, X_test, y_train, y_test)

    # Pasirenkame geriausią modelį pagal F1-score (jei keli vienodi – imame pirmą)
    best_model_name = results_df.iloc[0]["model"]
    best_model = models[best_model_name]
    print(f"\nGeriausiai pasirodęs modelis pagal F1-score: {best_model_name}")

    print("\n=== Verslo logikos stulpelio `needs_urgent_response` kūrimas ===")
    df_with_flags = add_needs_urgent_response_column(df, best_model, vectorizer)
    output_path = "McDonald_s_Reviews_with_flags.csv"
    df_with_flags.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Duomenys su `needs_urgent_response` išsaugoti į: {output_path}")


if __name__ == "__main__":
    main()

