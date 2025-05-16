import json
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# === Загрузка дообученной модели ===
model_path = "./model_lit_nar"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Назначаем читаемые метки
model.config.id2label = {0: "literal", 1: "narrow"}
model.config.label2id = {"literal": 0, "narrow": 1}

clf = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

def detect_anomalies_with_finetuned_model(prompt, response, context=""):
    """
    Анализирует ответ и возвращает вероятности по аномалиям: literal, narrow
    """
    text = f"Question: {prompt}\nContext: {context}\nAnswer: {response}"
    preds = clf(text)[0]
    return {p['label'].lower(): p['score'] for p in preds}

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def process_dataset_with_finetuned_model(data):
    rows = []
    for item in tqdm(data):
        prompt = item["q"]
        responses = item["v"]

        if isinstance(responses, list):
            for resp in responses:
                context = resp.get("c", item.get("c", item.get("context", "")))
                response_text = f"[{resp['c']}] {resp['a']}"
                scores = detect_anomalies_with_finetuned_model(prompt, response_text, context)
                rows.append({
                    "f": item["f"],
                    "s": item["s"],
                    "j": item["j"],
                    "q": prompt,
                    "v": response_text,
                    "context": context,
                    "score_literal": scores.get("literal", 0),
                    "score_narrow": scores.get("narrow", 0),
                    "timestamp": item["d"]
                })
        else:
            context = item.get("c", item.get("context", ""))
            response_text = responses
            scores = detect_anomalies_with_finetuned_model(prompt, response_text, context)
            rows.append({
                "f": item["f"],
                "s": item["s"],
                "j": item["j"],
                "q": prompt,
                "v": response_text,
                "context": context,
                "score_literal": scores.get("literal", 0),
                "score_narrow": scores.get("narrow", 0),
                "timestamp": item["d"]
            })
    return pd.DataFrame(rows)

def visualize_anomalies(df):
    sns.set(style="whitegrid")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df['score_literal'], bins=20, kde=True, ax=axs[0], color='red')
    axs[0].set_title("Буквальная интерпретация (score)")
    axs[0].set_xlabel("Вероятность")

    sns.histplot(df['score_narrow'], bins=20, kde=True, ax=axs[1], color='orange')
    axs[1].set_title("Ограниченность ответа (score)")
    axs[1].set_xlabel("Вероятность")

    plt.tight_layout()
    plt.show()

    pivot = df.groupby("j")[["score_literal", "score_narrow"]].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu")
    plt.title("Средние значения score по теме (j)")
    plt.show()

    df["flag_literal"] = df["score_literal"] > 0.7
    df["flag_narrow"] = df["score_narrow"] > 0.7

    literal_counts = df.groupby("f")["flag_literal"].sum().reset_index(name="Literal")
    narrow_counts = df.groupby("f")["flag_narrow"].sum().reset_index(name="Narrow")

    merged = pd.merge(literal_counts, narrow_counts, on="f")
    merged = pd.melt(merged, id_vars=["f"], value_vars=["Literal", "Narrow"], var_name="Type", value_name="Count")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=merged, x="f", y="Count", hue="Type")
    plt.xticks(rotation=45)
    plt.title("Количество аномалий по предметным областям")
    plt.tight_layout()
    plt.show()

def print_sample_predictions(df, n=5):
    print("\n=== Примеры предсказаний ===")
    sample = df.sample(n=min(n, len(df)))
    for _, row in sample.iterrows():
        print(f"\nQ: {row['q']}\nC: {row['context']}\nA: {row['v']}\nLiteral: {row['score_literal']:.2f}, Narrow: {row['score_narrow']:.2f}")

def print_top_anomalies(df, n=5):
    print("\n=== Топ literal аномалий ===")
    top_literal = df.sort_values("score_literal", ascending=False).head(n)
    for _, row in top_literal.iterrows():
        print(f"\nQ: {row['q']}\nC: {row['context']}\nA: {row['v']}\nLiteral: {row['score_literal']:.2f}, Narrow: {row['score_narrow']:.2f}")

    print("\n=== Топ narrow аномалий ===")
    top_narrow = df.sort_values("score_narrow", ascending=False).head(n)
    for _, row in top_narrow.iterrows():
        print(f"\nQ: {row['q']}\nC: {row['context']}\nA: {row['v']}\nLiteral: {row['score_literal']:.2f}, Narrow: {row['score_narrow']:.2f}")

if __name__ == "__main__":
    input_path = "Datasets/processed_data_with_anomalies.jsonl"
    output_path = "anomaly_predictions.csv"

    data = load_jsonl(input_path)
    df = process_dataset_with_finetuned_model(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Анализ завершён. Результаты сохранены в {output_path}")

    print_sample_predictions(df)
    print_top_anomalies(df)
    visualize_anomalies(df)