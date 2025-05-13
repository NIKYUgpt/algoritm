# version1.py

import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from transformers import pipeline
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
init(autoreset=True)

class QAValidator:
    def __init__(self):
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.semantic_threshold = 0.5
        self.duplicate_threshold = 0.9
        self.cluster_eps = 0.5
        self.contradiction_threshold = 0.7
        self.out_of_context_threshold = 0.2
        self.anomaly_stats = defaultdict(int)

    def load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def validate_item(self, item):
        anomalies = []

        if not item.get('current_segment'):
            anomalies.append(('missing_value', 'Пустой вопрос'))
        if not item.get('context'):
            anomalies.append(('missing_value', 'Пустой контекст'))
        if not item.get('target'):
            anomalies.append(('missing_value', 'Пустой ответ'))

        if len(item['context'].split()) < 5:
            anomalies.append(('short_context', 'Короткий контекст'))

        if len(item['target'].split()) > 20:
            anomalies.append(('long_target', 'Длинный ответ'))

        try:
            context_emb = self.semantic_model.encode(item['context'])
            target_emb = self.semantic_model.encode(item['target'])
            sim = cosine_similarity([context_emb], [target_emb])[0][0]
            if sim < self.semantic_threshold:
                anomalies.append(('low_semantic_similarity', f"Сходство: {sim:.2f}"))

            result = self.zero_shot_classifier(
                item['context'],
                candidate_labels=[item['target'], "противоречие", "нерелевантно"]
            )
            if result['labels'][0] in ["противоречие", "нерелевантно"]:
                anomalies.append(('conceptual_inconsistency', f"Метка: {result['labels'][0]}"))

            contradiction = self.zero_shot_classifier(
                item['context'],
                candidate_labels=["подтверждение", "противоречие", "нейтрально"],
                hypothesis_template="Ответ: {}"
            )
            if contradiction['labels'][0] == "противоречие" and contradiction['scores'][0] > self.contradiction_threshold:
                anomalies.append(('contradictory_context', f"Противоречие: {contradiction['scores'][0]:.2f}"))
        except Exception as e:
            anomalies.append(('error', str(e)))

        for a_type, _ in anomalies:
            self.anomaly_stats[a_type] += 1

        return anomalies

    def print_result(self, idx, item, anomalies):
        print(f"\n{Fore.CYAN}{'='*30} Запись #{idx + 1} {'='*30}")
        print(f"{Fore.YELLOW}Вопрос: {item['current_segment']}")
        print(f"{Fore.BLUE}Контекст: {item['context']}")
        print(f"{Fore.GREEN}Ответ: {item['target']}")

        if not anomalies:
            print(f"{Fore.GREEN}✓ ВАЛИДНО")
        else:
            print(f"{Fore.RED}✗ НЕВАЛИДНО ({len(anomalies)} ошибок):")
            for a_type, desc in anomalies:
                print(f"  - {a_type}: {desc}")

    def visualize_stats(self, total, valid):
        labels = ['Valid'] + list(self.anomaly_stats.keys())
        sizes = [valid] + list(self.anomaly_stats.values())

        plt.figure(figsize=(12, 5))

        # Pie chart
        plt.subplot(1, 2, 1)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title("Распределение записей")

        # Bar chart
        plt.subplot(1, 2, 2)
        plt.bar(self.anomaly_stats.keys(), self.anomaly_stats.values())
        plt.title("Распределение аномалий")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def run(self, path):
        data = self.load_json(path)
        total = len(data)
        valid_count = 0

        for i, item in enumerate(data):
            anomalies = self.validate_item(item)
            self.print_result(i, item, anomalies)
            if not anomalies:
                valid_count += 1

        print(f"\n{Fore.MAGENTA}=== Статистика ===")
        print(f"Всего записей: {total}")
        print(f"Валидных: {valid_count} ({valid_count/total*100:.1f}%)")
        print(f"Проблемных: {total - valid_count}")

        self.visualize_stats(total, valid_count)

if __name__ == "__main__":
    validator = QAValidator()
    validator.run("augmented_dt.json")
