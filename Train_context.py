import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# === Подготовка данных ===
class AnomalyDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                question = item.get("question", "")
                answer = item.get("answer", "")
                context = item.get("context", "")
                text = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
                labels = torch.tensor(item["labels"], dtype=torch.float32)
                inputs = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt"
                )
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["labels"] = labels
                self.samples.append(inputs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# === Параметры ===
model_ckpt = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=2,
    problem_type="multi_label_classification"
)

# === Загрузка датасета ===
train_dataset = AnomalyDataset("Trainv2/contextual_anomaly_dataset.jsonl", tokenizer)

# === Аргументы обучения ===
training_args = TrainingArguments(
    output_dir="./model_lit_nar",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    logging_steps=2e-5,
    save_strategy="no"
)

# === Тренировка ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# === Сохранение модели ===
model.save_pretrained("./model_lit_nar")
tokenizer.save_pretrained("./model_lit_nar")
