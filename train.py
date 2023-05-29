from transformers import TrainingArguments, AutoConfig

from training.metrics import compute_metrics
from training.model import DataCollatorCTCWithPadding, Wav2Vec2ForSpeechClassification, CTCTrainer
from training.preprocess import encoded_dusha, num_labels, label2id, id2label, processor

config = AutoConfig.from_pretrained(
    "lighteternal/wav2vec2-large-xlsr-53-greek",
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    finetuning_task="wav2vec2_emotion_ru",
    pooling_mode="mean"
)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

model = Wav2Vec2ForSpeechClassification.from_pretrained("facebook/wav2vec2-xls-r-300m", config=config)
model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir="emotion_recognition_ru",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=encoded_dusha["train"],
    eval_dataset=encoded_dusha["test"],
    tokenizer=processor.feature_extractor,
)

trainer.train()
