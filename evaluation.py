import torch
from datasets import load_dataset, Audio
from sklearn.metrics import classification_report
from transformers import AutoConfig, Wav2Vec2Processor, AutoModelForAudioClassification

test_dataset = load_dataset("KELONMYOSA/dusha_emotion_audio", split="test")
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))
dusha = test_dataset.remove_columns("file")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
model = AutoModelForAudioClassification.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)


def predict(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    features = processor(audio_arrays, sampling_rate=processor.feature_extractor.sampling_rate,
                         return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    examples["predicted"] = pred_ids
    return examples


result = test_dataset.map(predict, batched=True, batch_size=8)

label_names = [config.id2label[i] for i in range(config.num_labels)]
y_true = [label for label in result["label"]]
y_pred = result["predicted"]

print(classification_report(y_true, y_pred, target_names=label_names))
