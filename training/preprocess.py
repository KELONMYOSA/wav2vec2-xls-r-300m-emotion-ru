from datasets import Audio, load_dataset
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru")


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = processor(audio_arrays, sampling_rate=processor.feature_extractor.sampling_rate,
                       return_tensors="pt", padding=True)
    return inputs


dusha = load_dataset("KELONMYOSA/dusha_emotion_audio")
dusha = dusha.cast_column("audio", Audio(sampling_rate=16_000))
dusha = dusha.remove_columns("file")

labels = dusha["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label
num_labels = len(id2label)

encoded_dusha = dusha.map(preprocess_function, remove_columns="audio", batched=True)
