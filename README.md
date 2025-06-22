

# 📧 Spam Email Classifier with BiLSTM

This project demonstrates how to build a deep learning model using **Bidirectional LSTM** (BiLSTM) and **GloVe embeddings** to classify email messages as **Spam or Ham**.

✅ Built with:
- Keras (TensorFlow backend)
- GloVe word embeddings
- Enron spam dataset

---

## 🚀 Model and Live Demo

- 🤖 **Model Repo:** [`lokas/spam-emails-classifier`](https://huggingface.co/lokas/spam-emails-classifier)  
- 🌐 **Live Web Demo:** [`lokas/spam-email-detector-ui`](https://huggingface.co/spaces/lokas/spam-email-detector-ui)  
- 📓 **Training Notebook:** [`spam_detection.ipynb`](./spam_detection.ipynb)

---

## 🧠 Model Info

- Input: Raw email/message text
- Output: `"Spam"` or `"Ham"`
- Sequence length: 50 tokens
- Embedding: [GloVe 6B 100d](https://nlp.stanford.edu/projects/glove/)
- Training data: [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)

---

## 📊 Metrics

| Metric     | Score   |
|------------|---------|
| Accuracy   | ~99%    |
| F1 Score   | ~95%    |
| Precision  | High    |
| Recall     | High    |

---

## 💡 How to Use

```python
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_path = hf_hub_download("lokas/spam-emails-classifier", "model.h5")
tokenizer_path = hf_hub_download("lokas/spam-emails-classifier", "tokenizer.pkl")

model = load_model(model_path)
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

def predict_spam(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=50)
    pred = model.predict(padded)[0][0]
    return "🚫 Spam" if pred > 0.5 else "✅ Ham"

predict_spam("You won a free iPhone!")
