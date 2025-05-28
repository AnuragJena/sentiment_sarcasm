import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from langdetect import detect
from deep_translator import GoogleTranslator
import gradio as gr
import os
from datetime import datetime

# ✅ Translation utility
def translate_if_needed(text):
    try:
        lang = detect(text)
        if lang != 'en':
            return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        pass
    return text

# ✅ Load tokenizer and encoder
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
encoder_model = AutoModel.from_pretrained("xlm-roberta-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Multitask model with sentiment + sarcasm
class MultiTaskModel(nn.Module):
    def __init__(self, encoder, hidden_dim=128):
        super().__init__()
        self.encoder = encoder
        self.lstm = nn.LSTM(input_size=encoder.config.hidden_size, hidden_size=hidden_dim,
                             batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(hidden_dim * 2, 64)
        self.sentiment_out = nn.Linear(64, 1)
        self.sarcasm_out = nn.Linear(64, 1)

    def forward(self, input_ids, attention_mask):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(enc.last_hidden_state)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        x = self.dropout(context_vector)
        x = torch.relu(self.dense(x))
        return self.sentiment_out(x), self.sarcasm_out(x)

# ✅ Load trained multitask model
model = MultiTaskModel(encoder_model).to(device)
model_path = "/content/drive/MyDrive/Colab Notebooks/multitask_sentiment_sarcasm_review.pt"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ Main function for both input modes
def analyze_sentiment_and_sarcasm(mode, text, file, threshold):
    results = []
    if mode == "Text Input":
        original_text = text.strip()
        if not original_text:
            return pd.DataFrame([{"Error": "❌ Please enter some text."}])

        translated = translate_if_needed(original_text)
        encoding = tokenizer.encode_plus(
            translated,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            sentiment_out, sarcasm_out = model(input_ids=input_ids, attention_mask=attention_mask)
            sentiment_prob = torch.sigmoid(sentiment_out).squeeze().item()
            sarcasm_prob = torch.sigmoid(sarcasm_out).squeeze().item()

        pred = 1 if sentiment_prob > threshold else 0
        sentiment = "Positive" if pred == 1 else "Negative"
        is_sarcastic = sarcasm_prob > 0.5

        if is_sarcastic:
            if sentiment == "Positive":
                sentiment = "Sarcastic ➝ Negative"
                sentiment_prob = 1 - sentiment_prob
            elif sentiment == "Negative":
                sentiment = "Negative (Sarcastic)"

        results.append({
            "Original Review": original_text,
            "Translated Review": translated,
            "Predicted Sentiment": sentiment,
            "Confidence": round(sentiment_prob if pred == 1 else 1 - sentiment_prob, 4),
            "Raw Probability": round(sentiment_prob, 4),
            "Sarcasm Detected": "Yes" if is_sarcastic else "No"
        })
        return pd.DataFrame(results)

    elif mode == "File Upload":
        try:
            df = pd.read_csv(file.name)
        except Exception as e:
            return pd.DataFrame([{"Error": f"❌ Failed to read CSV: {e}"}])

        if "Review" not in df.columns:
            return pd.DataFrame([{"Error": "❌ CSV must contain a column named 'Review'."}])

        for original_text in df["Review"].dropna().tolist():
            translated = translate_if_needed(original_text)
            encoding = tokenizer.encode_plus(
                translated,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                sentiment_out, sarcasm_out = model(input_ids=input_ids, attention_mask=attention_mask)
                sentiment_prob = torch.sigmoid(sentiment_out).squeeze().item()
                sarcasm_prob = torch.sigmoid(sarcasm_out).squeeze().item()

            pred = 1 if sentiment_prob > threshold else 0
            sentiment = "Positive" if pred == 1 else "Negative"
            is_sarcastic = sarcasm_prob > 0.5

            if is_sarcastic:
                if sentiment == "Positive":
                    sentiment = "Sarcastic ➝ Negative"
                    sentiment_prob = 1 - sentiment_prob
                elif sentiment == "Negative":
                    sentiment = "Negative (Sarcastic)"

            results.append({
                "Original Review": original_text,
                "Translated Review": translated,
                "Predicted Sentiment": sentiment,
                "Confidence": round(sentiment_prob if pred == 1 else 1 - sentiment_prob, 4),
                "Raw Probability": round(sentiment_prob, 4),
                "Sarcasm Detected": "Yes" if is_sarcastic else "No"
            })

        output_df = pd.DataFrame(results)
        output_dir = "/content/drive/MyDrive/sentiment_outputs/"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"translated_predicted_sentiments_{timestamp}.csv"
        output_path = os.path.join(output_dir, file_name)
        output_df.to_csv(output_path, index=False)
        return output_df

# ✅ Gradio UI with text or file input
gr.Interface(
    fn=analyze_sentiment_and_sarcasm,
    inputs=[
        gr.Radio(["Text Input", "File Upload"], label="Choose Input Mode", value="Text Input"),
        gr.Textbox(label="Enter Review (only for Text Input mode)", lines=3),
        gr.File(label="Upload CSV with 'Review' column (only for File Upload mode)"),
        gr.Slider(0.1, 0.9, step=0.01, value=0.5, label="Sentiment Threshold")
    ],
    outputs=gr.Dataframe(headers=["Original Review", "Translated Review", "Predicted Sentiment", "Confidence", "Raw Probability", "Sarcasm Detected"]),
    title="Multilingual Sentiment & Sarcasm Detector",
    description="Choose between manual text input or CSV upload to detect sentiment and sarcasm."
).launch()