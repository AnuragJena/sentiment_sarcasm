# Multilingual Sentiment Analysis with Sarcasm Detection

This project is part of a master's thesis and focuses on building a deep learning model that performs sentiment classification and sarcasm detection on multilingual customer reviews. It utilizes transformer-based encoders and attention mechanisms to handle complex linguistic features such as sarcasm, contextual ambiguity, and multi-language support.

## 📌 Features
- Detects sentiment (positive, negative) from multilingual text
- Identifies sarcastic tone in reviews
- Built using XLM-RoBERTa, BiLSTM, and Attention layers
- Export results with probabilities and labels
- Ready for integration with Gradio for UI or Flask for deployment

## 🧠 Model Architecture
- Shared encoder: `XLM-RoBERTa`
- Contextual modeling: `Bidirectional LSTM`
- Importance capture: `Attention Layer`
- Two output heads: Sentiment & Sarcasm (Multi-task)

## 🚀 How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Open the notebook:
   Final_Multilingual_Sentiment_Sarcasm.ipynb
3. Run all cells to:
   - Train/load the model
   - Perform prediction
   - Export results
