# Hadoop Log Classification with BiLSTM

A deep learning project for classifying Hadoop log entries by severity level (INFO, WARN, ERROR) using a Bidirectional LSTM neural network.

---

## 📋 Project Overview

This project implements an automated log classification system that analyzes Hadoop log messages and categorizes them into three severity levels. The model achieves **99% accuracy** on test data, making it highly effective for log monitoring and analysis tasks.

### Key Features

- Bidirectional LSTM architecture for sequential text analysis
- Robust text preprocessing pipeline
- Class imbalance handling with weighted training
- Comprehensive evaluation metrics
- Model and tokenizer persistence for deployment

---

## 🎯 Performance Results

```
Overall Accuracy: 99%

Classification Report:
              precision    recall  f1-score   support
       ERROR       1.00      0.94      0.97        17
        INFO       0.99      0.99      0.99       161
        WARN       0.99      1.00      1.00       122

Confusion Matrix:
[[ 16   1   0]     <- ERROR
 [  0 160   1]     <- INFO
 [  0   0 122]]    <- WARN
```

### Training Progress

- **Epoch 1**: 81.3% validation accuracy
- **Epoch 2**: 96.3% validation accuracy
- **Epoch 3**: 97.3% validation accuracy
- **Epoch 4**: 98.7% validation accuracy (best model)
- Early stopping triggered at epoch 7

---

## 🏗️ Architecture

### Model Structure

```
Input Layer (Text sequences)
    ↓
Embedding Layer (128 dimensions, 20K vocab)
    ↓
Bidirectional LSTM (128 units)
    ↓
Dropout (30%)
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (20%)
    ↓
Output Layer (3 classes, Softmax)
```

### Model Configuration

- **Vocabulary Size**: 20,000 words
- **Sequence Length**: 120 tokens
- **Embedding Dimension**: 128
- **Batch Size**: 64
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

---

## 📊 Dataset
[Link Text](https://github.com/logpai/loghub/blob/20168f0fc076be8a75d22c01896076090d4a3c6e/Hadoop/Hadoop_2k.log_structured.csv)
### Data Source
- **File**: `Hadoop_2k.log_structured.csv`
- **Total Records**: 2,000 log entries
- **Features Used**: `Content` (log message) and `Level` (severity)

### Data Split

- **Training Set**: 70% (1,400 samples)
- **Validation Set**: 15% (300 samples)
- **Test Set**: 15% (300 samples)

### Class Distribution

The dataset contains three severity levels with the following test set distribution:
- INFO: 161 samples (53.7%)
- WARN: 122 samples (40.7%)
- ERROR: 17 samples (5.6%)

---

## 🔧 Text Preprocessing Pipeline

The preprocessing function applies several transformations to clean and normalize log messages:

1. **Case Normalization**: Convert all text to lowercase
2. **Path Removal**: Replace file paths (`/path/to/file`) with spaces
3. **Number Tokenization**: Replace numerical values and timestamps with spaces
4. **Special Character Removal**: Keep only alphanumeric characters, underscores, and spaces
5. **Whitespace Normalization**: Remove extra spaces

### Example Transformation

```
Original: "2024-01-15 10:23:45 ERROR /usr/hadoop/data/file123.log Connection timeout"
Cleaned:  "error connection timeout"
```

---

## 🚀 Installation & Usage

### Prerequisites

```bash
pip install numpy pandas scikit-learn tensorflow
```

### Required Libraries

- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `scikit-learn`: Data splitting and evaluation metrics
- `tensorflow`: Deep learning framework

### Running the Project

1. **Prepare your data**: Place `Hadoop_2k.log_structured.csv` in the working directory

2. **Train the model**:
```python
python hadoop_logs.py
```

3. **Model artifacts** will be saved in `saved_model/` directory:
   - `lstm_Hadoop.h5`: Trained model weights
   - `tokenizer.json`: Text tokenizer configuration
   - `label_map.json`: Label encoding mapping
   - `test_predictions.csv`: Test set predictions

### Making Predictions

```python
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('saved_model/lstm_Hadoop.h5')
with open('saved_model/tokenizer.json', 'r') as f:
    tokenizer = tokenizer_from_json(f.read())
with open('saved_model/label_map.json', 'r') as f:
    label_map = json.load(f)

# Preprocess new log
def predict_log_level(log_message):
    cleaned = clean_text(log_message)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=120, padding='post')
    pred = model.predict(padded)
    class_idx = np.argmax(pred)
    return label_map[str(class_idx)], float(np.max(pred))

# Example
log = "Connection refused to namenode on port 8020"
level, confidence = predict_log_level(log)
print(f"Predicted: {level} (confidence: {confidence:.2%})")
```

---

## 🎓 Technical Details

### Handling Class Imbalance

The project implements **class weighting** to address the imbalanced distribution (ERROR logs are much rarer than INFO/WARN):

```python
Class weights:
- ERROR: Higher weight (~7.4x)
- INFO: Standard weight (~1.0x)
- WARN: Moderate weight (~1.3x)
```

This ensures the model pays more attention to minority classes during training.

### Stratified Splitting

The code uses a custom `safe_train_test_split` function that:
- Attempts stratified splitting to preserve class distribution
- Falls back to random splitting if stratification fails (e.g., with very rare classes)
- Ensures robust data splitting across different datasets

### Training Callbacks

- **ModelCheckpoint**: Saves the best model based on validation accuracy
- **EarlyStopping**: Stops training if validation accuracy doesn't improve for 3 consecutive epochs

---

## 📁 Project Structure

```
.
├── hadoop_logs.py                      # Main training script
├── Hadoop_2k.log_structured.csv        # Input dataset
└── saved_model/
    ├── lstm_Hadoop.h5                  # Trained model
    ├── tokenizer.json                  # Tokenizer configuration
    ├── label_map.json                  # Label mappings
    └── test_predictions.csv            # Evaluation results
```

---

## 🔍 Key Functions

### `load_data(path)`
Loads and validates the CSV dataset, ensuring required columns exist.

### `clean_text(text)`
Preprocesses log messages by removing noise and normalizing text.

### `safe_train_test_split(X, y, ...)`
Performs stratified or non-stratified data splitting with fallback logic.

### `texts_to_padded_sequences(texts)`
Converts text to numerical sequences and pads to fixed length.

---

## 📈 Future Improvements

- Experiment with transformer architectures (BERT, DistilBERT)
- Add multi-label classification for subcategories
- Implement real-time log stream processing
- Extend to other log formats (Apache, Nginx, etc.)
- Add anomaly detection for unusual log patterns
- Create a web interface for interactive predictions

---

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:
- Support for additional log formats
- Hyperparameter optimization
- Model compression for edge deployment
- Integration with log aggregation tools (ELK Stack, Splunk)

---

## 📝 License

This project is open source and available for educational and research purposes.

---

## 👤 Author

Created as part of a machine learning project for log analysis and monitoring systems.

---

## 🙏 Acknowledgments

- Dataset: Hadoop 2k structured logs
- Framework: TensorFlow/Keras
- Inspired by practical needs in DevOps and system monitoring