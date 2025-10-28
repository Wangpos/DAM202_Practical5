# English to French Neural Machine Translation

A complete implementation of a sequence-to-sequence model with attention mechanism for translating English sentences to French using TensorFlow 2.x.

## Project Overview

This project implements a state-of-the-art neural machine translation (NMT) system that translates English text to French using:

- **Bidirectional LSTM Encoder** for processing input sequences
- **LSTM Decoder with Luong Attention** for generating translations
- **Teacher Forcing** during training for faster convergence
- **BLEU Score Evaluation** for translation quality assessment
- **Attention Visualization** for model interpretability

## Model Architecture

### High-Level Architecture Flow

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Raw English │───▶│ Preprocess  │───▶│ Tokenization │───▶│  Embedding  │───▶│   Encoder   │───▶│  Attention  │
│             │    │ & Normalize │    │  & Padding   │    │   Layer     │    │  BiLSTM     │    │ Mechanism   │
│"I love you" │    │             │    │              │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │                     │                     │
                                                               ▼                     ▼                     ▼
                                                          [64,20,256]           [64,20,512]           [64,20] weights
                                                               │                     │                     │
                                                               │                     │                     ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Final French │◀───│ Detokenize  │◀───│ Postprocess  │◀───│   Softmax   │◀───│   Decoder   │◀───│ Context Vec │
│             │    │   & Clean   │    │ & Remove     │    │ & Sampling  │    │    LSTM     │    │ + Dec Input │
│"Je t'aime"  │    │             │    │   Tokens     │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               ▲                     ▲                     ▲
                                                          [64,vocab_size]        [64,512]             [64,512]
                                                               │                     │                     │
                                                               └─────────────────────┴─────────────────────┘
                                                                          Teacher Forcing Loop
```

### Data Flow with Tensor Shapes

```
"I love you"
     │
     ▼ [Preprocessing]
"<start> i love you <end>"
     │
     ▼ [Tokenization]
[1, 145, 892, 78, 2]
     │
     ▼ [Padding]
[1, 145, 892, 78, 2, 0, 0, 0, ...]  ◄── [batch_size=64, seq_len=20]
     │
     ▼ [Embedding]
Tensor[64, 20, 256]  ◄── Word embeddings
     │
     ▼ [Bidirectional LSTM]
Hidden States[64, 20, 512]  ◄── Context-aware representations
     │
     ▼ [Attention + Decoder Loop]
French tokens: "Je" → "t'" → "aime" → "<end>"
     │
     ▼ [Detokenization]
"Je t'aime"
```

### Detailed Component Breakdown

#### 1. **Encoder (Bidirectional LSTM)**

- **Input**: Tokenized English sentences
- **Processing**: Bidirectional LSTM processes sequences forward and backward
- **Output**: Context-aware hidden states for each time step
- **State Combination**: Forward and backward states are summed

#### 2. **Attention Mechanism (Luong Attention)**

- **Purpose**: Focuses on relevant parts of input during decoding
- **Calculation**: `attention_score = query · W · values`
- **Result**: Weighted combination of encoder hidden states

#### 3. **Decoder (LSTM with Attention)**

- **Input**: Previous target word + attention context
- **Processing**: LSTM generates hidden states
- **Attention**: Computes attention over encoder outputs
- **Output**: Probability distribution over French vocabulary

## Technical Specifications

### Hyperparameters

| Parameter           | Value   | Description                                 |
| ------------------- | ------- | ------------------------------------------- |
| Batch Size          | 64      | Number of sentence pairs per training batch |
| Embedding Dimension | 256     | Word embedding vector size                  |
| LSTM Units          | 512     | Hidden state dimension                      |
| Epochs              | 20      | Training iterations                         |
| Learning Rate       | 0.001   | Adam optimizer learning rate                |
| Max Sequence Length | Dynamic | Based on dataset statistics                 |

### Model Components

- **Vocabulary Size**: ~15,000-20,000 words (English and French)
- **Total Parameters**: ~8-10 million trainable parameters
- **Architecture**: Encoder-Decoder with Attention
- **Loss Function**: Sparse Categorical Crossentropy (ignores padding)
- **Optimizer**: Adam with gradient clipping (max norm = 1.0)

## Project Structure

```
p5 dam202/
├── English to French Translation Practical 5.ipynb  # Main notebook
├── README.md                                        # This file
├── fra.txt                                         # Dataset (downloaded)
└── checkpoints/                                    # Model checkpoints
    ├── ckpt-5.weights.h5
    ├── ckpt-10.weights.h5
    └── final_model.weights.h5
```

## Installation & Setup

### Prerequisites

```bash
pip install tensorflow>=2.8.0
pip install numpy pandas matplotlib scikit-learn
pip install unicodedata
```

### Dataset

The model uses the English-French translation dataset from Anki:

- **Source**: http://www.manythings.org/anki/fra-eng.zip
- **Format**: Tab-separated values (English\tFrench)
- **Size**: 30,000 sentence pairs (configurable)

## Usage

### 1. Training

```python
# Load and preprocess data
en, fr = create_dataset('fra.txt', num_examples=30000)

# Create tokenizers
input_tensor, target_tensor, en_tokenizer, fr_tokenizer, max_length_en, max_length_fr = create_tokenizers_and_datasets(en, fr)

# Initialize and train model
model = EncoderDecoderModel(...)
train_model(model, dataset, epochs=20, steps_per_epoch=steps_per_epoch)
```

### 2. Translation

```python
# Translate a sentence
result, attention_plot = evaluate_translation(
    "I love you", model, en_tokenizer, fr_tokenizer, max_length_en, max_length_fr
)
print(f"Translation: {result}")
```

### 3. Evaluation

```python
# Calculate BLEU score
bleu_score = calculate_bleu_score(reference, candidate)
print(f"BLEU Score: {bleu_score:.2f}")
```

## Performance Metrics

### Training Progress

- **Loss Reduction**: Typically drops from ~4.0 to ~1.5 over 20 epochs
- **Training Time**: ~2-3 minutes per epoch (on GPU)
- **Memory Usage**: ~4-6GB GPU memory

### Translation Quality

- **BLEU Score**: 15-25 (depending on training data size)
- **Common Translations**:
  - "I love you" → "Je t'aime"
  - "How are you?" → "Comment allez-vous ?"
  - "Good morning" → "Bonjour"

## Key Features

### 1. **Attention Visualization**

- Heatmap showing which English words the model focuses on
- Helps understand model decision-making process
- Useful for debugging translation errors

### 2. **Teacher Forcing**

- During training, uses actual target words as decoder input
- Speeds up convergence and improves stability
- Standard technique in sequence-to-sequence models

### 3. **Gradient Clipping**

- Prevents exploding gradients during training
- Ensures stable training process
- Set to maximum norm of 1.0

### 4. **Checkpoint Management**

- Saves model weights every 5 epochs
- Enables training resumption from any checkpoint
- Final model saved at completion

## Model Architecture Details

### Encoder Forward Pass with Data Flow

```
Input: [batch_size, seq_len] = [64, 20]
   │
   ▼ [Embedding Layer]
Embeddings: [64, 20, 256]
   │
   ▼ [Bidirectional LSTM]
┌──────────────────────────────────────────┐
│  Forward Pass: h₁ → h₂ → h₃ → h₄ → h₅   │
│     ↓      ↓      ↓      ↓      ↓        │ ──┐
│  [512]  [512]  [512]  [512]  [512]      │   │
│                                          │   ├─ Concatenated → [64, 20, 512]
│ Backward Pass: h₅ ← h₄ ← h₃ ← h₂ ← h₁   │   │
│     ↓      ↓      ↓      ↓      ↓        │ ──┘
│  [512]  [512]  [512]  [512]  [512]      │
└──────────────────────────────────────────┘
   │
   ▼ [State Combination]
Final States: h_final[64, 512], c_final[64, 512]
Hidden Sequence: encoder_outputs[64, 20, 512]
```

### Attention Calculation with Step-by-Step Flow

```
Decoder Hidden State (Query): [64, 512]
          │
          ▼ [Expand dimensions]
Query: [64, 1, 512]
          │
          ├─────────────────────────────────────┐
          │                                     │
          ▼ [Matrix multiplication]              ▼ [Linear transformation]
Encoder Outputs (Keys): [64, 20, 512] ────► W·Values: [64, 20, 512]
          │                                     │
          ▼ [Dot product attention]             │
Attention Scores: [64, 1, 20] ◄────────────────┘
          │
          ▼ [Softmax normalization]
Attention Weights: [64, 1, 20]
          │         ┌─ 0.8 (focus on "I")
          │         ├─ 0.1 (slight focus on "love")
          │         ├─ 0.1 (slight focus on "you")
          │         └─ 0.0 (padding tokens)
          │
          ▼ [Weighted sum]
Context Vector: [64, 512] ◄── Σ(weights × encoder_hidden_states)
```

### Decoder Forward Pass with Attention Integration

```
Previous Word Token: [64, 1] (e.g., "<start>" or "Je")
          │
          ▼ [Embedding]
Word Embedding: [64, 1, 256]
          │
          ▼ [LSTM Cell]
Decoder Hidden: [64, 1, 512] ──┐
          │                     │
          ▼ [Squeeze]            │
Decoder Output: [64, 512] ──────┤
          │                     │
          │ ┌───────────────────┘
          │ │
          ▼ ▼ [Concatenation]
Combined: [64, 1024] = [decoder_output + context_vector]
          │
          ▼ [Dense + Tanh]
Transformed: [64, 512]
          │
          ▼ [Output Dense Layer]
Logits: [64, vocab_size] = [64, 15000]
          │
          ▼ [Softmax]
Probabilities: [64, 15000]
          │
          ▼ [ArgMax]
Next Token ID: [64] ──► Feed back as next input
```

## 🎨 Visualization Examples

### Attention Heatmap

The model learns to align English and French words correctly:

```
English:  I    love  you
French:   Je   t'    aime
Attention: ▓▓▓  ░░░   ░░░
          ░░░  ▓▓▓   ░░░
          ░░░  ░░░   ▓▓▓
```

## Common Issues & Solutions

### 1. **Memory Errors**

- Reduce batch size to 32 or 16
- Decrease sequence length
- Use gradient accumulation

### 2. **Poor Translation Quality**

- Increase training data size
- Train for more epochs
- Adjust learning rate
- Use beam search for inference

### 3. **Training Instability**

- Ensure gradient clipping is enabled
- Check for data preprocessing errors
- Verify tokenizer consistency

## Future Improvements

### Short-term

- [ ] Implement beam search decoding
- [ ] Add model ensembling
- [ ] Include validation loss monitoring
- [ ] Add early stopping

### Long-term

- [ ] Transformer architecture implementation
- [ ] Subword tokenization (BPE/SentencePiece)
- [ ] Multi-language support
- [ ] Production deployment pipeline

## References & Resources

### Academic Papers

1. [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) - Luong et al.
2. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Bahdanau et al.

### Datasets

- [Many Things - Parallel Corpora](http://www.manythings.org/anki/)
- [OPUS - Open Parallel Corpus](http://opus.nlpl.eu/)

### Tools & Libraries

- [TensorFlow 2.x Documentation](https://www.tensorflow.org/)
- [BLEU Score Implementation](https://www.nltk.org/api/nltk.translate.html)

## License

This project is for educational purposes. The dataset is from Many Things (Anki) and follows their usage terms.

---
_Built with using TensorFlow and attention to detail_
