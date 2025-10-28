# 🏗 Neural Machine Translation Architecture & Calculation Flow

## 🔧 Complete Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    ENGLISH TO FRENCH TRANSLATION SYSTEM                                         │
│                                          Neural Machine Translation                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

INPUT PROCESSING PIPELINE
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   INPUT TEXT    │    │   PREPROCESSING   │    │   TOKENIZATION  │    │   TENSOR CONV   │
│                 │    │                  │    │                 │    │                 │
│ "I love you"    │───▶│ • Unicode norm   │───▶│ • Word2Idx     │───▶│ [1, 4, 2, 3]   │
│                 │    │ • Punctuation    │    │ • Padding      │    │ Padded tensor   │
│                 │    │ • Start/End tags │    │ • Vocab lookup │    │ [64, 20]        │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │                       │
        │ Raw Text              │ Clean Text            │ Token List            │ Tensor
        ▼                       ▼                       ▼                       ▼
   "I love you"        "<start> i love you <end>"    [1,145,892,78,2]     [1,145,892,78,2,0...]

ENCODER ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         ENCODER SECTION                                                          │
│  ┌─────────────────┐         ┌──────────────────────────┐         ┌─────────────────────────────┐              │
│  │   EMBEDDING     │         │   BIDIRECTIONAL LSTM     │         │     HIDDEN STATES          │              │
│  │                 │         │                          │         │                             │              │
│  │ Input[64,20]   │  ───▶   │  ┌──────────────────────┐ │  ───▶   │ Forward:  [64,20,512]      │              │
│  │      ↓         │  256     │  │    Forward LSTM      │ │  512    │ Backward: [64,20,512]      │              │
│  │ Emb[64,20,256] │         │  │ h₁ → h₂ → h₃ → h₄   │ │         │ Combined: [64,20,512]      │              │
│  │                 │         │  └──────────────────────┘ │         │                             │              │
│  │ Lookup Table:   │         │           ║              │         │ States per timestep:        │              │
│  │ word_id → vec   │         │           ▼              │         │ h₁[512] h₂[512] h₃[512]   │              │
│  │                 │         │  ┌──────────────────────┐ │         │                             │              │
│  │                 │         │  │   Backward LSTM      │ │         │ Final encoder states:       │              │
│  │                 │         │  │ h₄ ← h₃ ← h₂ ← h₁   │ │         │ h_final: [64, 512]         │              │
│  │                 │         │  └──────────────────────┘ │         │ c_final: [64, 512]         │              │
│  └─────────────────┘         └──────────────────────────┘         └─────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                      │
                                            Hidden States Flow
                                                      ▼
DECODER WITH ATTENTION ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        DECODER SECTION                                                          │
│                                                                                                                 │
│  TIMESTEP t: START TOKEN                   ATTENTION MECHANISM                    OUTPUT GENERATION            │
│  ┌─────────────────┐         ┌──────────────────────────┐         ┌─────────────────────────────┐              │
│  │  TARGET INPUT   │         │     LUONG ATTENTION      │         │    PREDICTION LAYER         │              │
│  │                 │         │                          │         │                             │              │
│  │ <start> → [1]  │  ───▶   │  ┌──────────────────────┐ │  ───▶   │ ┌─────────────────────────┐ │              │
│  │      ↓         │  256     │  │ Query: dec_hidden    │ │  ctx    │ │  Dense + Softmax        │ │              │
│  │ Emb[64,1,256]  │         │  │ [64,512]             │ │  [512]  │ │  [64, 15000]            │ │              │
│  │      ↓         │         │  │        ┌─────────────┐│ │         │ │                         │ │              │
│  │ LSTM[64,1,512] │  ──┐    │  │ Keys/Values:         ││ │         │ │ "Je"→234 "t'"→445      │ │              │
│  │      ↓         │    │    │  │ enc_output[64,20,512]││ │         │ │ "aime"→667              │ │              │
│  │ Dec_h[64,512]  │    │    │  └─────────────────────┘│ │         │ └─────────────────────────┘ │              │
│  └─────────────────┘    │    │                          │         └─────────────────────────────┘              │
│                         │    │  Attention Calculation:  │                       │                              │
│                         │    │  ┌─────────────────────┐ │                       │                              │
│                         │    │  │ scores = Q·K^T      │ │                       │                              │
│                         │    │  │ weights= softmax(s) │ │                       ▼                              │
│                         │    │  │ context= Σ(w·V)     │ │              Next token prediction                   │
│                         │    │  └─────────────────────┘ │                       │                              │
│                         │    └──────────────────────────┘                       │                              │
│                         │                   │                                   │                              │
│                         └─── Concatenate ───┘                                   │                              │
│                             [dec_hidden + context]                              │                              │
│                                     ↓                                           │                              │
│                             Transform & Output                                   │                              │
│                                                                                  │                              │
└─────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────┘
                                                                                   │
                                           TEACHER FORCING LOOP                    │
                                                   │                               │
                                                   ▼                               ▼
                              Next timestep: Use predicted token as input    Final Output
                              "Je" → Decoder → Attention → "t'" → "aime"     "Je t'aime"

COMPLETE DATA FLOW ARROWS:
═══════════════════════════

Input Processing:    Text ═══▶ Clean ═══▶ Tokens ═══▶ Tensors
                              │         │           │
Encoder Flow:        Tensors ═╪═▶ Embed ═╪═▶ BiLSTM ═╪═▶ Hidden_States
                              │         │           │
Attention Flow:      Query ═══╪═════════╪═══════════╪═▶ Attention_Weights
                              │         │           │                │
Decoder Flow:        Target ══╪═════════╪═══════════╪═▶ Context ═════╪═▶ Prediction
                              │         │           │           │     │
Output Flow:         Prediction ════════╪═══════════╪═══════════╪═════╪═▶ French_Text
                                       │           │           │     │
Feedback Loop:       Next_Input ◀══════╪═══════════╪═══════════╪═════╪
                     (Teacher Forcing)  │           │           │     │
                                       │           │           │     │
Loss Calculation:    Target_Truth ══════╪═══════════╪═══════════╪═════╪═▶ CrossEntropy_Loss
                                       │           │           │     │              │
Gradient Flow:       Weights ◀═════════╪═══════════╪═══════════╪═════╪══════════════╪
                     (Backpropagation)  │           │           │     │              │
                                       │           │           │     │              │
Parameter Updates:   Optimizer ◀═══════╪═══════════╪═══════════╪═════╪══════════════╪
                     (Adam + Clipping)  │           │           │     │              │
```

## 🧮 Mathematical Calculations Flow

### 1. **Input Processing**

```python
# Original sentence
sentence = "I love you"

# Preprocessing
sentence = preprocess_sentence(sentence)
# Result: "<start> i love you <end>"

# Tokenization
tokens = sentence.split()
# Result: ["<start>", "i", "love", "you", "<end>"]

# Convert to indices
indices = [en_tokenizer.word2idx[word] for word in tokens]
# Result: [1, 4, 2, 5, 3]  # Example indices

# Padding to max_length
padded = indices + [0] * (max_length - len(indices))
# Result: [1, 4, 2, 5, 3, 0, 0, 0, ...]
```

### 2. **Encoder Calculations**

#### Embedding Layer

```python
# Input shape: [batch_size, seq_len] = [64, 20]
# Embedding matrix: [vocab_size, embedding_dim] = [15000, 256]

embedded = embedding_layer(input_ids)
# Output shape: [64, 20, 256]

# Mathematical operation:
# embedded[i][j] = embedding_matrix[input_ids[i][j]]
```

#### Bidirectional LSTM

```python
# Forward LSTM
for t in range(seq_len):
    # Input gate
    i_t = σ(W_ii @ x_t + W_hi @ h_{t-1} + b_i)

    # Forget gate
    f_t = σ(W_if @ x_t + W_hf @ h_{t-1} + b_f)

    # Candidate values
    g_t = tanh(W_ig @ x_t + W_hg @ h_{t-1} + b_g)

    # Output gate
    o_t = σ(W_io @ x_t + W_ho @ h_{t-1} + b_o)

    # Cell state
    c_t = f_t * c_{t-1} + i_t * g_t

    # Hidden state
    h_t = o_t * tanh(c_t)

# Backward LSTM (same equations, reverse order)

# Final combination
h_combined = h_forward + h_backward  # Element-wise addition
```

### 3. **Attention Mechanism (Luong) - Detailed Calculation Flow**

```python
# Step-by-step attention calculation with arrows showing data flow

                    ENCODER OUTPUTS                         DECODER HIDDEN
                    [64, 20, 512]                         [64, 512]
                         │                                      │
                         │ (Keys & Values)                     │ (Query)
                         ▼                                      ▼
                ┌─────────────────┐                   ┌─────────────────┐
                │ All encoder     │                   │ Current decoder │
                │ hidden states   │                   │ hidden state    │
                │ for each word   │                   │ (what to focus  │
                │ in input        │                   │ on next)        │
                └─────────────────┘                   └─────────────────┘
                         │                                      │
                         │                                      │ expand_dims
                         │                                      ▼
                         │                            ┌─────────────────┐
                         │                            │ Query expanded  │
                         │                            │ [64, 1, 512]    │
                         │                            └─────────────────┘
                         │                                      │
                         │ apply linear transformation          │
                         ▼                                      │
                ┌─────────────────┐                           │
                │ W = [512, 512]  │◄──────────────────────────┘
                │ weight matrix   │
                └─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ values_transformed
                │ = values @ W    │
                │ [64, 20, 512]   │
                └─────────────────┘
                         │
                         │ matrix multiplication
                         ▼
                ┌─────────────────┐
                │ scores = query  │
                │ @ keys^T        │
                │ [64, 1, 20]     │
                └─────────────────┘
                         │
                         │ apply softmax
                         ▼
                ┌─────────────────┐      Example weights:
                │ attention_weights│   ┌─ 0.8 (focus on "I")
                │ = softmax(scores)│   ├─ 0.1 ("love")
                │ [64, 1, 20]     │   ├─ 0.1 ("you")
                └─────────────────┘   └─ 0.0 (padding)
                         │
                         │ weighted sum
                         ▼
                ┌─────────────────┐
                │ context_vector  │
                │ = Σ(weights×enc)│
                │ [64, 512]       │
                └─────────────────┘
                         │
                         │ This context vector contains
                         │ relevant information from
                         ▼ the input sequence
            ┌──────────────────────────────┐
            │ Context sent to decoder      │
            │ for next word generation     │
            └──────────────────────────────┘

# Mathematical formula with clear data flow:
# ════════════════════════════════════════════
#
# Input Flow:
# encoder_outputs[64,20,512] ──┬─→ keys[64,20,512]
#                              └─→ values[64,20,512]
# decoder_hidden[64,512] ─────────→ query[64,1,512]
#
# Calculation Flow:
# Step 1: scores = query @ (W @ keys)^T        [64,1,20]
# Step 2: attention_weights = softmax(scores)  [64,1,20]
# Step 3: context = attention_weights @ values [64,512]
#
# Output Flow:
# context[64,512] ──→ concatenate with decoder_output ──→ final_prediction
```

### 4. **Decoder Calculations**

```python
# Input: Previous word embedding + context vector
decoder_input = embedding(previous_word)  # [64, 1, 256]

# LSTM forward pass
decoder_output, (h_n, c_n) = lstm(decoder_input, (h_prev, c_prev))
# decoder_output: [64, 1, 512]

# Combine with attention context
combined = concat([decoder_output.squeeze(1), context_vector], dim=-1)
# combined: [64, 1024]  # 512 + 512

# Apply transformation
transformed = tanh(W_c @ combined + b_c)  # [64, 512]

# Final output layer
logits = W_out @ transformed + b_out  # [64, vocab_size]
probabilities = softmax(logits)  # [64, vocab_size]

# Get predicted word
predicted_id = argmax(probabilities, dim=-1)  # [64]
```

### 5. **Training Loss Calculation**

```python
# Teacher forcing: Use actual target sequence as decoder input
decoder_inputs = target_sequence[:, :-1]  # All but last token
decoder_targets = target_sequence[:, 1:]  # All but first token

# Forward pass through model
predictions, attention_weights = model([encoder_inputs, decoder_inputs])
# predictions: [batch_size, target_seq_len, vocab_size]

# Calculate loss (ignore padding tokens)
loss_fn = SparseCategoricalCrossentropy(from_logits=True, reduction='none')
loss = loss_fn(decoder_targets, predictions)  # [batch_size, target_seq_len]

# Create mask for padding tokens
mask = (decoder_targets != 0).float()  # [batch_size, target_seq_len]
masked_loss = loss * mask

# Final loss
total_loss = masked_loss.sum() / mask.sum()

# Mathematical formula:
# loss = -Σ_i Σ_t mask[i][t] * log(P(y_true[i][t] | y_pred[i][t]))
```

## 📊 Detailed Parameter Flow

### Tensor Shapes Throughout the Model

| Stage         | Tensor              | Shape              | Description                  |
| ------------- | ------------------- | ------------------ | ---------------------------- |
| **Input**     | `input_ids`         | `[64, 20]`         | Batch of tokenized sentences |
| **Embedding** | `embedded`          | `[64, 20, 256]`    | Word embeddings              |
| **Encoder**   | `enc_output`        | `[64, 20, 512]`    | Bidirectional LSTM output    |
| **Encoder**   | `enc_state_h`       | `[64, 512]`        | Final hidden state           |
| **Encoder**   | `enc_state_c`       | `[64, 512]`        | Final cell state             |
| **Attention** | `attention_weights` | `[64, 20]`         | Attention distribution       |
| **Attention** | `context_vector`    | `[64, 512]`        | Weighted encoder states      |
| **Decoder**   | `dec_input`         | `[64, 1]`          | Single target token          |
| **Decoder**   | `dec_embedded`      | `[64, 1, 256]`     | Decoder input embedding      |
| **Decoder**   | `dec_output`        | `[64, 512]`        | Decoder LSTM output          |
| **Output**    | `logits`            | `[64, vocab_size]` | Final predictions            |

### Memory Requirements

```python
# Model parameters (approximate)
encoder_embedding = vocab_en * embedding_dim = 15000 * 256 = 3.84M
decoder_embedding = vocab_fr * embedding_dim = 15000 * 256 = 3.84M
encoder_lstm = 4 * (embedding_dim + units) * units = 4 * (256 + 512) * 512 = 1.57M
decoder_lstm = 4 * (embedding_dim + units) * units = 4 * (256 + 512) * 512 = 1.57M
attention_layer = units * units = 512 * 512 = 0.26M
output_layer = units * vocab_fr = 512 * 15000 = 7.68M

# Total parameters ≈ 18.76M parameters
# Memory usage ≈ 75MB for model weights (FP32)
```

## 🎯 Complete Translation Flow: "I love you" → "Je t'aime"

### Visual Step-by-Step Process with Data Arrows

```
INPUT TRANSFORMATION PIPELINE:
══════════════════════════════

"I love you"
     │ [Raw text input]
     │
     ▼ ✂️ [Preprocessing: unicode_to_ascii, add tokens]
"<start> i love you <end>"
     │ [Normalized text with special tokens]
     │
     ▼ 🔤 [Tokenization: split + word2idx lookup]
["<start>", "i", "love", "you", "<end>"]
     │ [Token list]
     │
     ▼ 🔢 [Index conversion]
[1, 145, 892, 78, 2]
     │ [Numerical indices]
     │
     ▼ ⚡ [Padding to max_length=20]
[1, 145, 892, 78, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     │ [Padded tensor ready for model]
     ▼

ENCODER PROCESSING:
═══════════════════

Tensor[64, 20] (batch_size=64, seq_len=20)
     │
     ▼ 📚 [Embedding lookup: index → dense vector]
Embeddings[64, 20, 256]
     │ Each word becomes a 256-dim vector
     │
     ▼ 🧠 [Bidirectional LSTM processing]

┌─ Forward Pass ──────────────────────────────────────┐
│  h₁ ──→ h₂ ──→ h₃ ──→ h₄ ──→ h₅                   │
│  │      │      │      │      │                     │
│ "<start>" "i"  "love" "you" "<end>"                 │
│ [512]  [512]  [512]  [512]  [512]                   │
└─────────────────────────────────────────────────────┘
           │                    │
           ▼                    ▼
┌─ Backward Pass ─────────────────────────────────────┐
│  h₅ ◄── h₄ ◄── h₃ ◄── h₂ ◄── h₁                   │
│  │      │      │      │      │                     │
│ "<end>" "you" "love" "i" "<start>"                  │
│ [512]  [512]  [512]  [512]  [512]                   │
└─────────────────────────────────────────────────────┘
           │                    │
           ▼                    ▼
Combined Hidden States[64, 20, 512]
     │ [Context-aware representations for each word]
     │
     ▼ 📤 [Output to decoder]
Encoder Memory: All hidden states + final states

DECODER GENERATION (Timestep by Timestep):
══════════════════════════════════════════

🔄 TIMESTEP 1: Generate "Je"
┌─────────────────────────────────────────────────────────┐
│ Input: <start> token [1]                                │
│   │                                                     │
│   ▼ 📚 [Embedding]                                      │
│ Embedded[64, 1, 256]                                    │
│   │                                                     │
│   ▼ 🧠 [LSTM]                                           │
│ Decoder Hidden[64, 512]                                 │
│   │                                                     │
│   ▼ 🎯 [ATTENTION CALCULATION]                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Query: decoder_hidden[64, 512]                      │ │
│ │   │                                                 │ │
│ │   ▼ 🔍 [Compare with all encoder states]            │ │
│ │ Attention Scores[64, 20]:                           │ │
│ │ ┌─ <start>: 0.8  (High focus)                       │ │
│ │ ├─ i:       0.15 (Some focus)                       │ │
│ │ ├─ love:    0.03 (Low focus)                        │ │
│ │ ├─ you:     0.02 (Low focus)                        │ │
│ │ └─ <end>:   0.0  (No focus)                         │ │
│ │   │                                                 │ │
│ │   ▼ 🔗 [Weighted combination]                       │ │
│ │ Context Vector[64, 512]                             │ │
│ └─────────────────────────────────────────────────────┘ │
│   │                                                     │
│   ▼ 🔗 [Concatenate decoder + context]                 │
│ Combined[64, 1024]                                      │
│   │                                                     │
│   ▼ 🎲 [Dense + Softmax]                               │
│ Probabilities[64, 15000]:                              │
│ ┌─ "Je": 0.45      (Highest probability)               │
│ ├─ "I":  0.12      (Lower probability)                 │
│ ├─ "Il": 0.08      (Lower probability)                 │
│ └─ ...: 0.35       (Other words)                       │
│   │                                                     │
│   ▼ 🎯 [ArgMax selection]                              │
│ Predicted: "Je" (token_id: 234)                        │
└─────────────────────────────────────────────────────────┘

🔄 TIMESTEP 2: Generate "t'"
┌─────────────────────────────────────────────────────────┐
│ Input: "Je" token [234] (from previous prediction)      │
│   │                                                     │
│   ▼ 📚 [Embedding]                                      │
│ Embedded[64, 1, 256]                                    │
│   │                                                     │
│   ▼ 🧠 [LSTM with previous states]                      │
│ Decoder Hidden[64, 512] (updated)                       │
│   │                                                     │
│   ▼ 🎯 [ATTENTION CALCULATION]                          │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Attention Scores[64, 20]:                           │ │
│ │ ┌─ <start>: 0.1  (Low focus)                        │ │
│ │ ├─ i:       0.2  (Some focus)                       │ │
│ │ ├─ love:    0.6  (High focus) ◄── MAIN ATTENTION    │ │
│ │ ├─ you:     0.1  (Some focus)                       │ │
│ │ └─ <end>:   0.0  (No focus)                         │ │
│ │   │                                                 │ │
│ │   ▼ 🔗 [Weighted combination focusing on "love"]    │ │
│ │ Context Vector[64, 512]                             │ │
│ └─────────────────────────────────────────────────────┘ │
│   │                                                     │
│   ▼ 🎲 [Generate next word]                            │
│ Predicted: "t'" (token_id: 445)                        │
└─────────────────────────────────────────────────────────┘

🔄 TIMESTEP 3: Generate "aime"
┌─────────────────────────────────────────────────────────┐
│ Input: "t'" token [445]                                 │
│   │                                                     │
│   ▼ [Similar process with updated attention]            │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Attention Scores[64, 20]:                           │ │
│ │ ┌─ <start>: 0.05 (Very low)                         │ │
│ │ ├─ i:       0.05 (Very low)                         │ │
│ │ ├─ love:    0.2  (Some focus)                       │ │
│ │ ├─ you:     0.7  (High focus) ◄── MAIN ATTENTION    │ │
│ │ └─ <end>:   0.0  (No focus)                         │ │
│ └─────────────────────────────────────────────────────┘ │
│   │                                                     │
│   ▼ 🎲 [Generate final content word]                   │
│ Predicted: "aime" (token_id: 667)                      │
└─────────────────────────────────────────────────────────┘

🔄 TIMESTEP 4: Generate "<end>"
┌─────────────────────────────────────────────────────────┐
│ Input: "aime" token [667]                               │
│   │                                                     │
│   ▼ [Process and detect end of sequence]                │
│ Predicted: "<end>" (token_id: 2)                       │
│   │                                                     │
│   ▼ 🛑 [Stop generation]                               │
│ SEQUENCE COMPLETE                                       │
└─────────────────────────────────────────────────────────┘

OUTPUT POST-PROCESSING:
═══════════════════════

Generated tokens: ["Je", "t'", "aime", "<end>"]
     │
     ▼ 🧹 [Remove special tokens]
["Je", "t'", "aime"]
     │
     ▼ 🔗 [Join with spaces]
"Je t' aime"
     │
     ▼ ✂️ [Clean spacing for French]
"Je t'aime"
     │
     ▼ ✅ [FINAL TRANSLATION]

ATTENTION VISUALIZATION MATRIX:
═══════════════════════════════

         Input:  <start>  i    love  you   <end>
Output:     ┌─     ▓▓▓   ░░░   ░░░  ░░░    ░░░   ← "Je"    (focus on <start>)
            ├─     ░░░   ░░░   ▓▓▓  ░░░    ░░░   ← "t'"    (focus on "love")
            └─     ░░░   ░░░   ░░░  ▓▓▓    ░░░   ← "aime"  (focus on "you")

Legend: ▓▓▓ = High attention (0.6-0.8), ░░░ = Low attention (0.0-0.2)

This shows the model correctly learned to:
• Start French sentence when seeing English start
• Translate "love" to French verb components "t'aime"
• Map "you" to the correct French pronoun handling
```

This architecture enables the model to learn complex language mappings while maintaining interpretability through attention mechanisms!
