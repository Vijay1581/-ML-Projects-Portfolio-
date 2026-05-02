import torch
import torch.nn as nn 
from torch.utils.data import (
    Dataset, DataLoader
)
import numpy as np 
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
)
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Step 1: Data Preparation 
print("=" * 55)
print("BERT Fine -tuning - Sentiment Analysis")
print("=" * 55)

# Sample Data
texts = [
    # Positive
        "This movie is amazing and wonderful",
    "I loved this film it was fantastic",
    "Great movie excellent acting superb",
    "Best film I have ever seen brilliant",
    "Wonderful story great characters loved",
    "Outstanding performance incredible movie",
    "Fantastic film highly recommend watching",
    "Excellent movie beautiful story amazing",
    "Perfect film loved every moment brilliant",
    "Superb acting great direction wonderful",
    "Amazing cinematography loved the story",
    "Brilliant movie highly entertaining great",
    "Wonderful performances excellent script",
    "Great film enjoyed every scene fantastic",
    "Loved this movie outstanding brilliant",
    "Incredible film amazing story loved it",
    "Best movie great acting superb direction",
    "Fantastic story excellent performances",
    "Wonderful film loved the characters great",
    "Amazing movie brilliant story perfect",
    # Negative
    "This movie is terrible and boring",
    "I hated this film it was awful",
    "Bad movie poor acting horrible",
    "Worst film I have ever seen terrible",
    "Boring story bad characters hated it",
    "Disappointing performance terrible movie",
    "Awful film do not recommend watching",
    "Terrible movie ugly story boring",
    "Poor film hated every moment awful",
    "Bad acting poor direction terrible",
    "Boring cinematography hated the story",
    "Awful movie highly disappointing bad",
    "Terrible performances poor script",
    "Bad film did not enjoy any scene awful",
    "Hated this movie disappointing terrible",
    "Horrible film bad story hated it",
    "Worst movie bad acting poor direction",
    "Terrible story poor performances",
    "Bad film hated the characters awful",
    "Boring movie terrible story poor",
]
labels = [1]*20 + [0]*20

print(f"Total Samples : {len(texts)}")
print(f"Positive      : {sum(labels)}")
print(f"Negative      : {len(labels)-sum(labels)}")

# Step 2: Tokenizer
print("\n" + "=" * 55)
print("BERT TOKENIZER")
print("=" * 55)

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased'
)

print(f"Vocab Size    : {tokenizer.vocab_size}")
print(f"Max Length    : {tokenizer.model_max_length}")

# Sample Tokenization
sample = texts[0]
tokens = tokenizer(
    sample,
    max_length=64,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
print(f"\nSample Text    : {sample}")
print(f"Token IDs        : {tokens['input_ids'][0][:10]}...")
print(f"Attention Mask   : {tokens['attention_mask'][0][:10]}...")

# Step 3: Dataset Class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels,
                 tokenizer, max_length=64):
        self.texts         = texts
        self.labels        = labels
        self.tokenizer     = tokenizer
        self.max_length    = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text      = self.texts[idx]
        label     = self.labels[idx]

        encoding  = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids'     : encoding[
                'input_ids'].squeeze(),
            'attention_mask':encoding[
                'attention_mask'].squeeze(),
            'label'         : torch.tensor(
                label, dtype=torch.long)   
        }

# Train/Test Split
split          = int(len(texts) * 0.8)
train_texts    = texts[:split]
test_texts     = texts[split:]
train_labels   = labels[:split]
test_labels    = labels[split:] 

train_dataset  = SentimentDataset(
    train_texts, train_labels, tokenizer
)
test_dataset   = SentimentDataset(
    test_texts, test_labels, tokenizer
)

train_loader   = DataLoader(
    train_dataset, batch_size=8, shuffle=True
)
test_loader    = DataLoader(
    test_dataset, batch_size=8, shuffle=False
)

print(f"Train Size    : {len(train_dataset)}")
print(f"Test Size     : {len(test_dataset)}")

# Step 4: BERT Model
print("\n" + "=" * 55)
print("BERT MODEL")
print("=" * 50)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

# Freeze BERT Layers
for param in model.bert.parameters():
    param.requires_grad = False

# Unfreeze LAst 2 Layers
for param in model.bert.encoder.layer[-2:].parameters():
    param.requires_grad = True

total_params    = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Params      : {total_params:,}")
print(f"Trainable Params  : {trainable_params:,}")
print(f"Frozen Params     : "
      f"{total_params-trainable_params:,}")

# Step 5: Training Setup
epochs    = 5
optimizer = AdamW(
    filter(lambda p: p.requires_grad,
           model.parameters()),
    lr=2e-5,
    weight_decay=0.01
)

total_steps = len(train_loader) * epochs
scheduler   = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps    = total_steps // 10,
    num_training_steps  = total_steps
)

criterion = nn.CrossEntropyLoss()

print(f"\nEpochs         : {epochs}")
print(f"Learning Rate    : 2e-5")
print(f"Total Steps      : {total_steps}")

# Step 6: Training
print("\n" + "=" * 55)
print("TRAINING")
print("=" * 55)

train_losses = []
test_losses  = []
train_accs   = []
test_accs    = []

for epoch in range (epochs):
    # Train
    model.train()
    running_loss   = 0.0 
    correct_train  = 0
    total_train    = 0

    for batch in train_loader:
        input_ids      = batch['input_ids']
        attention_mask = batch['attention_mask']
        label          = batch['label']

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label
        )

        loss    = outputs.loss
        logits  = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()
        scheduler.step()

        running_loss  += loss.item()
        preds          = logits.argmax(dim=1)
        correct_train += (preds == label).sum().item()
        total_train   += label.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc  = correct_train / total_train

    # Evaluate
    model.eval()
    running_test  = 0.0
    correct_test  = 0
    total_test    = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids        = batch['input_ids']
            attention_mask   = batch['attention_mask']
            label            = batch['label']

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=label
            )

            loss           = outputs.loss
            logits         = outputs.logits
            running_test  += loss.item()
            preds          = logits.argmax(dim=1)
            correct_test  += (preds == label).sum().item()
            total_test    += label.size(0)

    test_loss = running_test / len(test_loader)
    test_acc  = correct_test / total_test

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] " 
          f"Train Loss: {train_loss:.4f} "
          f"Test Loss : {test_loss:.4f} " 
          f"Train Acc : {train_acc:.2%} "
          f"Test Acc  : {test_acc:.2%}")

# Step 7: Evaluation
print("\n" + "=" * 55)
print("EVALUATION")
print("=" * 55)

model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids       = batch['input_ids']
        attention_mask  = batch['attention_mask']
        label           = batch['label']

        optputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        preds = outputs.logits.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(label.numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"\nFinal Accuracy : {acc:.2%}")
print(f"\nClassification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=['Negative', 'Positive']
))

# Step 8: Visualization 
fig, axes = plt.subplots(1, 2,
                         figsize=(14, 5))

# Plot 1: Loss Curve
axes[0].plot(train_losses, 'b-o',
             label='Train Loss', linewidth=2)
axes[0].plot(test_losses, 'r-o',
             label='Test Loss', linewidth=2)
axes[0].set_title('BERT Loss Curve',
                  fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Accuracy Curve
axes[1].plot(train_accs, 'b-o',
             label='Train Accuracy',
             linewidth=2)
axes[1].plot(test_accs, 'r-o',
             label='Test Accuracy',
             linewidth=2)
axes[1].set_title('BERT Accuracy Curve',
                  fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle(
    'BERT Fine-tunning - Sentiment Analysis',
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig('bert_finetuning.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved!")

# Step 9: New Text Prediction
print("\n" + "=" * 55)
print("NEW TEXT PREDICTION")
print("=" * 55)

def predict_sentiment(text, model,
                      tokenizer):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )

    probs  = torch.softmax(outputs.logits, dim=1)
    pred   = probs.argmax().item()
    conf   = probs.max().item()

    return ('POSITIVE' if pred == 1
            else 'NEGATIVE'), conf

# Test Reviews
test_reviews = [
    "This movie is absolutely amazing!",
    "Terrible film, waste of time",
    "Great acting, loved every moment",
    "Boring and disappointing",
    "Outstanding performance by all actors"
]

print(f"\n{'Review':<40} {'Sentiment':<12} {'Confidence'}")
print("=" * 65)

for review in test_reviews:
    sentiment, conf = predict_sentiment(
        review, model, tokenizer
    )
    emoji = " " if sentiment == "POSITIVE" \
            else " "
    print(f"{review[:38]:<40} "
          f"{emoji} {sentiment:<10} "
          f"{conf:.2%}")

# Final Summary
print("\n" + "=" * 55)
print("FINAL SUMMARY")
print("=" * 55)
print(f"Model           : BERT-base-uncased")
print(f"Task            : Sentiment Analysis")
print(f"Train Size      : {len(train_dataset)}")
print(f"Test Size       : {len(test_dataset)}")
print(f"Epochs          : {epochs}")
print(f"Final Train     : {train_accs[-1]:.2%}")
print(f"Final Test      : {test_accs[-1]:.2%}")
print(f"Best Test       : {max(test_accs):.2%}")
print(f"\nTotal Params  : {total_params:,}")
print(f"Trainable       : {trainable_params:,}")