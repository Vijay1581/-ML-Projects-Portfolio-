import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    Dataset, DataLoader
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# =============================================
# 📊 Step 1: Setup
# =============================================
print("=" * 60)
print("NLP ADVANCED PROJECT")
print("News Category Classification")
print("=" * 60)

# =============================================
# 📰 Step 2: Data Generation
# =============================================
print("\n" + "=" * 60)
print("DATA GENERATION")
print("=" * 60)

news_data = {
    'Technology': [
        "Apple releases new iPhone with AI features",
        "Google launches advanced AI language model",
        "Microsoft acquires gaming company for billions",
        "Tesla announces new electric vehicle battery",
        "Meta develops virtual reality headset technology",
        "Amazon introduces drone delivery service system",
        "Samsung unveils foldable smartphone display",
        "Intel develops faster computer processor chip",
        "Twitter changes algorithm for better content",
        "Netflix uses AI to recommend shows users",
        "Python programming language gains popularity",
        "Cloud computing market grows significantly",
        "Cybersecurity threats increase globally",
        "Artificial intelligence transforms industries",
        "Robotics automation changes manufacturing",
        "Blockchain technology adoption increases",
        "5G network rollout accelerates worldwide",
        "Software development trends evolve rapidly",
        "Data science skills highly demanded market",
        "Machine learning models improve accuracy",
    ],
    'Sports': [
        "India wins cricket world cup final match",
        "Virat Kohli scores century against Australia",
        "FIFA world cup held in exciting tournament",
        "Ronaldo scores hat trick in football match",
        "Olympics 2024 Paris games begin officially",
        "Federer announces retirement from tennis",
        "NBA finals Lakers win championship title",
        "Premier league Manchester City wins trophy",
        "Sachin Tendulkar cricket records broken",
        "Boxing champion wins heavyweight title fight",
        "Formula one race driver wins championship",
        "Swimming world record broken at Olympics",
        "Basketball team wins national tournament",
        "Football player transfers for record fee",
        "Tennis grand slam tournament results",
        "Marathon runner breaks world record time",
        "Hockey team wins international tournament",
        "Badminton player wins gold medal championship",
        "Wrestling championship held in stadium",
        "Athletics world championship results announced",
    ],
    'Business': [
        "Stock market hits record high today",
        "RBI announces interest rate policy change",
        "Startup receives billion dollar funding round",
        "Company reports quarterly earnings results",
        "Merger acquisition deal announced officially",
        "GDP growth rate increases economic outlook",
        "Foreign investment increases in India market",
        "Bank announces new loan interest rates",
        "Oil prices fluctuate in global market",
        "Real estate market shows growth signs",
        "Inflation rate decreases consumer spending",
        "Trade deal signed between two countries",
        "Business expansion plans announced company",
        "Revenue growth reported financial quarter",
        "Market capitalization reaches trillion dollars",
        "Economic recession fears impact markets",
        "Currency exchange rate fluctuations observed",
        "Investment portfolio diversification strategy",
        "Corporate profits increase market confidence",
        "Financial sector growth opportunities expand",
    ],
    'Health': [
        "New vaccine developed against deadly virus",
        "Cancer research breakthrough announced doctors",
        "WHO declares global health emergency situation",
        "Mental health awareness campaign launched",
        "Diabetes treatment new drug approved FDA",
        "Hospital introduces robotic surgery system",
        "Nutrition research reveals healthy diet tips",
        "Exercise benefits heart health study shows",
        "Medical breakthrough saves thousands lives",
        "Yoga meditation reduces stress anxiety",
        "Covid variant new strain detected globally",
        "Blood pressure control natural remedies",
        "Sleep disorders affect millions worldwide",
        "Vitamin deficiency health problems causes",
        "Obesity epidemic rising concern globally",
        "Alternative medicine treatment gains popularity",
        "Genetic therapy treats inherited diseases",
        "Antibiotic resistance growing health concern",
        "Pregnancy care important health guidelines",
        "Eye care vision problems increasing globally",
    ],
    'Politics': [
        "Election results announced in state voting",
        "Prime minister addresses parliament session",
        "Government announces new policy reform",
        "Opposition party protests parliament decision",
        "International summit leaders discuss issues",
        "Constitution amendment passed parliament vote",
        "Foreign minister visits neighboring country",
        "Budget presented parliament finance minister",
        "New minister appointed cabinet reshuffle",
        "Political party wins state assembly election",
        "Diplomatic relations improve between nations",
        "President signs executive order policy",
        "Senate votes new legislation passed",
        "Political debate candidates discuss issues",
        "Government corruption scandal investigation",
        "Public protest against government policy",
        "Treaty signed international peace agreement",
        "Referendum held constitutional change vote",
        "Sanctions imposed against country violation",
        "United nations resolution passed members",
    ]
}

# DataFrame ತಯಾರಿಸಿ
texts  = []
labels = []

for category, articles in news_data.items():
    for article in articles:
        texts.append(article)
        labels.append(category)

df = pd.DataFrame({
    'text'    : texts,
    'category': labels
})

# Shuffle
df = df.sample(frac=1, random_state=42
               ).reset_index(drop=True)

print(f"Total Articles : {len(df)}")
print(f"Categories     : {df['category'].unique()}")
print(f"\nCategory Distribution:")
print(df['category'].value_counts())

# =============================================
# 🔧 Step 3: Text Preprocessing
# =============================================
print("\n" + "=" * 60)
print("TEXT PREPROCESSING")
print("=" * 60)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Stop Words
stop_words = {
    'the', 'a', 'an', 'in', 'on', 'at',
    'to', 'for', 'of', 'and', 'or', 'is',
    'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can',
    'this', 'that', 'these', 'those', 'with',
    'from', 'by', 'as', 'it', 'its', 'their',
    'they', 'he', 'she', 'we', 'you', 'i',
    'not', 'but', 'if', 'up', 'out', 'about'
}

def remove_stopwords(text):
    words = text.split()
    return ' '.join([
        w for w in words
        if w not in stop_words
    ])

df['clean_text'] = df['text'].apply(
    preprocess_text
)
df['clean_text'] = df['clean_text'].apply(
    remove_stopwords
)

print(f"Original  : {df['text'][0]}")
print(f"Cleaned   : {df['clean_text'][0]}")

# =============================================
# 🔢 Step 4: Tokenization & Encoding
# =============================================
print("\n" + "=" * 60)
print("TOKENIZATION")
print("=" * 60)

# Vocabulary ತಯಾರಿಸಿ
all_words  = ' '.join(df['clean_text']).split()
word_freq  = Counter(all_words)
vocab      = ['<PAD>', '<UNK>'] + [
    w for w, c in word_freq.most_common()
    if c >= 1
]
word2idx   = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Vocab Size : {vocab_size}")
print(f"Top 10 Words:")
for w, c in word_freq.most_common(10):
    print(f"  {w:<20}: {c}")

# Label Encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
num_classes = len(le.classes_)

print(f"\nClasses    : {list(le.classes_)}")
print(f"Num Classes: {num_classes}")

# Encode Text
max_len = 20

def encode_text(text, word2idx, max_len):
    tokens  = text.split()[:max_len]
    encoded = [
        word2idx.get(w, 1) for w in tokens
    ]
    padded  = encoded + [0] * (
        max_len - len(encoded)
    )
    return padded

df['encoded'] = df['clean_text'].apply(
    lambda x: encode_text(x, word2idx, max_len)
)

# =============================================
# 📦 Step 5: Dataset & DataLoader
# =============================================
X = torch.LongTensor(df['encoded'].tolist())
y = torch.LongTensor(df['label'].tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y
)

print(f"\nTrain Size : {X_train.shape[0]}")
print(f"Test Size  : {X_test.shape[0]}")

class NewsDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = NewsDataset(X_train, y_train)
test_dataset  = NewsDataset(X_test, y_test)

train_loader  = DataLoader(
    train_dataset, batch_size=16,
    shuffle=True
)
test_loader   = DataLoader(
    test_dataset, batch_size=16,
    shuffle=False
)

# =============================================
# 🧠 Step 6: Models
# =============================================
print("\n" + "=" * 60)
print("MODELS")
print("=" * 60)

# Model 1: TextCNN
class TextCNN(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim, num_classes,
                 num_filters=64):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim,
            padding_idx=0
        )

        # Multiple Kernel Sizes
        self.conv1 = nn.Conv1d(
            embed_dim, num_filters,
            kernel_size=2, padding=1
        )
        self.conv2 = nn.Conv1d(
            embed_dim, num_filters,
            kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            embed_dim, num_filters,
            kernel_size=4, padding=2
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_filters * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Embedding
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)

        # Conv + Pool
        c1  = self.relu(self.conv1(emb))
        c1  = torch.max(c1, dim=2)[0]

        c2  = self.relu(self.conv2(emb))
        c2  = torch.max(c2, dim=2)[0]

        c3  = self.relu(self.conv3(emb))
        c3  = torch.max(c3, dim=2)[0]

        # Concatenate
        out = torch.cat([c1, c2, c3], dim=1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out

# Model 2: BiLSTM
class BiLSTM(nn.Module):
    def __init__(self, vocab_size,
                 embed_dim, hidden_size,
                 num_classes, num_layers=2):
        super(BiLSTM, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embed_dim, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attention = nn.Linear(
            hidden_size * 2, 1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        emb         = self.embedding(x)
        out, _      = self.lstm(emb)

        # Attention
        attn_w      = torch.softmax(
            self.attention(out), dim=1
        )
        context     = (out * attn_w).sum(dim=1)
        out         = self.classifier(context)
        return out

# Initialize Models
embed_dim    = 64
hidden_size  = 128

textcnn = TextCNN(
    vocab_size, embed_dim, num_classes
)
bilstm  = BiLSTM(
    vocab_size, embed_dim,
    hidden_size, num_classes
)

print(f"TextCNN Parameters: "
      f"{sum(p.numel() for p in textcnn.parameters()):,}")
print(f"BiLSTM Parameters : "
      f"{sum(p.numel() for p in bilstm.parameters()):,}")

# =============================================
# 🏋️ Step 7: Training Function
# =============================================
def train_nlp_model(model, name,
                    train_loader,
                    test_loader,
                    epochs=30):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=0.001
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )

    train_losses = []
    test_losses  = []
    train_accs   = []
    test_accs    = []

    print(f"\nTraining {name}...")
    print(f"{'Epoch':<8} {'TLoss':>8} "
          f"{'VLoss':>8} {'TAcc':>8} "
          f"{'VAcc':>8}")
    print("-" * 45)

    for epoch in range(epochs):
        # Train
        model.train()
        r_loss  = 0.0
        c_train = 0
        t_train = 0

        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            out  = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()

            r_loss  += loss.item()
            pred     = out.argmax(dim=1)
            t_train += y_b.size(0)
            c_train += (pred==y_b).sum().item()

        t_loss = r_loss / len(train_loader)
        t_acc  = c_train / t_train

        # Evaluate
        model.eval()
        r_test  = 0.0
        c_test  = 0
        t_test  = 0

        with torch.no_grad():
            for X_b, y_b in test_loader:
                out    = model(X_b)
                loss   = criterion(out, y_b)
                r_test += loss.item()
                pred   = out.argmax(dim=1)
                t_test += y_b.size(0)
                c_test += (pred==y_b).sum().item()

        v_loss = r_test / len(test_loader)
        v_acc  = c_test / t_test

        scheduler.step()
        train_losses.append(t_loss)
        test_losses.append(v_loss)
        train_accs.append(t_acc)
        test_accs.append(v_acc)

        if (epoch+1) % 5 == 0:
            print(f"{epoch+1:<8} {t_loss:>8.4f} "
                  f"{v_loss:>8.4f} {t_acc:>8.2%} "
                  f"{v_acc:>8.2%}")

    print(f"\nBest {name}: "
          f"{max(test_accs):.2%}")

    return {
        'train_losses': train_losses,
        'test_losses' : test_losses,
        'train_accs'  : train_accs,
        'test_accs'   : test_accs,
        'best_acc'    : max(test_accs)
    }

# =============================================
# 🏋️ Step 8: Train Both Models
# =============================================
print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

cnn_results  = train_nlp_model(
    textcnn, "TextCNN",
    train_loader, test_loader,
    epochs=30
)

lstm_results = train_nlp_model(
    bilstm, "BiLSTM",
    train_loader, test_loader,
    epochs=30
)

# =============================================
# 📊 Step 9: Evaluation
# =============================================
print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

def evaluate_nlp(model, loader,
                 name, le):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_b, y_b in loader:
            out  = model(X_b)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.numpy())
            all_labels.extend(y_b.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n{name}:")
    print(f"Accuracy : {acc:.2%}")
    print(classification_report(
        all_labels, all_preds,
        target_names=le.classes_
    ))
    return all_preds, all_labels

cnn_preds, cnn_labels = evaluate_nlp(
    textcnn, test_loader, "TextCNN", le
)
lstm_preds, lstm_labels = evaluate_nlp(
    bilstm, test_loader, "BiLSTM", le
)

# =============================================
# 📊 Step 10: Visualization
# =============================================
fig, axes = plt.subplots(2, 3,
                          figsize=(18, 12))

# Plot 1: Loss Curves
axes[0, 0].plot(cnn_results['train_losses'],
                'b-', label='CNN Train',
                linewidth=2)
axes[0, 0].plot(cnn_results['test_losses'],
                'b--', label='CNN Test',
                linewidth=2)
axes[0, 0].plot(lstm_results['train_losses'],
                'r-', label='LSTM Train',
                linewidth=2)
axes[0, 0].plot(lstm_results['test_losses'],
                'r--', label='LSTM Test',
                linewidth=2)
axes[0, 0].set_title('Loss Curves',
                      fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Accuracy Curves
axes[0, 1].plot(cnn_results['train_accs'],
                'b-', label='CNN Train',
                linewidth=2)
axes[0, 1].plot(cnn_results['test_accs'],
                'b--', label='CNN Test',
                linewidth=2)
axes[0, 1].plot(lstm_results['train_accs'],
                'r-', label='LSTM Train',
                linewidth=2)
axes[0, 1].plot(lstm_results['test_accs'],
                'r--', label='LSTM Test',
                linewidth=2)
axes[0, 1].set_title('Accuracy Curves',
                      fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Model Comparison
models_comp = ['TextCNN', 'BiLSTM']
best_accs   = [
    cnn_results['best_acc'],
    lstm_results['best_acc']
]
colors = ['#3498db', '#e74c3c']
bars   = axes[0, 2].bar(
    models_comp, best_accs,
    color=colors, edgecolor='black'
)
axes[0, 2].set_title('Model Comparison',
                      fontweight='bold')
axes[0, 2].set_ylabel('Best Accuracy')
axes[0, 2].set_ylim(0, 1.0)
for bar, acc in zip(bars, best_accs):
    axes[0, 2].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f'{acc:.2%}',
        ha='center', fontweight='bold'
    )

# Plot 4: TextCNN Confusion Matrix
cnn_cm = confusion_matrix(
    cnn_labels, cnn_preds
)
sns.heatmap(cnn_cm, annot=True, fmt='d',
            cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[1, 0])
axes[1, 0].set_title(
    'TextCNN Confusion Matrix',
    fontweight='bold'
)
axes[1, 0].tick_params(
    axis='x', rotation=45
)

# Plot 5: BiLSTM Confusion Matrix
lstm_cm = confusion_matrix(
    lstm_labels, lstm_preds
)
sns.heatmap(lstm_cm, annot=True, fmt='d',
            cmap='Greens',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            ax=axes[1, 1])
axes[1, 1].set_title(
    'BiLSTM Confusion Matrix',
    fontweight='bold'
)
axes[1, 1].tick_params(
    axis='x', rotation=45
)

# Plot 6: Word Frequency
axes[1, 2].barh(
    [w for w, _ in
     word_freq.most_common(10)][::-1],
    [c for _, c in
     word_freq.most_common(10)][::-1],
    color='#9b59b6', edgecolor='black'
)
axes[1, 2].set_title('Top 10 Words',
                      fontweight='bold')
axes[1, 2].set_xlabel('Frequency')

plt.suptitle(
    'NLP Advanced — News Classification',
    fontsize=16, fontweight='bold'
)
plt.tight_layout()
plt.savefig('nlp_advanced.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved!")

# =============================================
# 🔮 Step 11: New Article Prediction
# =============================================
print("\n" + "=" * 60)
print("NEW ARTICLE PREDICTION")
print("=" * 60)

def predict_category(text, model,
                     word2idx, le,
                     max_len=20):
    model.eval()
    clean = preprocess_text(text)
    clean = remove_stopwords(clean)
    enc   = encode_text(
        clean, word2idx, max_len
    )
    tensor = torch.LongTensor([enc])

    with torch.no_grad():
        out  = model(tensor)
        prob = torch.softmax(out, dim=1)
        pred = prob.argmax().item()
        conf = prob.max().item()

    return le.classes_[pred], conf

test_articles = [
    "AI technology revolutionizes healthcare",
    "Cricket team wins world championship",
    "Stock market reaches all time high",
    "New vaccine prevents deadly disease",
    "Government announces election schedule",
    "Python programming most popular language",
    "Football player wins golden boot award",
    "Economy grows despite global challenges",
    "Mental health awareness program launched",
    "Parliament passes important legislation",
]

print(f"\n{'Article':<45} {'Category':<15} {'Conf'}")
print("-" * 70)

for article in test_articles:
    cat, conf = predict_category(
        article, bilstm,
        word2idx, le
    )
    print(f"{article[:43]:<45} "
          f"{cat:<15} {conf:.2%}")

# =============================================
# 📊 Final Summary
# =============================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Dataset      : News Articles")
print(f"Total        : {len(df)}")
print(f"Categories   : {num_classes}")
print(f"Vocab Size   : {vocab_size}")
print(f"Max Length   : {max_len}")
print(f"\nTextCNN:")
print(f"  Parameters : "
      f"{sum(p.numel() for p in textcnn.parameters()):,}")
print(f"  Best Acc   : {cnn_results['best_acc']:.2%}")
print(f"\nBiLSTM:")
print(f"  Parameters : "
      f"{sum(p.numel() for p in bilstm.parameters()):,}")
print(f"  Best Acc   : {lstm_results['best_acc']:.2%}")

winner = 'BiLSTM' \
    if lstm_results['best_acc'] > \
       cnn_results['best_acc'] \
    else 'TextCNN'
print(f"\nWinner       : {winner} 🏆")