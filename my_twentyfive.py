import mlflow 
import mlflow.sklearn 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,  cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix
)
import joblib
import json
import os 
import time
import warnings
warnings.filterwarnings('ignore')

# Step 1: Setup MLflow
print("=" * 60)
print("MLOps - MLflow Experiment Tracking")
print("=" * 60)

# MLflow Experiment
mlflow.set_experiment("iris_classification")

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: iris_classification")

# Step 2: Data Preparation
print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

iris       = load_iris()
X, y       = iris.data, iris.target
scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Dataset      : Iris")
print(f"Features     : {X.shape[1]}")
print(f"Classes      : {iris.target_names}")
print(f"Train Size   : {X_train.shape[0]}")
print(f"Test Size    : {X_test.shape[0]}")

# Step 3: Experiment Tracking
print("\n" + "=" * 60)
print("EXPERIMENT TRACKING")
print("=" * 60)

# Models Parameters
experiments = [
    {
        'name'  : 'RandomForest_100',
        'model' : RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        'params': {
            'n_estimators': 100,
            'random_state': 42
        }
    },
    {
        'name'  : 'RandomForest_200',
        'model' : RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            random_state=42
        ),
        'params': {
            'n_estimators': 200,
            'max_depth'   : 5,
            'random_state': 42
        }
    },
    {
        'name'  : 'GradientBoosting',
        'model' : GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ),
        'params': {
            'n_estimators' : 100,
            'learning_rate': 0.1,
            'random_state' : 42
        }
    },
    {
        'name' : 'LogisticRegression',
        'model': LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        'params': {
            'max_iter'     : 1000,
            'C'            : 1.0,
            'random_state' : 42
        }
    },
    {
        'name'  : 'SVM_RBF',
        'model' : SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42
        ),
        'params': {
            'kernel'       : 'rbf',
            'C'            : 1.0,
            'random_state' : 42
        }
    }
]

results = []

for exp in experiments:
    print(f"\nRunning: {exp['name']}...")
    
    with mlflow.start_run(
        run_name=exp['name']
    ):

        # Training Time
        start_time = time.time()

        # Train
        exp['model'].fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict
        y_pred     = exp['model'].predict(X_test)
        y_pred_train = exp['model'].predict(X_train)

        # Metrics
        test_acc   = accuracy_score(
            y_test, y_pred
        )
        train_acc  = accuracy_score(
            y_test, y_pred
        )

        # Cross Validation
        cv_scores = cross_val_score(
            exp['model'], X_scaled, y, cv=5
        )
        cv_mean   = cv_scores.mean()
        cv_std    = cv_scores.std()

        # MLflow Log Parameters
        mlflow.log_params(exp['params'])

        # MLflow Log Metrics
        mlflow.log_metrics({
            'test_accuracy'   : test_acc,
            'train_accuracy'  : train_acc,
            'cv_mean'         : cv_mean,
            'cv_std'          : cv_std,
            'train_time'      : train_time
        })

        # MLflow Log Model
        mlflow.sklearn.log_model(
            exp['model'],
            f"model_{exp['name']}"
        )

        # Results Store
        results.append({
            'Name'      : exp['name'],
            'Test Acc'  : test_acc,
            'Train Acc' : train_acc,
            'CV Mean'   : cv_mean,
            'CV Std'    : cv_std,
            'Train Time': train_time
        })

        print(f"  Test Acc   : {test_acc:.2%}")
        print(f"  CV Mean    : {cv_mean:.2%}")
        print(f"  Train Time : {train_time:.3f}s")

# Step 4: Results Analysis
print("\n" + "=" * 60)
print("RESULTS ANALYSIS")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values(
    'CV Mean', ascending=False
)
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\nBest Model  : {best['Name']}")
print(f"Best CV Mean: {best['CV Mean']:.2%}")

# Step 5: Best Model Save
print("\n" + "=" * 60)
print("BEST MODEL SAVE")
print("=" * 60)

best_exp   = None
best_score = 0

for exp in experiments:
    score = accuracy_score(
        y_test,
        exp['model'].predict(X_test)
    )
    if score > best_score:
        best_score = score
        best_exp   = exp

print(f"Best Model  : {best_exp['name']}")
print(f"Best Score  : {best_score:.2%}")

os.makedirs('mlops_models', exist_ok=True)
joblib.dump(
    best_exp['model'],
    'mlops_models/scaler.pkl'
)

model_metadata = {
    'model_name'   : best_exp['name'],
    'accuracy'     : float(best_score),
    'features'     : list(iris.feature_names),
    'classes'      : list(iris.target_names),
    'created_at'   : time.strftime(
        '%Y-%m-%d %H:%M:%S'
    )
}
with open('mlops_models/metadata.json',
          'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"Saved to     : mlops_models/")

# Step 6: Model Monitoring
print("\n" + "=" * 60)
print("MODEL MONITORING")
print("=" * 60)

def monitor_model(model, scaler,
                  data, true_labels,
                  threshold=0.90):
    """Model Performance Monitor"""

    scaled_data = scaler.transform(data)
    predictions = model.predict(scaled_data)
    accuracy    = accuracy_score(
        true_labels, predictions
    )

    status = "HEALTHY" \
        if accuracy >= threshold \
            else "DEGRADED"

    print(f"\nMonitoring Report:")
    print(f"    Accuracy   : {accuracy:.2%}")
    print(f"    Threshold  : {threshold:.2%}")
    print(f"    Status     : {status}")

    return accuracy, status

# Monitor
acc, status = monitor_model(
    best_exp['model'],
    scaler,
    X_test, y_test
)

# Data Drift Simulation 
print("\nData Drift Simulation:")
np.random.seed(42)
drifted_X = X_test + np.random.normal(
    0, 0.5, X_test.shape
)
drift_acc, drift_status = monitor_model(
    best_exp['model'],
    scaler,
    drifted_X, y_test,
    threshold=0.90
)

# Step 7: Visualization
fig, axes = plt.subplots(2, 2,
                         figsize=(14, 10))

# Plot 1: Accuracy Comparison
names     = results_df['Name'].tolist()
cv_means  = results_df['CV Mean'].tolist()
test_accs = results_df['Test Acc'].tolist()

x      = np.arange(len(names))
width  = 0.35

bars1 = axes[0, 0].bar(
    x - width/2, cv_means, width,
    label='CV Mean',
    color='#3498db', edgecolor='black'
)
bars2 = axes[0, 0].bar(
    x + width/2, test_accs, width,
    label='Test Acc',
    color='#2ecc71', edgecolor='black'
)
axes[0, 0].set_title('Model Comparison',
                     fontweight='bold')
axes[0, 0].set_xticklabels(
    [n.replace('_', '\n') for n in names],
    fontsize=7
)
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].set_ylim(0.8, 1.0)
axes[0, 0].grid(True, alpha=0.3)

for bar in bars1: 
    axes[0, 0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.002,
        f'{bar.get_height():.2%}',
        ha='center', fontsize=7,
        fontweight='bold'
    )

# Plot 2: Training Time
times = results_df['Train Time'].tolist()
axes[0, 1].bar(
    range(len(names)), times,
    color='#e74c3c', edgecolor='black'
)
axes[0, 1].set_xticks(range(len(names)))
axes[0, 1].set_xticklabels(
    [n.replace('_', '\n') for n in names],
    fontsize=7
)
axes[0, 1].set_title('Training Time',
                     fontweight='bold')
axes[0, 1].set_ylabel('Time (seconds)')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Best Model Confusion Matrix
import seaborn as sns 
y_pred_best = best_exp['model'].predict(
    X_test
)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(
    cm, annot=True, fmt='d',
    cmap='Blues',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    ax=axes[1, 0]
)
axes[1, 0].set_title(
    f'Best Model CM ({best_exp["name"]})',
    fontweight='bold'
)

# Plot 4: Monitoring Dashboard
categories  = ['Normal', 'Drifted']
accuracies  = [acc, drift_acc]
colors      = ['#2ecc71'
               if a >= 0.90
               else '#e74c3c'
               for a in accuracies]

bars = axes[1, 1].bar(
    categories, accuracies,
    color=colors, edgecolor='black'
)
axes[1, 1].axhline(
    y=0.90, color='red',
    linestyle='--', linewidth=2,
    label='Threshold (90%)'
)
axes[1, 1].set_title(
    'Model Monitoring Dashboard',
    fontweight='bold'
)
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim(0.5, 1.0)
axes[1, 1].legend()

for bar, acc_val in zip(bars, accuracies):
    axes[1, 1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.01,
        f'{acc_val:.2%}',
        ha='center',
        fontweight='bold'
    )

plt.suptitle(
    'MLops - Experiment Tracking & Monitoring',
    fontsize=14, fontweight='bold'
)
plt.show()
print("Plot saved!")

# Final Summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print("Experiments Run  : {len(experiments)}")
print(f"Best Model      : {best_exp['name']}")
print(f"Best Accuracy   : {best_score:.2%}")
print(f"\nMLflow UI:")
print(f"  Run: mlflow ui")
print(f"  URL: http://localhost:5000")
print(f"\nModel Saved   : mlops_models/")
print(f"Normal Acc      : {acc:.2%} {status}")
print(f"Drifted Acc     : {drift_acc:.2%} "
      f"{drift_status}")