import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rcParams

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_performance_comparison_chart():
    """Model performans karşılaştırma grafiği"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Model isimleri
    models = ['Logistic Regression', 'Random Forest']
    
    # Accuracy
    accuracy_scores = [0.871, 0.843]
    bars1 = ax1.bar(models, accuracy_scores, color=['#2E86AB', '#A23B72'])
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_ylim(0, 1)
    for bar, score in zip(bars1, accuracy_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # ROC-AUC
    roc_auc_scores = [0.983, 0.984]
    bars2 = ax2.bar(models, roc_auc_scores, color=['#2E86AB', '#A23B72'])
    ax2.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ROC-AUC Score')
    ax2.set_ylim(0.9, 1.0)
    for bar, score in zip(bars2, roc_auc_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score
    f1_scores = [0.880, 0.852]
    bars3 = ax3.bar(models, f1_scores, color=['#2E86AB', '#A23B72'])
    ax3.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('F1-Score')
    ax3.set_ylim(0, 1)
    for bar, score in zip(bars3, f1_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Cross-Validation
    cv_scores = [0.940, 0.942]
    cv_errors = [0.065, 0.052]
    bars4 = ax4.bar(models, cv_scores, yerr=cv_errors, capsize=5, 
                   color=['#2E86AB', '#A23B72'])
    ax4.set_title('Cross-Validation Score', fontsize=14, fontweight='bold')
    ax4.set_ylabel('CV Score')
    ax4.set_ylim(0.8, 1.0)
    for bar, score in zip(bars4, cv_scores):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix_heatmap():
    """Confusion matrix heatmap"""
    # Logistic Regression confusion matrix
    cm_lr = np.array([[12, 0, 0],
                     [0, 40, 7],
                     [1, 1, 9]])
    
    # Random Forest confusion matrix
    cm_rf = np.array([[12, 0, 0],
                     [2, 38, 7],
                     [2, 0, 9]])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Logistic Regression
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                ax=ax1)
    ax1.set_title('Logistic Regression Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Random Forest
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                ax=ax2)
    ax2.set_title('Random Forest Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_class_performance_chart():
    """Sınıf bazlı performans grafiği"""
    classes = ['Negative', 'Neutral', 'Positive']
    
    # Logistic Regression class performance
    lr_precision = [0.92, 0.98, 0.56]
    lr_recall = [1.00, 0.85, 0.82]
    lr_f1 = [0.96, 0.91, 0.67]
    
    # Random Forest class performance
    rf_precision = [0.75, 1.00, 0.56]
    rf_recall = [1.00, 0.81, 0.82]
    rf_f1 = [0.86, 0.89, 0.67]
    
    x = np.arange(len(classes))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Logistic Regression
    bars1 = ax1.bar(x - width, lr_precision, width, label='Precision', color='#2E86AB')
    bars2 = ax1.bar(x, lr_recall, width, label='Recall', color='#A23B72')
    bars3 = ax1.bar(x + width, lr_f1, width, label='F1-Score', color='#F18F01')
    
    ax1.set_title('Logistic Regression - Class Performance', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment Classes')
    ax1.set_ylabel('Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Random Forest
    bars1 = ax2.bar(x - width, rf_precision, width, label='Precision', color='#2E86AB')
    bars2 = ax2.bar(x, rf_recall, width, label='Recall', color='#A23B72')
    bars3 = ax2.bar(x + width, rf_f1, width, label='F1-Score', color='#F18F01')
    
    ax2.set_title('Random Forest - Class Performance', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment Classes')
    ax2.set_ylabel('Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('class_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_dataset_distribution_chart():
    """Veri seti dağılım grafiği"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall dataset distribution
    labels = ['Neutral', 'Negative', 'Positive']
    sizes = [235, 60, 55]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Overall Dataset Distribution (350 samples)', fontsize=14, fontweight='bold')
    
    # Train/Test split
    train_sizes = [188, 48, 44]  # Train set
    test_sizes = [47, 12, 11]    # Test set
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, train_sizes, width, label='Train Set', color='#2E86AB')
    bars2 = ax2.bar(x + width/2, test_sizes, width, label='Test Set', color='#A23B72')
    
    ax2.set_title('Train/Test Split Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment Classes')
    ax2.set_ylabel('Number of Samples')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_table():
    """Performans tablosu"""
    data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Cross-Validation'],
        'Logistic Regression': [0.871, 0.902, 0.871, 0.880, 0.983, '0.940 (±0.065)'],
        'Random Forest': [0.843, 0.888, 0.843, 0.852, 0.984, '0.942 (±0.052)']
    }
    
    df = pd.DataFrame(data)
    
    # Create a styled table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('performance_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def create_all_figures():
    """Tüm grafikleri oluştur"""
    print("Creating all figures for the manuscript...")
    
    create_performance_comparison_chart()
    create_confusion_matrix_heatmap()
    create_class_performance_chart()
    create_dataset_distribution_chart()
    performance_table = create_performance_table()
    
    print("All figures created successfully!")
    print("\nPerformance Table:")
    print(performance_table.to_string(index=False))
    
    return performance_table

if __name__ == "__main__":
    # Tüm grafikleri oluştur
    performance_table = create_all_figures()
    
    print("\n=== Figure Files Created ===")
    print("1. model_performance_comparison.png")
    print("2. confusion_matrices.png") 
    print("3. class_performance.png")
    print("4. dataset_distribution.png")
    print("5. performance_table.png")
    
    print("\n=== Manuscript Integration ===")
    print("These figures should be referenced in the manuscript as:")
    print("- Figure 1: Model Performance Comparison")
    print("- Figure 2: Confusion Matrices")
    print("- Figure 3: Class-wise Performance Analysis")
    print("- Figure 4: Dataset Distribution")
    print("- Table 1: Comprehensive Performance Metrics") 