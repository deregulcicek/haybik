import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_fscore_support, accuracy_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def create_ssa_dataset():
    """
    SSA analizi için dengeli veri seti oluştur
    """
    # SSA ile ilgili yorumlar
    comments = [
        # Negative (20 örnek)
        "Bu algoritma beni sürekli aynı içerikle karşılaştırıyor, kendimi tuzağa düşmüş hissediyorum",
        "Sosyal medyada sadece benim görüşlerime uygun içerik görüyorum, bu endişe verici",
        "Algoritmalar beni manipüle ediyor, gerçek dünyadan kopuk hissediyorum",
        "Bu platformlar beni sürekli aynı kişilerle bağlantıya geçiriyor",
        "Dijital yabancılaşma yaşıyorum, gerçek insanlarla bağlantım azaldı",
        "Kendimi sosyal medyada izole edilmiş hissediyorum",
        "Algoritma beni sadece benzer görüşlere sahip kişilerle bağlantıya geçiriyor",
        "Bu sistem beni gerçek dünyadan koparıyor",
        "Sosyal medyada yabancılaşma yaşıyorum",
        "Algoritma beni manipüle ediyor ve kontrol ediyor",
        "Kendimi dijital bir hapiste gibi hissediyorum",
        "Bu platformlar beni gerçek insanlardan uzaklaştırıyor",
        "Sosyal medyada kendimi yalnız hissediyorum",
        "Algoritma beni sadece belirli içeriklerle sınırlıyor",
        "Bu sistem beni gerçek dünyadan izole ediyor",
        "Dijital yabancılaşma yaşıyorum",
        "Sosyal medyada kendimi tuzağa düşmüş hissediyorum",
        "Algoritma beni manipüle ediyor",
        "Bu platformlar beni gerçek insanlardan koparıyor",
        "Kendimi sosyal medyada yabancılaşmış hissediyorum",
        
        # Neutral (15 örnek)
        "Algoritma benim için içerik seçiyor ama nasıl çalıştığını bilmiyorum",
        "Sosyal medyada farklı içerikler görüyorum ama neden bu içerikleri gördüğümü anlamıyorum",
        "Platform benim ilgi alanlarımı tahmin ediyor gibi görünüyor",
        "Algoritma benim için içerik filtreliyor ama bu iyi mi kötü mü emin değilim",
        "Sosyal medyada çeşitli içeriklerle karşılaşıyorum",
        "Algoritma benim davranışlarımı analiz ediyor gibi görünüyor",
        "Platform benim için kişiselleştirilmiş içerik sunuyor",
        "Sosyal medyada farklı görüşlerle karşılaşıyorum",
        "Algoritma benim tercihlerimi öğreniyor gibi görünüyor",
        "Bu sistem benim için içerik seçiyor",
        "Sosyal medyada çeşitli perspektiflerle tanışıyorum",
        "Algoritma benim ilgi alanlarımı anlıyor",
        "Platform benim için öneriler sunuyor",
        "Sosyal medyada farklı konularla karşılaşıyorum",
        "Algoritma benim davranışlarımı takip ediyor",
        "Bu sistem benim için içerik küratörlüğü yapıyor",
        
        # Positive (15 örnek)
        "Algoritma sayesinde çeşitli görüşlerle karşılaşıyorum, bu iyi",
        "Platform benim ilgi alanlarımı anlıyor ve uygun içerik sunuyor",
        "Sosyal medyada farklı perspektiflerle tanışıyorum",
        "Algoritma beni yeni konularla tanıştırıyor",
        "Bu sistem benim için kişiselleştirilmiş deneyim sunuyor",
        "Algoritma benim için ilginç içerikler buluyor",
        "Platform benim tercihlerimi anlıyor ve uygun öneriler sunuyor",
        "Sosyal medyada çeşitli görüşlerle tanışıyorum",
        "Algoritma benim için kaliteli içerik seçiyor",
        "Bu sistem benim deneyimimi iyileştiriyor",
        "Algoritma benim ilgi alanlarımı keşfediyor",
        "Platform benim için uygun içerikler sunuyor",
        "Sosyal medyada farklı bakış açılarıyla tanışıyorum",
        "Algoritma benim için kişiselleştirilmiş öneriler yapıyor",
        "Bu sistem benim deneyimimi zenginleştiriyor"
    ]
    
    # Etiketler - uzunlukları kontrol et
    print(f"Comments length: {len(comments)}")
    labels = [0] * 20 + [1] * 15 + [2] * 15
    print(f"Labels length: {len(labels)}")
    
    # DataFrame oluştur
    data = pd.DataFrame({
        'comment': comments,
        'sentiment': labels
    })
    
    print(f"Veri seti oluşturuldu: {data.shape}")
    print(f"Sınıf dağılımı:\n{data['sentiment'].value_counts()}")
    
    return data

def preprocess_data(data):
    """
    Veriyi ön işleme
    """
    # Metin temizleme
    data['comment_clean'] = data['comment'].str.lower()
    
    # Türkçe karakterleri normalize et
    data['comment_clean'] = data['comment_clean'].str.replace('ç', 'c')
    data['comment_clean'] = data['comment_clean'].str.replace('ğ', 'g')
    data['comment_clean'] = data['comment_clean'].str.replace('ı', 'i')
    data['comment_clean'] = data['comment_clean'].str.replace('ö', 'o')
    data['comment_clean'] = data['comment_clean'].str.replace('ş', 's')
    data['comment_clean'] = data['comment_clean'].str.replace('ü', 'u')
    
    print("Veri ön işleme tamamlandı")
    return data

def analyze_ssa_keywords(data):
    """
    SSA anahtar kelime analizi
    """
    ssa_keywords = [
        'alienation', 'trapped', 'echo chamber', 'filter bubble',
        'manipulation', 'isolation', 'yabancılaşma', 'tuzağa düşmüş',
        'yankı odası', 'filtre balonu', 'manipülasyon', 'izolasyon',
        'synthetic social alienation', 'ssa', 'dijital yabancılaşma',
        'sosyal yabancılaşma', 'dijital izolasyon', 'sosyal izolasyon'
    ]
    
    print("\nSSA Keywords Analysis:")
    total_comments = len(data)
    ssa_count = 0
    
    for keyword in ssa_keywords:
        count = data['comment_clean'].str.contains(keyword, case=False).sum()
        if count > 0:
            percentage = (count / total_comments) * 100
            print(f"{keyword}: {count} occurrences ({percentage:.1f}%)")
            ssa_count += count
    
    ssa_percentage = (ssa_count / total_comments) * 100
    print(f"\nTotal SSA-related content: {ssa_count}/{total_comments} ({ssa_percentage:.1f}%)")
    
    return ssa_percentage

def run_analysis():
    """
    Ana analiz süreci
    """
    print("=== Improved SSA Sentiment Analysis ===")
    
    # 1. Veri seti oluştur
    data = create_ssa_dataset()
    
    # 2. Veri ön işleme
    data = preprocess_data(data)
    
    # 3. SSA anahtar kelime analizi
    ssa_percentage = analyze_ssa_keywords(data)
    
    # 4. Veriyi böl
    X = data['comment_clean']
    y = data['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} örnek")
    print(f"Test set: {len(X_test)} örnek")
    
    # 5. Özellik çıkarımı
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Özellik boyutu: {X_train_vec.shape[1]}")
    
    # 6. Sınıf dengesizliği ele alma
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
    
    print(f"Balanced train set: {len(X_train_balanced)} örnek")
    
    # 7. Model eğitimi
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{name} Training...")
        
        # Model eğitimi
        model.fit(X_train_balanced, y_train_balanced)
        
        # Tahminler
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            roc_auc = None
        
        # Çapraz doğrulama
        cv_scores = cross_val_score(
            model, X_train_balanced, y_train_balanced, 
            cv=5, scoring='f1_weighted'
        )
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        if roc_auc:
            print(f"  ROC-AUC: {roc_auc:.3f}")
        else:
            print(f"  ROC-AUC: Not available")
        print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['negative', 'neutral', 'positive']))
    
    return results, ssa_percentage

if __name__ == "__main__":
    results, ssa_percentage = run_analysis()
    
    print("\n=== Analysis Summary ===")
    print(f"SSA Content Percentage: {ssa_percentage:.1f}%")
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.3f if metrics['roc_auc'] else 'N/A'}")
        print(f"  CV Score: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std'] * 2:.3f})")
    
    print("\n=== Analysis Complete ===") 