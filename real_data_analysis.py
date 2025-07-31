import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, 
    precision_recall_fscore_support, accuracy_score
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def read_docx_file(file_path):
    """
    Word dosyasını oku ve metin olarak döndür
    """
    try:
        from docx import Document
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Boş olmayan paragrafları al
                text.append(paragraph.text.strip())
        return text
    except ImportError:
        print("python-docx kütüphanesi yüklü değil. Lütfen: pip install python-docx")
        return []
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        return []

def extract_comments_from_text(text_list):
    """
    Metin listesinden yorumları çıkar
    """
    comments = []
    for text in text_list:
        # Satır satır böl
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Kısa satırları atla
                comments.append(line)
    return comments

def load_real_data():
    """
    Orijinal train ve test dosyalarını yükle
    """
    print("=== Orijinal Veri Yükleme ===")
    
    # Train dosyasını oku
    print("Train dosyası okunuyor...")
    train_text = read_docx_file('interviews_train.docx')
    print(f"Train dosyasından {len(train_text)} paragraf okundu")
    
    # Test dosyasını oku
    print("Test dosyası okunuyor...")
    test_text = read_docx_file('interviews_test.docx')
    print(f"Test dosyasından {len(test_text)} paragraf okundu")
    
    # Yorumları çıkar
    train_comments = extract_comments_from_text(train_text)
    test_comments = extract_comments_from_text(test_text)
    
    print(f"Train yorumları: {len(train_comments)}")
    print(f"Test yorumları: {len(test_comments)}")
    
    # İlk birkaç yorumu göster
    print("\nTrain yorumları örneği:")
    for i, comment in enumerate(train_comments[:5]):
        print(f"{i+1}. {comment[:100]}...")
    
    print("\nTest yorumları örneği:")
    for i, comment in enumerate(test_comments[:5]):
        print(f"{i+1}. {comment[:100]}...")
    
    return train_comments, test_comments

def create_combined_dataset(train_comments, test_comments):
    """
    Train ve test verilerini birleştir ve etiketle
    """
    print("\n=== Veri Seti Oluşturma ===")
    
    # Tüm yorumları birleştir
    all_comments = train_comments + test_comments
    
    # Daha kapsamlı duygu analizi için anahtar kelime bazlı etiketleme
    negative_keywords = [
        'yabancılaşma', 'tuzağa düşmüş', 'manipüle', 'izole', 'yalnız', 
        'kontrol', 'hapis', 'kopuk', 'endişe', 'korku', 'kaygı', 'stres',
        'problem', 'zor', 'kötü', 'olumsuz', 'rahatsız', 'sıkıntı',
        'frustrated', 'angry', 'sad', 'worried', 'anxious', 'trapped',
        'alienated', 'isolated', 'manipulated', 'controlled'
    ]
    
    positive_keywords = [
        'iyi', 'güzel', 'faydalı', 'yardımcı', 'olumlu', 'mutlu', 
        'memnun', 'başarılı', 'gelişim', 'ilerleme', 'hoş', 'keyifli',
        'eğlenceli', 'yararlı', 'destek', 'motivasyon', 'başarı',
        'happy', 'good', 'great', 'useful', 'helpful', 'positive',
        'enjoy', 'like', 'love', 'benefit', 'improve'
    ]
    
    # SSA ile ilgili anahtar kelimeler
    ssa_keywords = [
        'algorithm', 'algoritma', 'echo chamber', 'filter bubble',
        'yankı odası', 'filtre balonu', 'synthetic social alienation',
        'ssa', 'dijital yabancılaşma', 'sosyal yabancılaşma'
    ]
    
    labels = []
    ssa_counts = []
    
    for comment in all_comments:
        comment_lower = comment.lower()
        
        # SSA kelime sayısı
        ssa_count = sum(1 for word in ssa_keywords if word in comment_lower)
        ssa_counts.append(ssa_count)
        
        # Negative kelime sayısı
        neg_count = sum(1 for word in negative_keywords if word in comment_lower)
        # Positive kelime sayısı
        pos_count = sum(1 for word in positive_keywords if word in comment_lower)
        
        # Daha hassas etiketleme
        if neg_count > pos_count and neg_count > 0:
            labels.append(0)  # Negative
        elif pos_count > neg_count and pos_count > 0:
            labels.append(2)  # Positive
        else:
            labels.append(1)  # Neutral
    
    # DataFrame oluştur
    data = pd.DataFrame({
        'comment': all_comments,
        'sentiment': labels,
        'ssa_count': ssa_counts
    })
    
    print(f"Toplam yorum: {len(data)}")
    print(f"Sınıf dağılımı:\n{data['sentiment'].value_counts()}")
    print(f"SSA içeren yorumlar: {sum(data['ssa_count'] > 0)}")
    
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
        'sosyal yabancılaşma', 'dijital izolasyon', 'sosyal izolasyon',
        # Normalize edilmiş versiyonlar
        'yabancilasma', 'tuzaga dusmus', 'yanki odasi', 'filtre balonu',
        'manipulasyon', 'izolasyon', 'dijital yabancilasma', 'sosyal yabancilasma',
        'dijital izolasyon', 'sosyal izolasyon'
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

def run_real_data_analysis():
    """
    Orijinal verilerle analiz
    """
    print("=== Orijinal Veri ile SSA Analizi ===")
    
    # 1. Orijinal verileri yükle
    train_comments, test_comments = load_real_data()
    
    if not train_comments and not test_comments:
        print("Veri yüklenemedi. Sentetik veri kullanılıyor...")
        return None, None
    
    # 2. Veri seti oluştur
    data = create_combined_dataset(train_comments, test_comments)
    
    # 3. Veri ön işleme
    data['comment_clean'] = data['comment'].str.lower()
    data['comment_clean'] = data['comment_clean'].str.replace('ç', 'c')
    data['comment_clean'] = data['comment_clean'].str.replace('ğ', 'g')
    data['comment_clean'] = data['comment_clean'].str.replace('ı', 'i')
    data['comment_clean'] = data['comment_clean'].str.replace('ö', 'o')
    data['comment_clean'] = data['comment_clean'].str.replace('ş', 's')
    data['comment_clean'] = data['comment_clean'].str.replace('ü', 'u')
    
    print(f"\nDataset shape: {data.shape}")
    print(f"Class distribution:\n{data['sentiment'].value_counts()}")
    
    # 4. SSA anahtar kelime analizi
    ssa_percentage = analyze_ssa_keywords(data)
    
    # 5. Veriyi böl (orijinal train/test ayrımını koru)
    train_size = len(train_comments)
    X_train = data['comment_clean'][:train_size]
    y_train = data['sentiment'][:train_size]
    X_test = data['comment_clean'][train_size:]
    y_test = data['sentiment'][train_size:]
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # 6. Özellik çıkarımı
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature dimension: {X_train_vec.shape[1]}")
    
    # 7. Sınıf dengesizliği ele alma
    if len(np.unique(y_train)) > 1:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_vec, y_train)
        print(f"Balanced train set: {X_train_balanced.shape[0]} samples")
    else:
        X_train_balanced, y_train_balanced = X_train_vec, y_train
        print("Single class in train set, SMOTE skipped")
    
    # 8. Model eğitimi ve değerlendirme
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
        if len(np.unique(y_train_balanced)) > 1:
            cv_scores = cross_val_score(
                model, X_train_balanced, y_train_balanced, 
                cv=min(5, len(np.unique(y_train_balanced))), scoring='f1_weighted'
            )
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        else:
            cv_mean, cv_std = 0, 0
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std
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
        print(f"  CV Score: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
        
        print("\nClassification Report:")
        # Dinamik sınıf isimleri
        unique_classes = np.unique(y_test)
        class_names = []
        for cls in unique_classes:
            if cls == 0:
                class_names.append('negative')
            elif cls == 1:
                class_names.append('neutral')
            elif cls == 2:
                class_names.append('positive')
        
        print(classification_report(y_test, y_pred, target_names=class_names))
    
    return results, ssa_percentage

if __name__ == "__main__":
    results, ssa_percentage = run_real_data_analysis()
    
    if results:
        print("\n=== Analysis Summary ===")
        print(f"SSA Content Percentage: {ssa_percentage:.1f}%")
        
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            if metrics['roc_auc']:
                print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
            else:
                print(f"  ROC-AUC: N/A")
            print(f"  CV Score: {metrics['cv_mean']:.3f} (+/- {metrics['cv_std'] * 2:.3f})")
        
        print("\n=== Analysis Complete ===")
    else:
        print("Analiz tamamlanamadı.") 