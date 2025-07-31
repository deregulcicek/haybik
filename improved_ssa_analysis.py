import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_fscore_support, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class ImprovedSSAAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.class_names = ['negative', 'neutral', 'positive']
        
    def create_expanded_dataset(self):
        """
        SSA analizi için genişletilmiş veri seti oluştur
        """
        # SSA ile ilgili yorumlar - daha kapsamlı veri seti
        ssa_comments = [
            # Negative SSA Comments (20 örnek)
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
            
            # Neutral SSA Comments (15 örnek)
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
            
            # Positive SSA Comments (15 örnek)
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
            "Bu sistem benim deneyimimi zenginleştiriyor",
            "Algoritma benim için ilginç konular buluyor",
            "Platform benim tercihlerimi anlıyor",
            "Sosyal medyada çeşitli perspektiflerle karşılaşıyorum",
            "Algoritma benim için uygun içerikler seçiyor"
        ]
        
        # Etiketler (0: negative, 1: neutral, 2: positive)
        labels = [0] * 20 + [1] * 15 + [2] * 15  # 50 total comments
        
        # DataFrame oluştur
        self.data = pd.DataFrame({
            'comment': ssa_comments,
            'sentiment': labels
        })
        
        print(f"Genişletilmiş veri seti oluşturuldu: {self.data.shape}")
        print(f"Sınıf dağılımı:\n{self.data['sentiment'].value_counts()}")
        
        return self.data
    
    def preprocess_data(self):
        """
        Veriyi ön işleme
        """
        # Eksik değerleri temizle
        self.data = self.data.dropna()
        
        # Metin verilerini temizle
        self.data['comment_clean'] = self.data['comment'].str.lower()
        
        # Türkçe karakterleri normalize et
        self.data['comment_clean'] = self.data['comment_clean'].str.replace('ç', 'c')
        self.data['comment_clean'] = self.data['comment_clean'].str.replace('ğ', 'g')
        self.data['comment_clean'] = self.data['comment_clean'].str.replace('ı', 'i')
        self.data['comment_clean'] = self.data['comment_clean'].str.replace('ö', 'o')
        self.data['comment_clean'] = self.data['comment_clean'].str.replace('ş', 's')
        self.data['comment_clean'] = self.data['comment_clean'].str.replace('ü', 'u')
        
        print("Veri ön işleme tamamlandı")
        print(f"Sınıf dağılımı:\n{self.data['sentiment'].value_counts()}")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Veriyi train ve test olarak böl
        """
        X = self.data['comment_clean']
        y = self.data['sentiment']
        
        # Stratified split - sınıf dağılımını koru
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {len(X_train)} örnek")
        print(f"Test set: {len(X_test)} örnek")
        print(f"Train sınıf dağılımı: {y_train.value_counts().to_dict()}")
        print(f"Test sınıf dağılımı: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def extract_features(self, X_train, X_test):
        """
        TF-IDF özellik çıkarımı
        """
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            stop_words=None,  # Türkçe için stop words kullanmıyoruz
            min_df=2,
            max_df=0.95
        )
        
        # Train verilerinden özellik çıkar
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Özellik boyutu: {X_train_vec.shape[1]}")
        
        return X_train_vec, X_test_vec
    
    def handle_class_imbalance(self, X_train, y_train):
        """
        Sınıf dengesizliğini ele al
        """
        print("Sınıf dengesizliği analizi:")
        print(f"Original class distribution: {y_train.value_counts().to_dict()}")
        
        # SMOTE ile oversampling
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"Balanced class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        return X_train_balanced, y_train_balanced
    
    def train_models(self, X_train, y_train):
        """
        Farklı modelleri eğit
        """
        models = {}
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        lr_model.fit(X_train, y_train)
        models['LogisticRegression'] = lr_model
        
        # Random Forest
        rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10)
        rf_model.fit(X_train, y_train)
        models['RandomForest'] = rf_model
        
        # SVM
        svm_model = SVC(random_state=42, probability=True, C=1.0, kernel='rbf')
        svm_model.fit(X_train, y_train)
        models['SVM'] = svm_model
        
        return models
    
    def evaluate_models(self, models, X_test, y_test):
        """
        Modelleri değerlendir
        """
        results = {}
        
        for name, model in models.items():
            # Tahminler
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrikler
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='weighted'
            )
            
            # ROC-AUC hesapla
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            except:
                roc_auc = None
            
            # Sınıf bazında metrikler
            class_report = classification_report(
                y_test, y_pred, 
                target_names=self.class_names, 
                output_dict=True
            )
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'class_report': class_report,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"\n{name} Results:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-Score: {f1:.3f}")
            if roc_auc:
                print(f"ROC-AUC: {roc_auc:.3f}")
            else:
                print("ROC-AUC: Hesaplanamadı")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return results
    
    def cross_validation(self, X_train, y_train, cv_folds=5):
        """
        Çapraz doğrulama
        """
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=200),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        cv_results = {}
        
        for name, model in models.items():
            # Çapraz doğrulama skorları
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv_folds, scoring='f1_weighted'
            )
            
            cv_results[name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"\n{name} Cross-Validation Results:")
            print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"CV Scores: {cv_scores}")
        
        return cv_results
    
    def analyze_ssa_keywords(self):
        """
        SSA ile ilgili anahtar kelimeleri analiz et
        """
        ssa_keywords = [
            'alienation', 'trapped', 'echo chamber', 'filter bubble',
            'manipulation', 'isolation', 'yabancılaşma', 'tuzağa düşmüş',
            'yankı odası', 'filtre balonu', 'manipülasyon', 'izolasyon',
            'synthetic social alienation', 'ssa', 'dijital yabancılaşma',
            'sosyal yabancılaşma', 'dijital izolasyon', 'sosyal izolasyon',
            'echo chamber effect', 'filter bubble effect', 'algorithmic alienation'
        ]
        
        keyword_analysis = {}
        
        for keyword in ssa_keywords:
            # Verilerde anahtar kelime geçen yorumları say
            count = self.data['comment_clean'].str.contains(
                keyword, case=False
            ).sum()
            
            keyword_analysis[keyword] = {
                'count': count,
                'percentage': (count / len(self.data)) * 100
            }
        
        # En sık geçen anahtar kelimeleri sırala
        sorted_keywords = sorted(
            keyword_analysis.items(), 
            key=lambda x: x[1]['count'], 
            reverse=True
        )
        
        print("\nSSA Keywords Analysis:")
        for keyword, info in sorted_keywords:
            if info['count'] > 0:
                print(f"{keyword}: {info['count']} occurrences ({info['percentage']:.1f}%)")
        
        return keyword_analysis
    
    def run_complete_analysis(self):
        """
        Tam analiz süreci
        """
        print("=== Improved SSA Sentiment Analysis ===")
        
        # 1. Genişletilmiş veri seti oluştur
        self.create_expanded_dataset()
        
        # 2. Veri ön işleme
        self.preprocess_data()
        
        # 3. SSA anahtar kelime analizi
        keyword_analysis = self.analyze_ssa_keywords()
        
        # 4. Veriyi böl
        X_train, X_test, y_train, y_test = self.split_data()
        
        # 5. Özellik çıkarımı
        X_train_vec, X_test_vec = self.extract_features(X_train, X_test)
        
        # 6. Sınıf dengesizliği ele alma
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train_vec, y_train)
        
        # 7. Çapraz doğrulama
        cv_results = self.cross_validation(X_train_balanced, y_train_balanced)
        
        # 8. Model eğitimi
        models = self.train_models(X_train_balanced, y_train_balanced)
        
        # 9. Model değerlendirmesi
        results = self.evaluate_models(models, X_test_vec, y_test)
        
        return results, cv_results, keyword_analysis

# Kullanım örneği
if __name__ == "__main__":
    analyzer = ImprovedSSAAnalyzer()
    
    # Geliştirilmiş analizi çalıştır
    results, cv_results, keyword_analysis = analyzer.run_complete_analysis()
    
    print("\n=== Analysis Complete ===")
    print("Results saved for manuscript revision") 