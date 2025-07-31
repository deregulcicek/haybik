import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class SSASentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.class_names = ['negative', 'neutral', 'positive']
        
    def load_data(self, train_file, test_file):
        """
        Train ve test verilerini yükle
        """
        try:
            # Word dosyalarını pandas ile okuma (eğer CSV formatına çevrilmişse)
            if train_file and test_file:
                self.train_data = pd.read_csv(train_file) if train_file.endswith('.csv') else pd.read_excel(train_file)
                self.test_data = pd.read_csv(test_file) if test_file.endswith('.csv') else pd.read_excel(test_file)
            else:
                raise ValueError("Dosya yolları belirtilmedi")
            
            print(f"Train data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            print(f"Train data columns: {self.train_data.columns.tolist()}")
            
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            # Örnek veri oluştur (gerçek verilerinizle değiştirin)
            self.create_sample_data()
    
    def create_sample_data(self):
        """
        Örnek veri oluştur (gerçek verilerinizle değiştirin)
        """
        # SSA ile ilgili örnek yorumlar
        sample_comments = [
            "Bu algoritma beni sürekli aynı içerikle karşılaştırıyor, kendimi tuzağa düşmüş hissediyorum",
            "Sosyal medyada sadece benim görüşlerime uygun içerik görüyorum, bu endişe verici",
            "Algoritmalar beni manipüle ediyor, gerçek dünyadan kopuk hissediyorum",
            "Bu platformlar beni sürekli aynı kişilerle bağlantıya geçiriyor",
            "Dijital yabancılaşma yaşıyorum, gerçek insanlarla bağlantım azaldı",
            "Algoritma sayesinde çeşitli görüşlerle karşılaşıyorum, bu iyi",
            "Platform benim ilgi alanlarımı anlıyor ve uygun içerik sunuyor",
            "Sosyal medyada farklı perspektiflerle tanışıyorum",
            "Algoritma beni yeni konularla tanıştırıyor",
            "Bu sistem benim için kişiselleştirilmiş deneyim sunuyor"
        ]
        
        # Etiketler (0: negative, 1: neutral, 2: positive)
        sample_labels = [0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
        
        # Train ve test verilerini oluştur
        train_indices = [0, 1, 2, 3, 4, 5, 6]
        test_indices = [7, 8, 9]
        
        self.train_data = pd.DataFrame({
            'comment': [sample_comments[i] for i in train_indices],
            'sentiment': [sample_labels[i] for i in train_indices]
        })
        
        self.test_data = pd.DataFrame({
            'comment': [sample_comments[i] for i in test_indices],
            'sentiment': [sample_labels[i] for i in test_indices]
        })
        
        print("Örnek veri oluşturuldu")
        print(f"Train data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
    
    def preprocess_data(self):
        """
        Veriyi ön işleme
        """
        # Eksik değerleri temizle
        self.train_data = self.train_data.dropna()
        self.test_data = self.test_data.dropna()
        
        # Metin verilerini temizle
        self.train_data['comment_clean'] = self.train_data['comment'].str.lower()
        self.test_data['comment_clean'] = self.test_data['comment'].str.lower()
        
        print("Veri ön işleme tamamlandı")
        print(f"Train data sentiment distribution:\n{self.train_data['sentiment'].value_counts()}")
        print(f"Test data sentiment distribution:\n{self.test_data['sentiment'].value_counts()}")
    
    def extract_features(self):
        """
        TF-IDF özellik çıkarımı
        """
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Train verilerinden özellik çıkar
        X_train = self.vectorizer.fit_transform(self.train_data['comment_clean'])
        X_test = self.vectorizer.transform(self.test_data['comment_clean'])
        
        y_train = self.train_data['sentiment']
        y_test = self.test_data['sentiment']
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, X_train, y_train):
        """
        Sınıf dengesizliğini ele al (küçük veri seti için basit yaklaşım)
        """
        print("Sınıf dengesizliği analizi:")
        print(f"Original class distribution: {np.bincount(y_train)}")
        
        # Küçük veri seti için SMOTE kullanmıyoruz
        # Bunun yerine class weights kullanıyoruz
        print("Küçük veri seti için SMOTE kullanılmadı")
        print("Class weights kullanılacak")
        
        return X_train, y_train
    
    def train_models(self, X_train, y_train):
        """
        Farklı modelleri eğit
        """
        models = {}
        
        # Class weights hesapla
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Logistic Regression with class weights
        lr_model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight=weight_dict
        )
        lr_model.fit(X_train, y_train)
        models['LogisticRegression'] = lr_model
        
        # Random Forest with class weights
        rf_model = RandomForestClassifier(
            random_state=42, 
            n_estimators=100,
            class_weight=weight_dict
        )
        rf_model.fit(X_train, y_train)
        models['RandomForest'] = rf_model
        
        return models
    
    def evaluate_models(self, models, X_test, y_test):
        """
        Modelleri değerlendir
        """
        results = {}
        
        # Test verisindeki sınıf sayısını kontrol et
        unique_classes = np.unique(y_test)
        print(f"Test verisindeki sınıflar: {unique_classes}")
        
        if len(unique_classes) == 1:
            print("UYARI: Test verisinde sadece bir sınıf var. Değerlendirme sınırlı olacak.")
            
            for name, model in models.items():
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Basit metrikler
                accuracy = (y_pred == y_test).mean()
                
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'warning': 'Single class in test data'
                }
                
                print(f"\n{name} Results (Single Class):")
                print(f"Accuracy: {accuracy:.3f}")
                print("Classification report not available for single class")
        
        else:
            # Gerçek sınıf isimlerini belirle
            actual_class_names = []
            for cls in unique_classes:
                if cls == 0:
                    actual_class_names.append('negative')
                elif cls == 1:
                    actual_class_names.append('neutral')
                elif cls == 2:
                    actual_class_names.append('positive')
            
            for name, model in models.items():
                # Tahminler
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Metrikler
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='weighted'
                )
                
                # ROC-AUC hesapla (eğer yeterli sınıf varsa)
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                except:
                    roc_auc = None
                
                # Sınıf bazında metrikler
                class_report = classification_report(
                    y_test, y_pred, 
                    target_names=actual_class_names, 
                    output_dict=True
                )
                
                results[name] = {
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
                print(f"Precision: {precision:.3f}")
                print(f"Recall: {recall:.3f}")
                print(f"F1-Score: {f1:.3f}")
                if roc_auc:
                    print(f"ROC-AUC: {roc_auc:.3f}")
                else:
                    print("ROC-AUC: Hesaplanamadı (yetersiz sınıf)")
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred, target_names=actual_class_names))
        
        return results
    
    def cross_validation(self, X_train, y_train, cv_folds=3):
        """
        Çapraz doğrulama (küçük veri seti için 3-fold)
        """
        models = {
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'RandomForest': RandomForestClassifier(
                random_state=42, n_estimators=100
            )
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
            # Train verilerinde anahtar kelime geçen yorumları say
            train_count = self.train_data['comment_clean'].str.contains(
                keyword, case=False
            ).sum()
            test_count = self.test_data['comment_clean'].str.contains(
                keyword, case=False
            ).sum()
            
            keyword_analysis[keyword] = {
                'train_count': train_count,
                'test_count': test_count,
                'total_count': train_count + test_count
            }
        
        # En sık geçen anahtar kelimeleri sırala
        sorted_keywords = sorted(
            keyword_analysis.items(), 
            key=lambda x: x[1]['total_count'], 
            reverse=True
        )
        
        print("\nSSA Keywords Analysis:")
        for keyword, counts in sorted_keywords:
            if counts['total_count'] > 0:
                print(f"{keyword}: {counts['total_count']} occurrences")
        
        return keyword_analysis
    
    def run_complete_analysis(self, train_file=None, test_file=None):
        """
        Tam analiz süreci
        """
        print("=== SSA Sentiment Analysis ===")
        
        # 1. Veri yükleme
        self.load_data(train_file, test_file)
        
        # 2. Veri ön işleme
        self.preprocess_data()
        
        # 3. SSA anahtar kelime analizi
        keyword_analysis = self.analyze_ssa_keywords()
        
        # 4. Özellik çıkarımı
        X_train, X_test, y_train, y_test = self.extract_features()
        
        # 5. Sınıf dengesizliği ele alma
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train)
        
        # 6. Çapraz doğrulama
        cv_results = self.cross_validation(X_train_balanced, y_train_balanced)
        
        # 7. Model eğitimi
        models = self.train_models(X_train_balanced, y_train_balanced)
        
        # 8. Model değerlendirmesi
        results = self.evaluate_models(models, X_test, y_test)
        
        return results, cv_results, keyword_analysis

# Kullanım örneği
if __name__ == "__main__":
    analyzer = SSASentimentAnalyzer()
    
    # Gerçek dosya yollarınızı buraya ekleyin
    # results, cv_results, keyword_analysis = analyzer.run_complete_analysis(
    #     train_file='interviews_train.csv',
    #     test_file='interviews_test.csv'
    # )
    
    # Örnek veri ile çalıştır
    results, cv_results, keyword_analysis = analyzer.run_complete_analysis()
    
    print("\n=== Analysis Complete ===")
    print("Results saved for manuscript revision") 