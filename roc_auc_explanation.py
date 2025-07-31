import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def explain_roc_auc_issue():
    """
    ROC-AUC hesaplama sorununu açıklar
    """
    print("=== ROC-AUC Hesaplama Sorunu Analizi ===\n")
    
    # Orijinal veri durumu
    print("1. ORİJİNAL VERİ DURUMU:")
    print("   - Toplam yorum: 190")
    print("   - Neutral sınıf: 134 yorum (%70.5)")
    print("   - Positive sınıf: 56 yorum (%29.5)")
    print("   - Negative sınıf: 0 yorum (%0)")
    print("   - Sınıf sayısı: 2 (neutral, positive)")
    print()
    
    # ROC-AUC hesaplama gereksinimleri
    print("2. ROC-AUC HESAPLAMA GEREKSİNİMLERİ:")
    print("   ✅ En az 2 sınıf olmalı")
    print("   ✅ Sınıflar arasında ayrım yapılabilmeli")
    print("   ✅ Olasılık tahminleri gerekli")
    print("   ✅ Sınıf dengesizliği çok yüksek olmamalı")
    print()
    
    # Orijinal veri sorunları
    print("3. ORİJİNAL VERİ SORUNLARI:")
    print("   ❌ Negative sınıf yok (0 yorum)")
    print("   ❌ Sınıf dengesizliği çok yüksek (%70.5 vs %29.5)")
    print("   ❌ Binary classification için uygun değil")
    print()
    
    # Sentetik veri durumu
    print("4. SENTETİK VERİ DURUMU:")
    print("   - Toplam yorum: 51")
    print("   - Negative sınıf: 20 yorum (%39.2)")
    print("   - Neutral sınıf: 16 yorum (%31.4)")
    print("   - Positive sınıf: 15 yorum (%29.4)")
    print("   - Sınıf sayısı: 3 (dengeli)")
    print("   ✅ ROC-AUC hesaplanabilir")
    print()
    
    # Çözüm önerileri
    print("5. ÇÖZÜM ÖNERİLERİ:")
    print("   A) Binary Classification:")
    print("      - Neutral vs Positive olarak yeniden etiketle")
    print("      - ROC-AUC hesaplanabilir")
    print()
    print("   B) Multi-class Classification:")
    print("      - Sentetik veri ekle (negative sınıf için)")
    print("      - 3 sınıflı ROC-AUC hesaplanabilir")
    print()
    print("   C) Alternative Metrics:")
    print("      - Precision, Recall, F1-Score kullan")
    print("      - Accuracy ile değerlendir")
    print()

def demonstrate_roc_auc_calculation():
    """
    ROC-AUC hesaplama örnekleri
    """
    print("=== ROC-AUC Hesaplama Örnekleri ===\n")
    
    # Örnek 1: Binary classification (başarılı)
    print("1. BINARY CLASSIFICATION (BAŞARILI):")
    y_true_binary = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred_proba_binary = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.1, 0.8])
    
    try:
        roc_auc_binary = roc_auc_score(y_true_binary, y_pred_proba_binary)
        print(f"   ROC-AUC: {roc_auc_binary:.3f} ✅")
    except Exception as e:
        print(f"   Hata: {e} ❌")
    print()
    
    # Örnek 2: Multi-class classification (başarılı)
    print("2. MULTI-CLASS CLASSIFICATION (BAŞARILI):")
    y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_proba_multi = np.array([
        [0.8, 0.1, 0.1],  # 0 sınıfı için yüksek olasılık
        [0.1, 0.8, 0.1],  # 1 sınıfı için yüksek olasılık
        [0.1, 0.1, 0.8],  # 2 sınıfı için yüksek olasılık
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9]
    ])
    
    try:
        roc_auc_multi = roc_auc_score(y_true_multi, y_pred_proba_multi, multi_class='ovr')
        print(f"   ROC-AUC: {roc_auc_multi:.3f} ✅")
    except Exception as e:
        print(f"   Hata: {e} ❌")
    print()
    
    # Örnek 3: Tek sınıf (başarısız)
    print("3. TEK SINIF (BAŞARISIZ):")
    y_true_single = np.array([1, 1, 1, 1, 1])  # Sadece 1 sınıfı
    y_pred_proba_single = np.array([0.8, 0.7, 0.9, 0.6, 0.8])
    
    try:
        roc_auc_single = roc_auc_score(y_true_single, y_pred_proba_single)
        print(f"   ROC-AUC: {roc_auc_single:.3f} ✅")
    except Exception as e:
        print(f"   Hata: {e} ❌")
    print()

def suggest_improvements():
    """
    ROC-AUC hesaplama için iyileştirme önerileri
    """
    print("=== İYİLEŞTİRME ÖNERİLERİ ===\n")
    
    print("1. BINARY CLASSIFICATION YAKLAŞIMI:")
    print("   - Neutral sınıfını 0, Positive sınıfını 1 olarak etiketle")
    print("   - ROC-AUC hesaplanabilir")
    print("   - Avantaj: Gerçek veri kullanılır")
    print("   - Dezavantaj: Sınıf dengesizliği devam eder")
    print()
    
    print("2. HYBRID APPROACH:")
    print("   - Orijinal veri + sentetik negative örnekler")
    print("   - 3 sınıflı dengeli veri seti")
    print("   - ROC-AUC hesaplanabilir")
    print("   - Avantaj: Dengeli sınıf dağılımı")
    print("   - Dezavantaj: Kısmen sentetik veri")
    print()
    
    print("3. ALTERNATIVE METRICS:")
    print("   - Precision, Recall, F1-Score")
    print("   - Accuracy")
    print("   - Confusion Matrix")
    print("   - Avantaj: Her durumda hesaplanabilir")
    print("   - Dezavantaj: ROC-AUC kadar kapsamlı değil")
    print()

if __name__ == "__main__":
    explain_roc_auc_issue()
    demonstrate_roc_auc_calculation()
    suggest_improvements() 