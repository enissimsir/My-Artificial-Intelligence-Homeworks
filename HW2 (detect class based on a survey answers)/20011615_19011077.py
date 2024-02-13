import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt


# Burada öncelikle file_path ile dosya yolu alınıp dosya okunmaya başlıyor. her satirda anket cevaplari
# ve son sütunda gerçek yanıt-->y değeri vardır. Her satır için anketteki tüm cevaplara ulaşmak amacıyla 
# satırdaki benzersiz cümleler alınır ve data_array dizisine atılır. Buradaki amaç benzersiz cümlelerin
# hepsine birer anahtar atayarak benzersiz cümleleri(cevapları) sayı dizisine çevirmektir. Yani vektörize
# işlemi yaparak string dizisini sayı dizisine çeviriyoruz. Aşağıdaki iç içe for döngüsü ile int_array
#dizisine int değerler atılır.
cumleler=[]
def csv_to_array(file_path):
    data_array = {}
    i=1
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            for cumle in row:
                if cumle not in data_array:
                    data_array[cumle]=i
                    i+=1
            cumleler.append(row)
    return data_array

# CSV dosyasinin yolu
csv_file_path = 'dataset.csv'


yanitlar = csv_to_array(csv_file_path)
int_array = np.zeros((len(cumleler), len(cumleler[0])), dtype=int)

#burada yukarıda fonksiyonda belirlediğimiz benzersiz yanıt anahtarlarını int_array dizisine atarak string
#dizisinin yeni integer dizisi halini elde ediyoruz
for i,x in enumerate(cumleler):
    for j,cumle in enumerate(x):
        int_array[i][j]=yanitlar[cumle]
            
        
        
X = int_array[:, :10]  # İlk 10 sütun özellikler
y = int_array[:, 10]   # 11. sütun hedef değisken
        
## train ve test olarak ayırım yapmamamızın sebebi aşağıdaki cross_val_score fonksiyonunda kendisinin
# otomatik olarak yapacak olmasıdır.

# Özellik seçimi için SelectKBest
k = 5
skb = SelectKBest(score_func=chi2, k=k)
X_selected = skb.fit_transform(X, y)

# Özellik dönüsümü için PCA nesnesi
n_components = 5
pca = PCA(n_components=n_components)
X_transformed = pca.fit_transform(X_selected)

# Min-Max normalizasyonu yapmak için MinMaxScaler sinifi
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_transformed)

# Tahminleyicileri olusturuluyor
tahminleyiciler = [
    LogisticRegression(),
    KNeighborsClassifier(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

# 10 katli çapraz geçerleme için KFold nesnesini olusturuluyor
kf = KFold(n_splits=10)

# Her tahminleyici için 10 katli çapraz geçerleme sonuçlarini hesaplanir. 

# cross_val_score ile her tahminleyicinin başarısı alınır ve sonuclar dizisine eklenir.
# bu fonksiyon verileri train ve test olarak 10 parçaya ayırır 9 tanesini eğitir ve 1 tanesini test ederek 
# sonucu döndürür 
sonuclar = []
for tahminleyici in tahminleyiciler:
    sonuclar_tahminleyici = cross_val_score(tahminleyici, X_normalized, y, cv=kf)
    sonuclar.append(sonuclar_tahminleyici)
    print("Tahminleyici: ", tahminleyici.__class__.__name__)
    print("Ortalama Doğruluk: ", np.mean(sonuclar_tahminleyici))
    print("Tüm Sonuçlar: ", sonuclar_tahminleyici)

# T-test kullanılarak tahminleyiciler arasındaki farklara bakılır
for i in range(len(tahminleyiciler)):
    for j in range(i + 1, len(tahminleyiciler)):
        t_statistic, p_value = stats.ttest_rel(sonuclar[i], sonuclar[j])
        print("Karsilastirma: ", tahminleyiciler[i].__class__.__name__, " vs. ", tahminleyiciler[j].__class__.__name__)
        print("T istatistiği: ", t_statistic)
        print("P değeri: ", p_value)
        if p_value < 0.05:
            print("İstatistiksel olarak anlamli fark var.")
        else:
            print("İstatistiksel olarak anlamli fark yok.")
        print("-------------------------------------------")
        
        
# Tahminleyicilerin isimleri
tahminleyici_isimleri = [tahminleyici.__class__.__name__ for tahminleyici in tahminleyiciler]

# Tahminleyicilerin performans sonuçlari
performans_sonuclari = [np.mean(sonuclar_tahminleyici) for sonuclar_tahminleyici in sonuclar]


# Performans sonuçlarini tablo olarak gösterin
performans_tablosu = pd.DataFrame({'Tahminleyici': tahminleyici_isimleri, 'Ortalama Doğruluk': performans_sonuclari})
print(performans_tablosu)


# Performans sonuçlarini bar grafikle gösterin
plt.figure(figsize=(10, 6))
plt.bar(tahminleyici_isimleri, performans_sonuclari)
plt.xlabel('Tahminleyiciler')
plt.ylabel('Ortalama Doğruluk')
plt.title('Tahminleyicilerin Performans Sonuçlari')
plt.xticks(rotation=45)
plt.show()

