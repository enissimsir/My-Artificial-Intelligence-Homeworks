import csv
import string
import random
import time
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt

N = 40 #birey sayisi
M = 20 #eleman sayisi

#jenerasyonun ortalama fitnessinin tutmak için gerekli dizi

avgFitness=[]

#benzersiz kelimeleri tutmak için gerekli liste
unique_words = {}

#kelimeleri noktalama işaretlerinden ve sayılardan arındırıp tutmak için gerekli dizi
cleaned_words = []

#en başarılı bireylerin tutulduğu dizi
mostSuccesfullPeople=[]

def remove_punctuation(text):
    #Bir stringden noktalama işaretlerini kaldirir.
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


#Bu fonksiyon, her bir bireyin ilk yarısının olumlu ve son yarısını olumsuz olup olmanasına göre fitness değeri hesaplar.
#Bireyin elemanları metin dizisinin bütün satırındaki metinlerle karşılaştırır ve eğer bireyin ilk yarısında bu metindeki
#bir kelime geçiyor ise s1counter'i ikinci yarısında ise s2counter'i arttırır. Doğruluğun artması için metin içerisindeki
# 1,2 ve 3 harfli kelimeler alınmaz. Eğer s1 büyük ve kelime olumlu ise doğru hesaplanmıştır ve fitness değeri arttırılır.
# Eğer s2counter büyük ve olumsuz ise doğru hesaplanmıştır ve fitness değeri arttırılır.
def calculate_fitness(sublists, df):
    fitness = [0] * len(sublists)

    for birey_numarasi, sublist in enumerate(sublists):
        for i in range(999):
            s1Counter=0
            s2Counter=0
            text=remove_punctuation(df.iloc[i,0].lower())
            for j, word in enumerate(sublist[:int(M/2)]):
                if len(word) != 1 and len(word) != 2 and len(word) != 3 and word in text:
                    s1Counter += 1
            for j, word in enumerate(sublist[((-1)*(int(M/2))):]):
                if len(word) != 1 and len(word) != 2 and len(word) != 3 and word in text:
                    s2Counter += 1
            if df.iloc[i,1] == 1 and s1Counter>s2Counter:
                fitness[birey_numarasi] += 1
            elif df.iloc[i,1] == 0 and s2Counter>s1Counter:
                fitness[birey_numarasi] += 1

    return fitness

#Bu fonksiyon, verilen bireyi, kelime listesi ve fitness değeri kullanarak mutasyona uğratır
# Mutasyon sonucu elde edilen yeni bireyi döndürür. Mutasyon oranı fitness değerine bağlı olarak belirlenir
# ve bireyin her bir kelimesi, belirtilen mutasyon oranı üzerinden rastgele seçilen bir kelime ile değiştirilebilir.
def mutation(birey, word_list, fitness,total_fitness):

    fitnessRate=(fitness/total_fitness)*100
    mutation_rate = 1 / fitnessRate
    
    for i in range(len(birey)):
        if random.uniform(0, 1) < mutation_rate:
            # rastgele bir kelime seç
            random_word = random.choice(word_list)
            # kelimeyi bireyde değiştir
            birey[i] = random_word
    return birey


def createTable(sublists,i):
    table = PrettyTable()
    table.add_column("Positive Words", sublists[i][:int(M/2)])
    table.add_column("Negative Words", sublists[i][int(M/2):])
    print(table)

#Bu kod parçası, "amazon_cells_labelled.csv" adlı dosyadaki her bir satırı okur
#her bir satırdaki kelimeleri ayırarak, her kelimeyi bir sözlükte tutar.
#Bu sözlük, kelimelerin kaç kez geçtiğini sayar. Eğer bir kelime daha önce sözlükte yoksa,
# yeni bir girdi olarak eklenir ve sayacı 1 olarak ayarlanır.
# Eğer kelime sözlükte varsa, sayacı 1 artırılır.
# Sonuç olarak, sözlük, dosyadaki tüm kelimelerin birer birer tekrar sayıları ile birlikte toplandığı bir şekilde doldurulmuş olur.
with open('amazon_cells_labelled.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        words = row[0].split()
        for word in words:
            if word not in unique_words:
                unique_words[word] = 0
            unique_words[word] += 1

#sadece 68 defadan az geçen kelimeleri sözlüğe ekle doğruluğun arttırılması için
filtered_words = {word: count for word, count in unique_words.items() if count < 68}
# her birini tekil anahtar olarak kaydet
word_list = list(filtered_words.keys())


for word in word_list:
    # Sözcüğü küçük harflere dönüştür
    word = word.lower()

    # Noktalama işaretlerini kaldır
    word = word.translate(str.maketrans('', '', string.punctuation))

    # Sayilari kaldir
    word = ''.join([i for i in word if not i.isnumeric()])

    cleaned_words.append(word)

#boşlukları kaldır
final_words = list(filter(lambda x: x != '', cleaned_words))

# kelimelerii rastgele karıştır
random.seed(time.time())    
random.shuffle(final_words)

# N adet M elemanlı alt-liste oluştur
sublists = [final_words[i:i+M] for i in range(0, N*M,M)] 
# ikinci parametre = birey sayisi*eleman sayisi, son parametre = eleman sayisi


df = pd.read_csv('amazon_cells_labelled.csv')


fitness = calculate_fitness(sublists, df)

createTable(sublists,fitness.index(max(fitness)))

generation_count = 0 # jenerasyon sayısını takip etmek için
while generation_count!=20: 
    
    generation_count += 1 # her döngüde jenerasyon sayısını 1 artır
    print("Generation: ", generation_count)
    
    #fitness değerini hesapla
    fitness = calculate_fitness(sublists, df)
    print("Max fitness: ", max(fitness))  
    
    avgFitness.append(sum(fitness) / len(fitness))
    mostSuccesfullPeople.append(max(fitness))

    #burada crossover işlemleri gerçekleştirilir. parent1 ve parent2 fitness ağırlık değerlerine göre seçilir.
    #eğer parentlar aynı gelirse değiştirilir. Rastgeler bir crossover point hesaplanır ve crossover işlemi 
    # gerçekleştirilere çocuk oluşur, yeni jenerasyona eklenir.
    #Yeni jenerasyonun fitness değeri hesaplanır ve fitness değerine göre mutasyona gönderilir.
    new_generation = []
    random.seed(time.time())    
    for i in range(0, len(sublists)):
        parent1 = random.choices(sublists, weights=fitness)
        parent2 = random.choices(sublists, weights=fitness)
        while parent2 == parent1:
            if random.uniform(0, 1) <= 0.5:
                parent1 = random.choices(sublists, weights=fitness)
            else:
                parent2 = random.choices(sublists, weights=fitness)
        crossover_point = random.randint(3, M-3)
        child = parent1[0][:crossover_point] + parent2[0][crossover_point:]
        new_generation.append(child)

    lastFitness=calculate_fitness(new_generation, df)
    total_fitness = sum(lastFitness)  
    random.seed(time.time())
    for i, birey in enumerate(new_generation):
        if random.uniform(0, 1) <= 0.2:
            birey=mutation(birey,final_words,lastFitness[i],total_fitness)

    sublists=new_generation

createTable(sublists,fitness.index(max(fitness)))
    
# Grafik oluşturma
plt.plot(avgFitness, label='Average Fitness')
plt.plot(mostSuccesfullPeople, label='Most Successful People')

# Eksen ve başlık etiketleri
plt.xlabel('Generation')
plt.ylabel('Value')
plt.title('Fitness and Success over Generations')

# Gösterim
plt.legend()
plt.show()