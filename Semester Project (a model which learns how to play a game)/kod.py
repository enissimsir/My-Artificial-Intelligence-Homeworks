import math
import time
import numpy as np
import pygame
import sys
import random
from math import *
import tensorflow as tf
import matplotlib.pyplot as plt
#Oyun yapımında kullanılacak pygame kütüphanesi başlatılıyor

pygame.init()

#Ekranda çıkacak görüntünün genişliği ve yüksekliği
width = 700
height = 600
display = pygame.display.set_mode((width, height)) # genişlik ve yüksekliğe göre pygame modu ayarlanır

pygame.display.set_caption("Balon Vurma Oyunu") #Oyun adı başlığa yazılır

clock = pygame.time.Clock() # zaman tutulur

#balonların gezindiği bölge
margin = 100
lowerBound = 100

score = 0 # skoru takip eder
patlatilan = 0

# Grafik icin
scores = []

#Renk kodları RGB moda göre ayarlanır
white = (230, 230, 230)
lightBlue = (4, 27, 96)
red = (231, 76, 60)
lightGreen = (25, 111, 61)
darkGray = (40, 55, 71)
darkBlue = (64, 178, 239)
green = (35, 155, 86)
yellow = (244, 208, 63)
blue = (46, 134, 193)
purple = (155, 89, 182)
orange = (243, 156, 18)

#patlatılan balon sayısı yazısını tutacak pygame modu ayarlanır
font = pygame.font.SysFont("Arial", 25)

balloons = []#balonların tutulduğu dizi
noBalloon = 15 #balon sayısı


class Balloon:
    def __init__(self, speed): #hız değişkenini de atamak için init fonksiyonu kullanıldı        
        self.a = random.randint(30, 40) #değişkenlere erişmek için self a=balon bouyutu
        self.b = self.a + random.randint(0, 10)#balon boyutu
        self.x = random.randrange(margin, width - self.a - margin)#balon x konumu margin ile genişlik arasında  
        self.y = height - lowerBound#başlangıçta balon lowerbounddan yüksekte pencerenin altında başlar
        self.angle = 90#90 derece ile düz yukarı hareket
        self.speed = -speed#eksi yaparak balon yukarıya doğru hareket edecek
        self.proPool= [-1, -1, -1, 0, 0, 0, 0, 1, 1, 1]#olasılık havuzu gideceği açıyı belirleyecek
        self.length = random.randint(50, 100) #balon uzunluğu
        self.color = random.choice([red, green, purple, orange, yellow, blue]) #balonları rengi ayarlanır
        
        
    def move(self):
        #balonun hareket yönü için rastgele değer 
        direct = random.choice(self.proPool)

        #açı bu rastgele değere göre değiştirilir
        if direct == -1:
            self.angle += -10
        elif direct == 0:
            self.angle += 0
        else:
            self.angle += 10

        #x ve y bu koordinatlara göre güncellenir
        self.y += self.speed*sin(radians(self.angle))
        self.x += self.speed*cos(radians(self.angle))

        
        #balonun ekrandan çıkmaması,taşmaması için kontrol
        if (self.x + self.a > width) or (self.x < 0):
            if self.y > height/5:
                self.x -= self.speed*cos(radians(self.angle)) 
            else:
                self.reset()
        if self.y + self.b < 0 or self.y > height + 30:
            self.reset()
            
    #balonların görüntülenmesi
    def show(self):
        #balon ipi 
        pygame.draw.line(display, darkBlue, (self.x + self.a/2, self.y + self.b), (self.x + self.a/2, self.y + self.b + self.length))
        #balonun kendisi
        pygame.draw.ellipse(display, self.color, (self.x, self.y, self.a, self.b))
        pygame.draw.ellipse(display, self.color, (self.x + self.a/2 - 5, self.y + self.b - 3, 10, 10))
    
    #balonun patlayıp patlamaması
    #pos fare imleci eğer içerideyse +1 ve reset
    def burst(self):
        global score
        global patlatilan
        pos=pygame.mouse.get_pos()#fare imlecinin yerini alır

        if isonBalloon(self.x, self.y, self.a, self.b, pos):
            score += 20
            patlatilan += 1
            self.reset()
            
            
    #balonun başlangıç durumuna dönmesini sağlar
    def reset(self):
        self.a = random.randint(30, 40)
        self.b = self.a + random.randint(0, 10)
        self.x = random.randrange(margin, width - self.a - margin)
        self.y = height - lowerBound 
        self.angle = 90
        self.speed -= 0.002
        self.proPool = [-1, -1, -1, 0, 0, 0, 0, 1, 1, 1]
        self.length = random.randint(50, 100)
        self.color = random.choice([red, green, purple, orange, yellow, blue])
        
#noktanın balonun içinde olup olmadığı kontrolü
#a ve b balonun genişlik ve yükseklik pos kontrol noktası x ve y sol üst koşe
def isonBalloon(x, y, a, b, pos):
    if (x < pos[0] < x + a) and (y < pos[1] < y + b):
        return True
    else:
        return False
        
        
#fare imleci yanında işaret 
def pointer():
    pos = pygame.mouse.get_pos()#fare imlecinin yerini alır
    r = 25#işaretin boyutu
    l = 20#işaretin boyutu
    color = lightGreen
    for i in range(noBalloon):#balonların üzerinde imleç var mı kontrolü
        if isonBalloon(balloons[i].x, balloons[i].y, balloons[i].a, balloons[i].b, pos):
            color = red
    pygame.draw.ellipse(display, color, (pos[0] - r/2, pos[1] - r/2, r, r), 4)
    pygame.draw.line(display, color, (pos[0], pos[1] - l/2), (pos[0], pos[1] - l), 4)
    pygame.draw.line(display, color, (pos[0] + l/2, pos[1]), (pos[0] + l, pos[1]), 4)
    pygame.draw.line(display, color, (pos[0], pos[1] + l/2), (pos[0], pos[1] + l), 4)
    pygame.draw.line(display, color, (pos[0] - l/2, pos[1]), (pos[0] - l, pos[1]), 4)

#skorun olduğu platform
def lowerPlatform():
    pygame.draw.rect(display, darkGray, (0, height - lowerBound, width, lowerBound))
    
def showScore():
    scoreText = font.render("Skor : " + str(score), True, white)
    patlatilanText = font.render("Patlatilan: " + str(patlatilan), True, white)
    display.blit(scoreText, (150, height - lowerBound + 50))
    display.blit(patlatilanText, (450, height - lowerBound + 50))
    
def close():
    pygame.quit()
    sys.exit()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.9  # Gelecekteki ödüller için indirim değeri
        self.epsilon = 1.0  # Keşfetme ve sömürme dengesi için epsilon değeri
        self.epsilon_decay = 0.998  # Her adımda epsilon'un azalma hızı
        self.epsilon_min = 0.01  # Minimum epsilon değeri
        self.learning_rate = 0.001  # Öğrenme oranı

        self.model = self.build_model()

    #model oluşturulur
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer,loss='mse')    
        return model

    #hafızaya state kayıt edilir
    def remember(self, state, action, score, next_state, done):
        self.memory.append((state, action, score, next_state, done))

    #eylem seçer
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Keşfetme: Rastgele bir eylem seç
            return random.randint(0, self.action_size - 1)
        else:
            # Sömürme: En iyi tahmini eylemi seç
            q_values = self.model.predict(state)[0]
            if q_values[0] == np.max(q_values):
                # Yukarıya hareket et
                return 0
            elif q_values[1] == np.max(q_values):
                # Aşağıya hareket et
                return 1
            elif q_values[2] == np.max(q_values):
                # Sola hareket et
                return 2
            else:
                # Sağa hareket et
                return 3

    #Bu fonksiyon, deney belleğinden rastgele örnekler seçer,
    #hedef Q değerlerini günceller ve bu güncellenmiş örneklerle modeli eğitir.
    def replay(self, batch_size):
        #İlk olarak, deney belleğinden batch_size kadar örnek seçilir. Self memoryde geçmiş deneyimler
        minibatch = random.sample(self.memory, batch_size)

        states = []#eğitim örneklerini tutmak için 
        targets = []#hedef çıktılar için
        #her bir seçilen örnek için
        for state, action, score, next_state, done in minibatch:
            target = score
            if not done:
                # Q değerini güncelle,sonraki durum için tahmini Q değerleri
                target = (score + self.gamma * np.amax(self.model.predict(next_state)[0]))
            #tahmini Q değeri atılır
            target_f = self.model.predict(state)[0]
            #aksiyona target atılır
            target_f[action] = target
            #mevcut durum
            states.append(state[0])
            #güncellenen hedef Q değeri
            targets.append(target_f)

        # Modeli eğit
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

    # zamanla kendi öğrendiklerini kullanmak istediğiminz için Epsilon değerini azaltılır
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name) 

def game():
    #noBalloon değişkeni 10 olarak ayarlanır ve toplamda oluşturulacak balon sayısını belirler.
    #rastgele hız değeri seçilir
    for i in range(noBalloon):
        obj = Balloon(random.choice([1, 1, 2, 2, 2, 2, 3, 3, 3, 4]))
        balloons.append(obj)
    
    frame=0
    global score
    global patlatilan
    global scores
    loop = True#loop false olana akdar oyun devam eder
    batch_size = 32
    agent = DQNAgent(state_size=4, action_size=4)#action sağ sol yukarı aşağı 4 tane #state x,y,farex,farey

    while loop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:#pencere kapatma tuşuna basılırsa kapanır
                close()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:#q tuşu oyunu kapatır
                    plt.plot(scores)
                    plt.xlabel('Oyun Döngüsü')
                    plt.ylabel('Skor')
                    plt.title('Skor Değişimi')
                    plt.show()
                    close()
                if event.key == pygame.K_r:#r tuşu oyunu resetler
                    score = 0
                    patlatilan = 0
                    game()

        scores.append(patlatilan)      
        frame+=1
        
        #Find the balloon closest to the mouse cursor
        min_distance = float('inf')
        closest_balloon = None
        mouse_x, mouse_y = pygame.mouse.get_pos()
        
        for balloon in balloons:
            distance = math.sqrt((balloon.x - mouse_x)**2 + (balloon.y - mouse_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_balloon = balloon
                
        if closest_balloon is not None:
            if min_distance <= 200:  # Adjust the threshold as per your preference
                score += 1
            else:
                score -= 1
                
            edge_threshold = 40  # Adjust the threshold as per your preference
            
            if mouse_x < edge_threshold:
                mouse_x = width / 2
            elif mouse_x > width - edge_threshold:
                mouse_x = width / 2
            
            if mouse_y < edge_threshold:
                mouse_y = height / 2
            elif mouse_y > height - edge_threshold:
                mouse_y = height / 2
                
            pygame.mouse.set_pos((mouse_x, mouse_y))
            # Mevcut durumu al
            state = [closest_balloon.x, closest_balloon.y, pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]]
            state = np.reshape(state, [1, agent.state_size])
            
            for i in range(noBalloon):
                balloons[i].burst()
            #eylemi uygula ve yeni durumu al
            action = agent.act(state)
            if action == 0:
                new_x = pygame.mouse.get_pos()[0]
                new_y = pygame.mouse.get_pos()[1] + 30
                pygame.mouse.set_pos((new_x, new_y))
            elif action == 1:
                new_x = pygame.mouse.get_pos()[0]
                new_y = pygame.mouse.get_pos()[1] - 30
                pygame.mouse.set_pos((new_x, new_y))
            elif action == 2:
                new_x = pygame.mouse.get_pos()[0] - 30
                new_y = pygame.mouse.get_pos()[1]
                pygame.mouse.set_pos((new_x, new_y))
            else:
                new_x = pygame.mouse.get_pos()[0] + 30
                new_y = pygame.mouse.get_pos()[1]
                pygame.mouse.set_pos((new_x, new_y))
                
            
            next_state = [closest_balloon.x, closest_balloon.y, pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]]
            next_state = np.reshape(next_state, [1, agent.state_size])
            
            # if frame % (batch_size/2)==0:
                # Belleğe ekle
            agent.remember(state, action, score, next_state, loop)
            # Bellekten öğren
            if frame%30==0 and len(agent.memory) > batch_size:
                agent.replay(batch_size)


        # Ekranı güncelle
        display.fill(lightBlue)
        
        for i in range(noBalloon):#balonların ekranda gösterilmesi
            balloons[i].show()
            
        pointer()#fare imleci işareti için
        
        for i in range(noBalloon):#balonların hareketi
            balloons[i].move()
            
        # lowerPlatform()
        showScore()
        pygame.display.update()
        clock.tick(60)  

game()