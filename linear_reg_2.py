'''

    Linear Regression 2 Variabel
    Persamaan Umum : b0 + b1*x1 + b2*x2
    
    Program by prokoding

'''
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt

#mengambil data dari stok hp
df = pd.read_csv('stock_hp.csv')
print(df.head())

class Multi_Regression:

    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def start(self):
        n = len(self.x1)
        #sigma x1, x2, y
        s_x1 = sum(self.x1)
        s_x2 = sum(self.x2)
        s_y = sum(self.y)
        
        #sigma x1^2, x2^2
        quad_x1 = [x1**2 for x1 in self.x1]
        s_quad_x1 = sum(quad_x1)
        quad_x2 = [x2**2 for x2 in self.x2]
        s_quad_x2 = sum(quad_x2)
        
        #sigma x1*y, x2*y
        x1_y = [self.x1[i]*self.y[i] for i in range(len(self.x1))]
        s_x1_y = sum(x1_y)
        x2_y = [self.x2[i]*self.y[i] for i in range(len(self.x1))]
        s_x2_y = sum(x2_y)
        
        #sigma x1 * x2
        x1_x2 = [self.x1[i] * self.x2[i] for i in range(len(self.x1))]
        s_x1_x2 = sum(x1_x2)

        #operasi matriks
        left_side = np.array([[n, s_x1, s_x2],[s_x1, s_quad_x1, s_x1_x2],[s_x2, s_x1_x2, s_quad_x2]])
        right_side = np.array([s_y, s_x1_y, s_x2_y])
        hasil = np.linalg.solve(left_side, right_side)
        
        b0 = hasil[0]
        b1 = hasil[1]
        b2 = hasil[2]
        return b0, b1, b2


clf = Multi_Regression(df['battery(mAh)'],df['kamera(MP)'],df['Harga(juta)'])
b0, b1, b2 = clf.start()

battery = int(input('Masukan daya tahan baterry (mAh) : '))
kamera = int(input('Masukan kualitas kamera (MP) : '))
predict = b0 + (b1 * battery) + (b2 * kamera)
print('Harga yang cocok untuk memasarkan hp dengan battery ',battery,'000 mAh',' dan kualitas kamera ',kamera,'MP','adalah : ',round(predict,2))
