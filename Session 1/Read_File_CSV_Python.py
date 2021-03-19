import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
dt = pd.read_csv("baitap1.csv", delimiter=',')
#Hien_thi_du_lieu_"baitap1.csv"
print("\n2:")
print(dt)
#Hien_thi_cot_2
print("\n3:")
print(dt.iloc[:, 2:3])
#Hien_thi_tu_dong7_dong13
print("\n4:")
print((dt.loc[7:13]))
#Hien_thi_du_lieu_cot12_cua_dong_5
print("\n5:")
print(dt.iloc[5, 1:3])
#Tao_bien_X=>du_lieu_cot_2, Y=>du_lieu_cot_3
x = dt.iloc[:, 1]
y = dt.iloc[:, 2]
plt.scatter(x,y)
plt.title("BIỂU ĐỒ THỂ HIỆN SỰ TƯƠNG QUAN GIỮA TUỔI VÀ CÂN NẶNG")
plt.xlabel("TUỔI")
plt.ylabel("CÂN NẶNG")
plt.grid()
plt.show()
#In_so_chan
print("\n7:")
for i in range(0, 101):
    if i % 2 == 0:
        print(i)

