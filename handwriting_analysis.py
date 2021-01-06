from sklearn import datasets  # データセットのインポート
from matplotlib import pyplot as plt, cm

digits = datasets.load_digits()  # 手書き数字データ
data=digits.images[0]

plt.imshow(data.reshape(8,8),cmap=cm.gray_r,interpolation="nearest")
plt.show()

