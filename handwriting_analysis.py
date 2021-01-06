import os, sys, math
import joblib
from sklearn import datasets, svm

DIGITS_PKL = "digit-clf.pkl"  # モデルデータファイル名


# 予測モデルの作成
def train_digits():
    digits = datasets.load_digits()  # 手書き数字データ
    data_train = digits.data
    label_train = digits.target
    clf = svm.SVC(gamma=0.001)
    clf.fit(data_train, label_train)
    joblib.dump(clf, DIGITS_PKL)
    print("予測モデルを保存しました= ", DIGITS_PKL)
    return clf


# データから数字を予測する
def predict_digits(data: list):
    # 予測モデルファイルの読み込み
    if not os.path.exists(DIGITS_PKL):
        clf = train_digits()
    clf = joblib.load(DIGITS_PKL)
    # 予測
    n = clf.predict([data])
    print("判定結果= ", n)


# 手書き数字画像を8*8グレイスケールのデータ配列に変換
def image_to_data(name_imagefile: str):
    import numpy as np
    from PIL import Image
    image = Image.open(name_imagefile).convert("L")
    image = image.resize((8, 8), Image.ANTIALIAS)
    img = np.asarray(image, dtype=float)
    img = np.floor(16 - 16 * (img / 256))  # 行列演算
    # 変換後の画像表示(以下のコメント解除)
    # import matplotlib.pyplot as plt
    # plt.imshow(img)
    # plt.gray()
    # plt.show()
    img = img.flatten()
    print(img)
    return img


def main():
    # コマンドライン引数を得る
    if len(sys.argv) <= 1:
        print("USAGE:")
        print("python predict_digit.py imagefile")
        return
    imagefile = sys.argv[1]
    data = image_to_data(name_imagefile=imagefile)
    predict_digits(data)


if __name__ == "__main__":
    main()
