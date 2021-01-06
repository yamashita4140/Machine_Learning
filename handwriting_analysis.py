import os, sys
import joblib

from convert_image_to_data import image_to_data
from generate_model import train_digits

DIGITS_PKL = "digit-clf.pkl"  # モデルデータファイル名


# データから数字を予測する
def predict_digits(data: list):
    # 予測モデルファイルの読み込み
    if not os.path.exists(DIGITS_PKL):
        clf = train_digits()
    clf = joblib.load(DIGITS_PKL)
    # 予測
    n = clf.predict([data])
    print("判定結果= ", n)


def main():
    # コマンドライン引数を得る
    if len(sys.argv) <= 1:
        print("USAGE:")
        print("python handwriting_analysis.py imagefile")
        return
    imagefile = sys.argv[1]
    data = image_to_data(name_imagefile=imagefile)
    predict_digits(data)


if __name__ == "__main__":
    main()
