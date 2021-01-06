# 予測モデルの作成

import joblib
from sklearn import datasets, svm

DIGITS_PKL = "digit-clf.pkl"  # モデルデータファイル名


def train_digits():
    digits = datasets.load_digits()  # 手書き数字データ
    data_train = digits.data
    label_train = digits.target
    clf = svm.SVC(gamma=0.001)
    clf.fit(data_train, label_train)
    joblib.dump(clf, DIGITS_PKL)
    print("予測モデルを保存しました= ", DIGITS_PKL)
    return clf
