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
