import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Bu işlem zaman alabilir. Lütfen bekleyiniz!")
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)

for test in range(len(x_test)):
    for row in range(28):
        for x in range(28):
            if x_test[test][row][x] != 0:
                x_test[test][row][x] = 1


model = tf.keras.models.load_model('Ogrenilen_Numaralar.model')#CreateModel üzerinden yeni bir model yaratılıp test edilebilir
print(len(x_test))
tahminlemeler = model.predict(x_test)

sayac = 0

for x in range(len(tahminlemeler)):
    tahmin = (np.argmax(tahminlemeler[x]))
    asil = y_test[x]
    print("Tahmin numaram şu:", tahmin)
    print("Numara aslında bu:", asil)
    if tahmin != asil:
        sayac+=1

    #plt.imshow(x_test[x], cmap=plt.cm.binary) #Tahmin edilen sayıların görüntüsü için comment kaldırılabilir
    #plt.show()

print("Program ",len(tahminlemeler),"testten", sayac,"yanlış yaptı")

print(str(100 - ((sayac/len(tahminlemeler))*100)) + '% doğru')