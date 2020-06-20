# Car Crash Accident Project

Projemiz, tespit edilen nesnelerin koordinatların kesişme durumuna bakarak kaza tespiti yapmaktadır. Kaza tespiti yapmak için Darknet YOLO V3 kullanılmıştır. Otomobil, motosiklet, bisiklet ve otobüs koordinatlarına bakarak kaza tespiti yapılmaktadır. Algoritma "T" şeklindeki kazalarda, tek şeritli yolda gündüz vakti çekilmiş kaza videolarında düzgün çalışmaktadır.

## Projeyi kendi bilgisayarınızda çalıştırmak için

- [Darknet'i](https://github.com/AlexeyAB/darknet) bilgisayarınıza kurun.
- Darknet'i build ettikten sonra oluşan "darknet.so" dosyasını proje dizinine yapıştırıp dosya adını "libdarknet.so" olarak değiştirin.
- Virtual environment oluşturun (Python 3.6)
- Yüklemeniz için gerekli olan kütüphaneler `requirements.txt` de mevcut. Proje dizininde iken `pip install -r requirements.txt` komudu ile gerekli paketleri yükleyin
- [YOLOV3 weight dosyasını](https://pjreddie.com/media/files/yolov3.weights) proje dizinine indirin.

**NOT:** Opencv yi tabiki source dan build etmelisiniz. OpenCV'yi virtual enviroment'a eklemeyi unutmayın.

### Projeyi çalıştırmak için

```
python darknet_video.py --no-rec
```

Örnek kaza videolarını [linkten](https://drive.google.com/drive/folders/1lm260ufeMltoX2tUBRl1xpYUffo5p4Vc?usp=sharing) bulabilirsiniz.

![](https://github.com/afozbek/car-crash-accident/blob/son_hali/data/kaza_1.gif)
![](https://github.com/afozbek/car-crash-accident/blob/son_hali/data/kaza_2.gif)

## Authors

- Abdullah Furkan Özbek
- Ayşe Sena Modanlıoğlu
- Şevval Didem Değer
