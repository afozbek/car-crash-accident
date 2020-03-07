# Car Crash Accident Project

Yüklemeniz için gerekli olan paketler `requirements.txt` de mevcut.

- Virtual env oluşturun (Python 3.6)
- Proje dizininde iken `pip install -r requirements.txt` komudu ile gerekli paketleri yükleyin

**NOT:** 
- Opencv yi tabiki source dan build etmelisiniz.
- yolov3.weights dosyalarını ana dizine indirmelisiniz. [Link to yolov3.weight file](https://pjreddie.com/media/files/yolov3.weights)

- Eğer kütüphane kaynaklı sıkıntı olursa;
  - Darknet'i build ettikten sonra çıkan libdark.so yu kopyalayın
  - Bu projenin ana dizinine yapıştırıp, ismini `libdarknet.so` olarak güncelleyin
  - Sorununuz tekrar ederse Google'layın?

## Development
Development için **branch** oluşturarak gidiyoruz arkadaşlar. Master dan branch oluşturun. Geliştirme yapın sonra **pull request** oluşturun. 

## Brach Yapısı
master branch ine dokunmuyoruz arkadaşlar. Readme güncellemeleri falan için yapıcaksanız güncelleyin. Diğer türlü yapı şu şekilde
- master
  - development
    - develop_furkan
    - develop_sena
    - develop_sevval
şeklinde olacaktır

## Authors
- Abdullah Furkan Özbek
- Ayşe Sena Modanlıoğlu
- Şevval
