Bitirme Çalışması İş Paketleri: 20 Nisan 2025 9.Hafta

*İnfrared filtresiz kamera adaptive thresholding ile infrared ışığık kaynağı tespit edildi (güneş biraz daha filtrelenebilir) ve feature extraction ile iki kamera görseli üst üste hizalandı.

*Tensorflow Lite ve Yolo11n hazır veriseti ile detection denendi, işe yaramaz. Bu yüzden 100 fotoğraflık bir veriseti oluşturup labellandı ve Yolo11n modeli üzerinde denendi.
Arkaplan uyuşmazlığı ve veriseti yetersiliğinden ötürü istenilenden uzak performans veriyor. Sunum yapılacak ortamda yeni veriseti oluşturulacak.

*Hailo NPU'sunda modeli koşturmayı öğren. NPU nasıl çalışıyor anla.

*KCF Tracker entegre edildi. Yolo modeli PyTorch'dan NCNN'e çevrildi (Raspberry'de daha hızlı). Tracker her yeni tespit edilen obje için initialize ediliyor ve track işlemi ana döngüden ayrı bir threadde koşuyor. Ayrıca confidance ve size'ı en yüksek objeler track ediliyor. Yolo, FPS'e bağlı olarak frame skipleyebiliyor. Böylece her saniye çalışmaktansa, örenğin 10 FPS görünyü için 10 frame'de bir çalışıyor (süreye bağlı değil frame'e bağlı)

*Binocular vision ile z hesaplanabilmesi için kameraların konfigüre edilmesi gerek. Bunun içi 8x6'lık dama tahtasının iki kamera ile de çeşitli oryantasyonlarda fotoğrafını çek, kalibre et.

*Seeker kısmından çıkan artık hangi sinyal olacaksa (Line of Sight veya Pixel Error) onu Servo Motorlar için kontrol sinyaline çevirecek bir CAS (Kontrol Aktüatör Sistemi) tasarla.
4 adet kanatçıklı bir füzede fin defleksiyonu nasıl çalışıyor anla.

*Raspberry serial pinlerinden STM Discovery Board'a CAS'ın hesapladığı PWM değerini gönder. STM'den HAL kütüphanesi ile PWM çıktısı ver.

*Fusion’dan kanatçık çiz veya hazır .sdf bul ve 3D’den bastır. Füze gövdesi için silindir bul veya .sdf bastır.

*Deneysel: 2 eksenli gimball yapıp üzerine lazer konulacak. Bu gimball X ve Y eksenlerinde tarama yapmaya başlayacak. Kamerada sadece lazerden yansıyan bir infrared kaynak varsa tarama devam eder.
Şayet ekrana ikinci bir infrared kaynak girerse gimball ona doğrultulur.

