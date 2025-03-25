Bitirme Çalışması İş Paketleri: 25 Mart 2025 6.Hafta

*İnfrared filtresiz kamera adaptive thresholding ile infrared ışığık kaynağı tespit edildi (güneş biraz daha filtrelenebilir) ve feature extraction ile iki kamera görseli üst üste hizalandı.

*Tensorflow Lite ve Yolo11n hazır veriseti ile detection denendi, işe yaramaz. Bu yüzden 100 fotoğraflık bir veriseti oluşturup labellandı ve Yolo11n modeli üzerinde denendi.
Arkaplan uyuşmazlığı ve veriseti yetersiliğinden ötürü istenilenden uzak performans veriyor. Sunum yapılacak ortamda yeni veriseti oluşturulacak.

*Hailo NPU'sunda modeli koşturmayı öğren. NPU nasıl çalışıyor anla.

*FPS arttırmak için detectionı belirli aralıklarla koştur. Kalan iterasyonlarda KCF tracker koşsun.

*Track edilen objeye güdümlenmek için dönüşümlere bakıldı. Binocular vision ile z hesaplayıp güdümlenmeye bak. Veya bu çok uzun sürecekse Pixel PID araştır.

*Seeker kısmından çıkan artık hangi sinyal olacaksa (Line of Sight veya Pixel Error) onu Servo Motorlar için kontrol sinyaline çevirecek bir CAS (Kontrol Aktüatör Sistemi) tasarla.
4 adet kanatçıklı bir füzede fin defleksiyonu nasıl çalışıyor anla.

*Raspberry serial pinlerinden STM Discovery Board'a CAS'ın hesapladığı PWM değerini gönder. STM'den HAL kütüphanesi ile PWM çıktısı ver.

*Fusion’dan kanatçık çiz veya hazır .sdf bul ve 3D’den bastır. Füze gövdesi için silindir bul veya .sdf bastır.

*Deneysel: 2 eksenli gimball yapıp üzerine lazer konulacak. Bu gimball X ve Y eksenlerinde tarama yapmaya başlayacak. Kamerada sadece lazerden yansıyan bir infrared kaynak varsa tarama devam eder.
Şayet ekrana ikinci bir infrared kaynak girerse gimball ona doğrultulur.

