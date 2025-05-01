Bitirme Çalışması İş Paketleri: 1 Mayıs 2025 11.Hafta

*Adaptif thresholding yerine HSV veya HSL gibi bir renk uzayı ile infrared renk maskesi oluştur.

*Sunum yapılacak ortamda çakmak için yeni veriseti oluşturuldu. Yolov11s modeli bu verisetine göre eğitildi. Model PyTorch'dan HEF'e (High Efficiency IMage Format) dönüştürüldü ve Hailo8L üzerinde çalıştırıldı. FPS, tracking iyi. mAP50 %83.2.

*Gstreamer pipeline'ında çalışan detection'dan bbox çekip sanal bir switch üzerinden infrared ayrımının yapıldığı kod ile sensör füzyonunu tamamla.

*Ekranın ortasından bbox'un merkezine bir çizgi çek ve yatay ve dikeydeki hataları düzeltmesi için kontrolcüye setpoint ver.

*Plastik uçak modelinden veriseti oluştur ve modele ekle.

*Rpi <-> STM UART haberleşmesi kuruldu. MG90S servolar kontrol ediliyor.


