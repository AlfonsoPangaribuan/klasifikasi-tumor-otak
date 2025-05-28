## README - Analisis Tumor Otak Menggunakan CLIP dan SVM

### Deskripsi Singkat Proyek dan Latar Belakang Masalah

Proyek ini bertujuan untuk melakukan analisis komprehensif terhadap dataset citra MRI tumor otak dengan membandingkan performa dua pendekatan machine learning: Support Vector Machine (SVM) sebagai metode machine learning tradisional dan Contrastive Language-Image Pre-training (CLIP) sebagai representasi modern berbasis deep learning. Latar belakang utama dari penelitian ini adalah meningkatnya insidensi tumor otak secara global dan nasional, yang menyerang berbagai kelompok usia. Deteksi dini dan klasifikasi jenis tumor otak yang akurat sangat krusial untuk menentukan strategi pengobatan yang efektif dan meningkatkan peluang kesembuhan pasien. Magnetic Resonance Imaging (MRI) menjadi modalitas pencitraan utama karena kemampuannya menghasilkan citra resolusi tinggi dari struktur otak tanpa radiasi pengion. Namun, interpretasi manual citra MRI oleh ahli radiologi memakan waktu, bersifat subjektif, dan berpotensi memiliki akurasi rendah. Oleh karena itu, pengembangan sistem diagnosis otomatis menggunakan machine learning menjadi sangat penting untuk membantu tenaga medis dalam mengidentifikasi dan mengklasifikasikan tumor otak secara lebih cepat, objektif, dan efisien. Proyek ini secara spesifik membandingkan SVM, yang telah terbukti efektif dalam beberapa studi sebelumnya, dengan CLIP, sebuah model pre-trained multimodal yang diharapkan mampu memberikan pemahaman kontekstual yang lebih baik terhadap citra medis.

### Penjelasan Dataset

Dataset yang digunakan dalam proyek ini adalah kumpulan citra Magnetic Resonance Imaging (MRI) otak yang telah dikategorikan ke dalam empat kelas: glioma, meningioma, pituitary, dan normal (tidak ada tumor). Dataset ini bersumber dari [sebutkan sumber dataset jika diketahui dari notebook/laporan, misal Kaggle Dataset 'Brain Tumor MRI Dataset']. Karakteristik utama dataset mencakup variasi dalam intensitas piksel dan orientasi citra. Sebelum digunakan untuk melatih model, dataset melalui beberapa tahap prapemrosesan. Tahapan ini meliputi Exploratory Data Analysis (EDA) untuk memahami distribusi kelas dan karakteristik citra, augmentasi data untuk meningkatkan jumlah dan variasi data latih (misalnya rotasi, flip, zoom), normalisasi nilai piksel untuk menyeragamkan rentang intensitas, serta label encoding untuk mengubah label kelas kategorikal menjadi format numerik yang dapat diproses oleh model. Data kemudian dibagi menjadi set pelatihan (training set) dan set pengujian (testing set) untuk melatih dan mengevaluasi model secara objektif.

### Algoritma yang Digunakan

Proyek ini menggunakan algoritma **klasifikasi** untuk mengkategorikan citra MRI ke dalam salah satu dari empat kelas tumor otak yang telah ditentukan. Dua algoritma utama yang diimplementasikan dan dibandingkan adalah:

1.  **Support Vector Machine (SVM):** Merupakan algoritma machine learning supervised klasik yang bekerja dengan mencari hyperplane optimal untuk memisahkan data ke dalam kelas-kelas yang berbeda. Dalam konteks ini, SVM digunakan untuk mengklasifikasikan citra berdasarkan fitur-fitur yang diekstraksi (meskipun detail ekstraksi fitur spesifik mungkin perlu dilihat di kode, seringkali menggunakan fitur tekstur atau bentuk, atau langsung dari piksel setelah prapemrosesan). Kernel yang berbeda (seperti linear, RBF) dapat digunakan untuk menangani data yang tidak dapat dipisahkan secara linear.
2.  **Contrastive Language-Image Pre-training (CLIP):** Merupakan model deep learning modern yang dilatih pada skala besar menggunakan pasangan gambar dan teks. CLIP belajar untuk menghubungkan representasi visual dengan representasi tekstual yang sesuai. Dalam proyek ini, model CLIP (kemungkinan di-fine-tune pada dataset tumor otak) digunakan untuk mengekstraksi fitur gambar yang kaya secara semantik dan kemudian fitur tersebut digunakan untuk tugas klasifikasi. Pendekatan ini memanfaatkan kemampuan transfer learning dari model pre-trained besar untuk memahami konteks visual dalam citra medis.

### Panduan Menjalankan Kode

Kode analisis ini diimplementasikan dalam bentuk Notebook (misalnya Jupyter atau Kaggle Notebook) menggunakan bahasa pemrograman Python dan library machine learning umum. Berikut adalah langkah-langkah umum untuk menjalankan kode:

1.  **Setup Lingkungan:** Pastikan environment Python Anda memiliki semua library yang diperlukan. Library utama yang mungkin digunakan termasuk `tensorflow` atau `pytorch` (untuk CLIP), `scikit-learn` (untuk SVM dan metrik evaluasi), `pandas` (untuk manipulasi data), `numpy` (untuk operasi numerik), `matplotlib` dan `seaborn` (untuk visualisasi), serta `opencv-python` atau `PIL` (untuk pemrosesan gambar). Anda dapat menginstal dependensi menggunakan pip:
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn tensorflow # atau torch torchvision torchaudio
    ```
2.  **Unduh Dataset:** Dapatkan dataset citra MRI tumor otak dari sumber yang relevan. Pastikan struktur direktori dataset sesuai dengan yang diharapkan oleh kode (biasanya diatur dalam sel kode awal).
3.  **Jalankan Notebook:** Buka file Notebook (.ipynb) menggunakan Jupyter Notebook, JupyterLab, Google Colab, atau platform Kaggle.
4.  **Eksekusi Sel Kode:** Jalankan sel-sel kode secara berurutan. Urutan eksekusi umumnya meliputi:
    *   Impor library yang dibutuhkan.
    *   Memuat dan melakukan pra-pemrosesan data (termasuk augmentasi, normalisasi, splitting).
    *   Membangun dan melatih model SVM.
    *   Membangun (memuat pre-trained) dan melatih/fine-tuning model CLIP.
    *   Melakukan evaluasi pada data uji untuk kedua model.
    *   Menampilkan hasil evaluasi dan visualisasi perbandingan.

### Contoh Hasil Output dan Visualisasi

Notebook ini menghasilkan berbagai output dan visualisasi untuk menganalisis performa model. Beberapa contohnya meliputi:

*   **Visualisasi Data:** Grafik distribusi jumlah gambar per kelas, contoh gambar dari setiap kelas, dan distribusi intensitas piksel.
*   **Visualisasi Augmentasi:** Contoh citra sebelum dan sesudah proses augmentasi data.
*   **Metrik Evaluasi:** Tabel atau output teks yang menampilkan metrik kinerja seperti akurasi (accuracy), presisi (precision), recall, dan F1-score untuk setiap kelas dan secara keseluruhan (rata-rata makro/mikro) pada data uji, baik untuk model SVM maupun CLIP.
*   **Confusion Matrix:** Matriks konfusi divisualisasikan sebagai heatmap untuk kedua model. Ini membantu memahami jenis kesalahan klasifikasi yang dibuat oleh masing-masing model (misalnya, kelas mana yang sering tertukar).
*   **Kurva Pelatihan (untuk CLIP):** Grafik yang menunjukkan penurunan nilai loss dan peningkatan akurasi pada data pelatihan dan validasi selama proses training model CLIP.
*   **Grafik Perbandingan:** Diagram batang atau grafik garis yang membandingkan metrik evaluasi (akurasi, presisi, recall, F1-score) antara model SVM dan CLIP, baik secara keseluruhan maupun per kelas.

### Kesimpulan

Berdasarkan analisis dan perbandingan yang dilakukan dalam proyek ini, dapat ditarik kesimpulan mengenai efektivitas relatif antara metode SVM dan CLIP untuk tugas klasifikasi tumor otak dari citra MRI. Hasil evaluasi menunjukkan [**Masukkan ringkasan hasil utama di sini, contoh: bahwa model CLIP menunjukkan performa yang lebih unggul dibandingkan SVM dalam hal akurasi keseluruhan, presisi, recall, dan F1-score, meskipun SVM mungkin lebih efisien dalam hal waktu pelatihan.**]. Model CLIP, dengan kemampuannya memahami konteks visual melalui pre-training skala besar, tampaknya lebih baik dalam menangkap fitur-fitur relevan dari citra MRI dibandingkan pendekatan SVM tradisional dalam studi ini. Meskipun demikian, [**Sebutkan keterbatasan atau area pengembangan selanjutnya, contoh: SVM tetap menjadi baseline yang kuat dan mungkin lebih cocok untuk dataset yang lebih kecil atau sumber daya komputasi terbatas. Pengembangan selanjutnya dapat mencakup eksplorasi arsitektur CLIP yang berbeda, teknik augmentasi yang lebih canggih, atau kombinasi fitur dari kedua model.**]. Secara keseluruhan, penelitian ini menyoroti potensi besar model pre-trained seperti CLIP dalam meningkatkan akurasi diagnosis medis berbasis gambar, namun juga menegaskan pentingnya evaluasi komprehensif terhadap berbagai pendekatan sesuai dengan konteks masalah dan batasan yang ada.
