# Apakah email perlu disesuaikan dengan riwayat beli?

Uji uplift pada dataset Hillstrom (MineThatData E-Mail Analytics Challenge, 2008). Target: kunjungan ke toko online (`visit`).

## Ringkasan

Kami merekomendasikan mengirim email produk pria (Mens) ke semua pelanggan. Kebijakan ini menaikkan visit rate 7,7 poin persentase dibanding tidak mengirim email, dan terbukti lebih baik daripada mengirim email produk wanita (Womens) ke semua. Hasil ini bertahan di hold-out, yaitu data yang tidak disentuh selama penjelajahan.

Model uplift (S-learner dan T-learner LightGBM) tidak terbukti lebih baik daripada "kirim Mens ke semua", baik dengan fitur dasar maupun dengan tambahan rekayasa fitur. Artinya: dengan fitur yang tersedia, personalisasi belum memberi nilai tambah yang bisa dibuktikan. Ini bukan bukti bahwa personalisasi pasti tidak berguna (lihat [Batasan](#batasan)).

Hasil hanya berlaku untuk kunjungan ke toko online (`visit`). Dampak ke konversi, spend, unsubscribe, dan biaya kirim tidak dianalisis.

## Isi repositori

| Berkas | Isi |
|---|---|
| `notebooks/01_eda.ipynb` | Eksplorasi data: pemeriksaan randomisasi, efek email, heterogenitas per fitur |
| `notebooks/02_evaluation_framework_baseline_policy.ipynb` | Pembagian data (hold-out dikunci, 5 fold), metrik, dan patokan tanpa model |
| `notebooks/03_modeling.ipynb` | Model S/T-learner, rekayasa fitur, tuning satu hyperparameter, hold-out |
| `data/split_hillstrom.csv` | Label `split` (dev/holdout) dan `fold` (0..4; -1 = hold-out) per baris, supaya pembagian tidak berubah antar jalan |
| `data/Kevin_Hillstrom_Dataset.csv` | Data mentah (64.000 baris). Unduh dari sumber aslinya dan taruh di folder `data/` |

Urutan menjalankan: 01, 02, lalu 03. Notebook memakai `Path.cwd().parent` sebagai root, jadi jalankan dari dalam folder `notebooks/`. Pustaka: numpy, pandas, scikit-learn, lightgbm, matplotlib, seaborn, scipy.

## Pertanyaan dan setup

Dataset berisi 64.000 pelanggan dalam tiga kelompok (disebut *arm*): email produk pria, email produk wanita, dan tanpa email (kontrol). Ada dua setup dengan pertanyaan berbeda:

- **Setup B, "kirim apa ke siapa".** Untuk tiap pelanggan, pilih Mens, Womens, atau tanpa email. Dinilai dengan *nilai kebijakan*: rata-rata visit yang akan didapat kalau kebijakan itu dijalankan ke semua pelanggan.
- **Setup A, "siapa yang diprioritaskan kalau budget terbatas".** Pelanggan diurutkan berdasarkan skor uplift; treated = Mens atau Womens (digabung), control = tanpa email. Dinilai dengan *kurva uplift*. Setup ini tidak membedakan Mens dari Womens, jadi tidak menguji keputusan "kirim apa".

Fitur model ada tujuh: `recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`. Catatan: `mens` dan `womens` adalah **riwayat beli** (pernah membeli produk pria/wanita), bukan jenis kelamin dan bukan perlakuan. `conversion`, `spend`, dan `history_segment` tidak dipakai sebagai fitur: dua yang pertama baru diketahui setelah email dikirim (dan conversion hanya mungkin terjadi kalau sudah visit, sehingga akan membocorkan target), yang terakhir hanya `history` yang dikelompokkan ulang.

## Hasil

### Setup B: kirim apa ke siapa

Selisih dihitung sebagai pembanding dikurangi P1 (negatif = pembanding lebih rendah). Interval 95% dari bootstrap berpasangan.

| Kebijakan | Dev | Hold-out | Selisih vs P1, dev | Selisih vs P1, hold-out |
|---|---:|---:|---:|---:|
| P0 Tidak kirim | 0,1061 | 0,1062 | -0,0766 [-0,0843; -0,0690] | -0,0765 [-0,0878; -0,0636] |
| **P1 Mens ke semua** | 0,1828 | 0,1827 | patokan | patokan |
| P2 Womens ke semua | 0,1514 | 0,1513 | -0,0313 [-0,0405; -0,0230] | -0,0314 [-0,0454; -0,0186] |
| P3 Tabel mens × womens | 0,1825 | tidak dinilai | -0,0003 [-0,0059; +0,0043] | - |
| S_B (S-learner, FE-3) | - | 0,1824 | - | -0,0003 [-0,0029; +0,0022] |

P1 terbukti lebih baik daripada P0 dan P2 di dev maupun hold-out. P3 hanya dinilai di dev dan tidak terbukti berbeda dari P1; P1 dipilih karena lebih sederhana, bukan karena terbukti unggul.

**Model vs P1 di dev (5 seed per baris).**

| Versi fitur | S_B: verdict | S_B: kisaran titik | T_B: verdict | T_B: kisaran titik |
|---|---|---|---|---|
| 7 fitur dasar | belum terbukti 5/5 | -0,0011 s/d +0,0013 | belum terbukti 4/5, kalah 1/5 | -0,0047 s/d -0,0025 |
| FE-1 history/recency | belum terbukti 5/5 | -0,0003 s/d +0,0006 | kalah 3/5 | -0,0055 s/d -0,0039 |
| FE-2 zip × channel | belum terbukti 5/5 | -0,0007 s/d +0,0012 | kalah 1/5 | -0,0048 s/d -0,0021 |
| FE-3 mens + womens | belum terbukti 5/5 | -0,0002 s/d +0,0017 | kalah 2/5 | -0,0049 s/d -0,0031 |

Tidak ada model yang menang atas P1 di satu pun seed. S_B berada di sekitar P1: titiknya berganti tanda antar seed dan besarnya setara pergeseran akibat seed (sekitar 0,0024). T_B condong lebih buruk: titiknya negatif di semua seed. S_B tampak unggul atas T_B di banyak seed, tetapi itu terutama karena T_B lemah, bukan bukti S_B unggul.

**Di hold-out**, S_B vs P1 = -0,0003 [-0,0029; +0,0022], belum terbukti beda. Penyebabnya terlihat dari keputusannya: hanya 3,4% pelanggan yang dikirimi email Womens, jadi S_B praktis mengulang P1. Batas atas +0,0022 menyingkirkan keuntungan besar dari model, tetapi keuntungan yang sangat kecil tidak bisa dikesampingkan.

**Tuning satu hyperparameter** (`min_child_samples` = 1000 / 200 / 50, FE-3, 5 seed per nilai, hanya di dev, tanpa memilih pemenang): tidak ada jalan yang plus-plus terhadap P1. S_B vs P1 tetap di kisaran noise pelatihan, dan T_B tetap condong lebih buruk. Hold-out memakai nilai 200 sesuai protokol yang dikunci sebelumnya.

### Setup A: urutan skor kalau ada batas budget

*Area kurva* adalah luas antara kurva uplift kumulatif dan garis acak (k × ATE): makin besar, makin baik skor itu mengurutkan pelanggan. Area meringkas semua ukuran budget sekaligus, jadi tidak menunjukkan k terbaik.

| Skor | Area, dev | Area, hold-out |
|---|---:|---:|
| Acak (kontrol negatif) | -0,0007 [-0,0025; +0,0010] | 0,0003 |
| Skor 3 sel mens × womens | +0,0039 [+0,0019; +0,0056] | 0,0069 |
| S_A | +0,0043 | 0,0031 |
| T_A | +0,0030 | tidak dinilai |

- **Dev:** tidak ada model yang terbukti mengurutkan lebih baik daripada tabel 3 sel di seed mana pun (belum terbukti 5/5). S_A kira-kira setara tabel, T_A sedikit di bawahnya.
- **Hold-out:** tabel lebih baik daripada S_A. S_A dikurangi 3 sel = -0,0037 [-0,0060; -0,0015], negatif di kedua batas.
- Sebagian keunggulan tabel di hold-out mungkin berasal dari sel (1,1) (pernah membeli kedua kategori) yang kebetulan lebih responsif: uplift 0,1546 di hold-out vs 0,0809 di tabel dev. Dugaan penyebabnya belum diuji.

## Metode

**Model.** LightGBM. *S-learner*: satu model dengan arm sebagai fitur; uplift = prediksi dengan email dikurangi tanpa email. *T-learner*: satu model per arm; uplift = prediksi model arm email dikurangi model kontrol. Di setup B, kebijakan model = email dengan uplift prediksi tertinggi (Mens atau Womens) untuk tiap pelanggan.

**Patokan.**

| Kode | Kebijakan |
|---|---|
| P0 | Tidak kirim |
| P1 | Mens ke semua (**patokan yang harus dikalahkan**) |
| P2 | Womens ke semua |
| P3 | Tabel mens × womens: pilih arm terbaik per kelompok riwayat beli, dihitung cross-fit. Hanya 3 kelompok yang muncul di data (mens=0, womens=0 tidak ada) |
| Skor 3 sel | Skor uplift dari tabel yang sama; patokan untuk setup A |

**Rekayasa fitur** (satu perubahan per eksperimen, dari 7 fitur dasar; semuanya eksplorasi karena EDA tidak menunjukkan heterogenitas di fitur-fitur ini selain `mens`/`womens`):

- FE-1: `history / recency`
- FE-2: `zip_code × channel` (9 level)
- FE-3: `mens + womens` (nilainya 1 atau 2; nilai 2 identik dengan sel (1,1))

### Kenapa penilaiannya tidak bias

**Randomisasi.** Email dibagikan secara acak (menurut desain dataset) dengan peluang sekitar 1/3 per arm, dan pemeriksaan keseimbangan fitur sebelum perlakuan di EDA tidak menunjukkan penyimpangan (SMD < 0,01, chi-square tidak signifikan). Karena itu selisih antar arm dapat dibaca sebagai efek email. Nilai kebijakan dihitung dengan estimator IPW: hanya baris yang arm-nya kebetulan sama dengan pilihan kebijakan yang dihitung, dibobot 1/p (sekitar 3) supaya mewakili seluruh pelanggan. Fungsi penilai tidak memakai fitur apa pun, dan sudah diuji di data sintetis yang kebenarannya diketahui.

**Pembagian data.** Hold-out 30% (19.200 baris) dipisah dan dibuka satu kali. Dev (44.800 baris) dibagi 5 fold dengan stratifikasi arm × visit. Semua skor di dev adalah out-of-fold, jadi skor satu baris tidak pernah dibuat dengan melihat visit baris itu. Label pembagian disimpan di `data/split_hillstrom.csv`; hold-out dipisahkan lewat label itu, bukan lewat berkas data terpisah.

**Membaca noise.** Interval dibuat dengan bootstrap (1.000 kali untuk nilai kebijakan; 300 kali untuk area kurva tunggal; 500 kali untuk selisih area berpasangan). Satu resample dipakai untuk semua kebijakan, jadi interval selisih bersifat berpasangan. Keputusan hanya diambil dari interval selisih, bukan dari membandingkan dua interval dengan mata:

| Batas bawah / atas | Arti |
|---|---|
| plus-plus (+/+) | menang |
| minus-plus (-/+) | belum terbukti (interval memuat nol) |
| minus-minus (-/-) | kalah |

"Belum terbukti" berarti data belum cukup untuk membedakan, bukan bukti bahwa keduanya sama. Kata "terbukti" hanya dipakai kalau hasilnya muncul di mayoritas dari 5 seed (42, 5, 11, 2026, 26). Sebagai kontrol negatif, skor acak harus memberi area kurva mendekati nol, dan itu terpenuhi di dev dan hold-out.

**Batas penilaian.** Interval hanya mengukur noise penilaian (pengambilan sampel baris), bukan noise pelatihan model. Pergeseran antar seed sekitar 0,0024.

## Batasan

**Apa yang diukur.** Hasil hanya berlaku untuk kunjungan ke toko online (`visit`). Konversi dan spend sengaja tidak dianalisis: kejadian konversi hanya sekitar 120 sampai 270 per arm, dan target serta protokol dikunci sebelum hold-out dibuka. Biaya kirim, unsubscribe, dan efek jangka panjang tidak ada di data. Kenaikan visit belum tentu berarti kenaikan pembelian.

**Dari mana datanya.** Satu kampanye lama, pada pelanggan yang membeli dalam 12 bulan terakhir. Hasilnya belum tentu berlaku untuk pelanggan baru atau kampanye lain. Isi email (produk, penawaran, desain) tidak diketahui, jadi "kirim Mens" belum bisa dibedakan dari "kirim email dengan isi seperti ini". Pembeli produk wanita mungkin membeli untuk orang lain, jadi riwayat beli tidak sama dengan minat pelanggan itu sendiri.

**Arti "model tidak terbukti lebih baik".** Data belum cukup untuk membedakan, bukan bukti bahwa model dan P1 sama. Di hold-out, batas atas selisih sekitar +0,2 poin: keuntungan besar bisa dikesampingkan, keuntungan sangat kecil tidak.

**Seberapa luas pengujiannya.** Hold-out hanya menilai satu model dengan satu seed (S-learner); T-learner tidak dinilai di hold-out, jadi kesimpulan tentang T hanya dari dev. FE-3 dipilih demi konsistensi, bukan karena terbukti terbaik. Rekayasa fitur diuji sebanyak tiga eksperimen, tuning hanya satu hyperparameter, dan riwayat pelanggan yang lebih kaya tidak teruji. Tuning hanya dilakukan dan dinilai di setup B; setup A memakai hyperparameter asli (`min_child_samples` = 200), jadi kesimpulan setup A berlaku untuk pengaturan itu. Kesimpulan bahwa lima fitur tambahan tidak membantu berlaku untuk pengaturan yang kami coba, bukan untuk semua model yang mungkin.

**EDA sebelum pembagian data.** EDA (`01_eda.ipynb`) dijalankan pada data penuh sebelum hold-out dipisah, sehingga hold-out ikut terlihat dalam ringkasan kasar (randomisasi, efek rata-rata per arm, heterogenitas per fitur). Tidak ada model, fitur, atau seed yang dipilih dari hold-out. Hipotesis bahwa heterogenitas hanya ada di `mens`/`womens` terbentuk dari EDA ini, jadi hold-out tidak sepenuhnya murni untuk hipotesis tersebut.

**Batasan dari EDA.** Heterogenitas baru dilihat satu fitur per satu fitur; interaksi antar fitur belum diperiksa. Angka uplift per bin di EDA adalah perkiraan visual dari plot. Rekomendasi pasca-visit (kampanye lanjutan, hold-out acak pasca-deployment) adalah rekomendasi, bukan hasil data ini, dan butuh A/B test sendiri.

**Rekomendasi praktis.** Uji kecil dengan tiga kelompok (Mens, Womens, tanpa email), ukur spend dan unsubscribe, dan sisakan kelompok pembanding.

## Prosedur

**Aturan yang dikunci sebelum menjalankan.** Hold-out 30% (19.200 baris) dipisah di awal dan dibuka satu kali. Semua skor di dev adalah out-of-fold (5 fold). Hyperparameter awal dikunci dan tidak disetel setelah melihat skor. Prediksi ditulis sebelum setiap eksperimen dijalankan dan tidak diedit sesudahnya. Keputusan diambil dari interval selisih berpasangan, dan "terbukti" hanya dipakai kalau hasilnya muncul di mayoritas dari 5 seed.

**Protokol hold-out.** Ditetapkan sebelum hold-out dibuka dan hanya berdasarkan hasil dev: S-learner, FE-3, `min_child_samples` 200, seed 42, dilatih di seluruh dev. S-learner dipilih karena di dev tidak pernah terbukti lebih buruk dari P1 di seed mana pun, sedangkan T-learner terbukti lebih buruk di sebagian seed. Ini bukan klaim bahwa S terbukti lebih baik daripada T. FE-3 dipilih demi konsistensi: di dev tidak ada versi fitur yang unggul atas yang lain. Tidak ada model, rekayasa fitur, atau seed yang diubah setelah melihat hold-out.

**Urutan pembukaan hold-out.** Hold-out dibuka untuk setup B (kebijakan) lebih dulu, lalu setup A (kurva uplift). Keduanya pertanyaan yang berbeda, tetapi urutan ini dicatat supaya pembaca bisa menilai sendiri.

**Satu perubahan sebelum ada model yang dinilai.** Pemeriksaan "tepat 6.400 baris per kelompok" di hold-out diperlonggar, karena jumlah sebenarnya 6.392, 6.392, dan 6.416 (total tetap 19.200). Perubahan ini dibuat sebelum model apa pun dinilai di hold-out.

**Prediksi yang meleset (dicatat, tidak diedit).**

- "Model mengalahkan P1 dengan plus-plus": tidak terjadi di seed mana pun.
- "T-learner lebih stabil daripada S": rentang antar seed keduanya kira-kira sama (sekitar 0,0024 dan 0,0022).
- FE-1 diperkirakan menaikkan S_B sekitar +0,001 sampai +0,003 dan plus-plus di 3 dari 5 seed untuk setup A: tidak terjadi.
- FE-2 diperkirakan di bawah FE-1: hanya setengah benar.
- Hold-out S_B diperkirakan +0,0001: hasilnya -0,0003 (tanda di titik sekecil ini tidak bermakna).
- Hold-out setup A diperkirakan ketiga area negatif: ketiganya positif.
- Prediksi dari asisten (Claude): interval hold-out diperkirakan lebih lebar daripada dev, ternyata sedikit lebih sempit; dan T_B diperkirakan kalah sekitar 3 dari 5 seed di FE-2, ternyata hanya 1 dari 5.

## Proses kerja dan bantuan AI

Proyek ini dikerjakan dengan bantuan AI (Claude):

> *Secara umum konsep, keputusan, dan alur kerja diputuskan sendiri. Adapun codingan dan penulisan laporan banyak dibantu Claude*
