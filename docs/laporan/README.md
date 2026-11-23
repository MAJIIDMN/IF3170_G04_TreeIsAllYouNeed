# Laporan Tugas Besar 2 IF3170

Source LaTeX laporan. PDF final disalin ke `docs/` sesuai spesifikasi.

## Compile

```bash
cd docs/laporan
latexmk -pdf -outdir=build main.tex
cp build/main.pdf ../Laporan_IF3170_G04_TreeIsAllYouNeed.pdf   # PDF final untuk dikumpulkan
```

Bisa juga di Overleaf: upload seluruh isi `docs/laporan/` beserta folder `output/` dan `src/` (dipakai
untuk gambar dan potongan kode), atau sesuaikan `\graphicspath` di `preamble.tex`.

## Struktur

```
main.tex                 # kerangka: urutan bab
preamble.tex             # paket, header/footer, gaya kode, makro \kode dan \todo
cover.tex                # halaman sampul (isi nama dan NIM di sini)
dafpus.bib               # daftar pustaka
sections/
  01-deskripsi-persoalan/  # latar belakang, dataset, EDA, metrik
  02-pembahasan/
    01-kerangka-umum.tex     # struktur kode, preprocessing bersama, protokol validasi
    02-knn/ 03-dtl/ 04-logistic-regression/ 05-svm/ 06-naive-bayes/
      00-main.tex                  # \section algoritma
      01-implementasi.tex          # teori, kelas & fungsi, source code, save/load
      02-cleaning-preprocessing.tex# missing, duplikat, encoding, scaling, imbalance, FS, PCA
      03-modeling-validasi.tex     # skema validasi, tuning, model final
      04-hasil-perbandingan.tex    # scratch vs sklearn, confusion matrix, analisis
      05-bonus.tex                 # bonus khusus algoritma + optimasi
    07-perbandingan-antarmodel.tex
    08-cost-sensitive.tex    # bonus
    09-kaggle.tex
    10-rekap-bonus.tex
  03-kesimpulan-saran.tex
  04-pembagian-tugas.tex
  05-lampiran/             # tautan penting, form penggunaan Gen-AI
```

## Konvensi

- Gambar yang belum ada ditandai `\placeholdergambar{...}`. Ganti dengan `\includegraphics{nama}`;
  gambar dari `output/` bisa langsung dipanggil berdasarkan nama (lihat `\graphicspath`). Gambar khusus laporan
  ditaruh di `public/figures/`.
- Potongan kode disisipkan langsung dari `src/` dengan
  `\kode{../../src/algoritma/<file>.py}{baris_awal}{baris_akhir}{caption}{label}`. Periksa nomor baris
  setiap kali kode berubah.
- Bagian yang belum ditulis ditandai `\todo{...}` (teks merah). Sebelum final, ubah `\toggletodotrue`
  menjadi `\toggletodofalse` di `preamble.tex`, lalu pastikan tidak ada TODO yang tersisa.
