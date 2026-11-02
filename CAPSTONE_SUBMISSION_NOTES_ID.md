## Topik use case

Pilih **Other** dan isi: **DataPilot AI: Agentic Data Analysis & Intelligence Platform**.

## Tools

Pilih **IBM Bob** sebagai tools pengembangan. IBM Bob adalah coding assistant yang digunakan untuk membantu menulis, meninjau, dan memperbaiki kode. IBM Bob bukan model AI yang dijalankan oleh aplikasi DataPilot AI.

## Ringkasan proyek

DataPilot AI adalah platform analisis data yang membantu pengguna mengeksplorasi dataset CSV dan Excel melalui antarmuka Streamlit. Sistem menyediakan data profiling, pemeriksaan kualitas data, analisis statistik, agregasi, deteksi anomali, dan visualisasi interaktif. Untuk demonstrasi offline, DataPilot AI menggunakan rule-based agent orchestration untuk mengidentifikasi intent pertanyaan, menyusun rencana analisis, memilih tools yang terdaftar, menjalankan perhitungan deterministik, dan menyusun penjelasan berdasarkan hasil aktual dari dataset. Pemrosesan data dilakukan menggunakan Python, Pandas, DuckDB, SciPy, Scikit-learn, dan Plotly. Build pengumpulan ini tidak menggunakan LLM atau API AI eksternal.

## Problem statement

Pengguna yang memiliki dataset sering harus melakukan beberapa tahapan secara manual, seperti memahami struktur data, memeriksa nilai kosong dan duplikasi, menghitung statistik, mengelompokkan data, mendeteksi anomali, serta membuat visualisasi. Proses tersebut memerlukan keterampilan teknis dan dapat memakan waktu. DataPilot AI dikembangkan untuk menyederhanakan eksplorasi dan analisis awal melalui alur agent berbasis aturan yang memilih analytical tools berdasarkan permintaan pengguna dan menghasilkan hasil yang dapat ditelusuri kembali ke dataset.

## Target users

Target pengguna meliputi mahasiswa, peneliti, data analyst pemula, dan pengguna organisasi yang perlu mengeksplorasi dataset tetapi belum terbiasa menggunakan Python atau SQL. Sistem membantu mereka memahami struktur dataset, memeriksa kualitas data, melakukan analisis statistik dan agregasi, menemukan anomali, serta melihat visualisasi.

## Arsitektur prompt tiga lapisan untuk IBM Bob

### Layer 1 — Role & Context

Bertindak sebagai Senior Python Engineer dan Software Architect yang membantu mengembangkan DataPilot AI, sebuah platform analisis data berbasis Python dan Streamlit. Pahami struktur repository yang sudah ada, tujuan pengguna, serta batasan perangkat dan lingkungan pengembangan. IBM Bob digunakan sebagai coding assistant selama pengembangan, bukan sebagai runtime AI di dalam aplikasi.

### Layer 2 — Task & Requirements

Periksa dan kembangkan fitur ingestion CSV/XLSX, data profiling, data quality checks, rule-based intent detection, agent planning, tool selection, eksekusi tools Pandas/DuckDB/SciPy/Scikit-learn, visualisasi Plotly, dan riwayat analisis. Setiap pertanyaan yang didukung harus dipetakan ke intent dan tools yang sesuai. Hasil numerik wajib dihitung dari dataset aktif.

### Layer 3 — Constraints & Output

Jangan menggunakan API token, LLM eksternal, atau model lokal. Jangan membuat angka hardcoded, menjalankan Python arbitrer, atau mengganti kolom yang tidak tersedia dengan kolom lain. Pertahankan struktur proyek, validasi parameter dan kolom, tangani pertanyaan ambigu dengan pesan yang jelas, jalankan pengujian, dan dokumentasikan keterbatasan secara jujur.

## Iterasi prompt

1. **Prompt awal:** meminta IBM Bob merancang struktur modular dan fitur utama DataPilot AI.
2. **Penyempurnaan kebutuhan:** menambahkan ingestion, profiling, tool registry, agent orchestration, analisis statistik, deteksi anomali, dan visualisasi.
3. **Penyesuaian keterbatasan API:** mengubah mode demonstrasi menjadi rule-based offline karena kredit API eksternal tidak tersedia.
4. **Perbaikan kualitas:** menambahkan validasi semantic column mapping, penanganan kolom yang tidak tersedia, regression tests, dan pemeriksaan hasil analisis agar tidak menjawab pertanyaan produk menggunakan kolom operator.

Gunakan urutan ini sebagai catatan proses hanya sejauh sesuai dengan prompt yang benar-benar pernah digunakan. Lampirkan screenshot prompt aktual dari IBM Bob sebagai bukti.

## Potensi pengembangan lanjutan

Pengembangan berikutnya dapat mencakup perluasan bahasa dan variasi pertanyaan, peningkatan semantic column mapping, evaluasi agent menggunakan kumpulan pertanyaan dengan ground truth, predictive analytics, pembuatan laporan otomatis, dukungan multi-dataset, dan integrasi database. Integrasi LLM dapat dipertimbangkan pada versi mendatang apabila akses layanan dan anggaran tersedia. Fitur-fitur tersebut merupakan rencana, bukan klaim bahwa semuanya sudah diimplementasikan.

## Bukti yang perlu diunggah

- Screenshot layar utama/workspace IBM Bob.
- Screenshot prompt yang benar-benar digunakan, dengan tiga lapisan prompt terlihat.
- Screenshot iterasi atau revisi prompt di IBM Bob.
- Screenshot pengujian DataPilot AI: pertanyaan, intent/tool yang dipilih, hasil aktual, dan visualisasi.
- Link repository GitHub atau Google Drive yang dapat diakses reviewer.

## Catatan penting

Versi pengumpulan ini adalah **Rule-Based Agentic Data Analysis Platform**, bukan LLM-powered Agent. IBM Bob dipakai sebagai alat bantu pengembangan. Jangan menyebut LLM sebagai komponen runtime aplikasi.
