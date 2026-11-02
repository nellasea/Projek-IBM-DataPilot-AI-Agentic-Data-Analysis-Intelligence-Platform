# Mulai Menjalankan DataPilot AI (Windows)

## 1. Persiapan

Gunakan Python 3.11 atau lebih baru. Buka PowerShell di folder proyek.

```powershell
cd C:\IBM\DataPilot-Ai
```

## 2. Buat virtual environment (jika belum ada)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Jika PowerShell memblokir aktivasi script, jalankan untuk sesi terminal saat ini:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Install dependency

```powershell
pip install -e ".[dev]"
```

Instalasi memerlukan koneksi internet untuk mengunduh dependency Python. Tidak ada model AI atau API token yang perlu diunduh/dimasukkan.

## 4. Jalankan aplikasi

```powershell
streamlit run ui/app.py
```

## 5. Jalankan pengujian

```powershell
python -m pytest -q
```

Catat hasil yang benar-benar muncul di terminal. Jangan mengklaim semua tes lulus jika ada kegagalan.

## Mode aplikasi

Build ini berjalan dalam **Rule-Based Agent Mode**. Agent mengidentifikasi intent, memilih registered tools, dan menghitung hasil menggunakan dataset aktual. Build ini tidak memanggil OpenAI, Gemini, atau layanan LLM lainnya.

## Keamanan

Jangan membuat atau membagikan `.env` berisi API key. Jangan unggah folder `.venv`, `.git`, cache, atau database runtime ke repository publik.
