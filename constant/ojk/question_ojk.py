def get_question(id: str) -> str:
    # Dictionary yang menyimpan pasangan ID dan query string
    questions = {
        "a": "Apa judul peraturan 7/33/PBI/2005?",  # Pencabutan atas Peraturan Bank Indonesia Nomor 5/17/PBI/2003 tentang Persyaratan dan Tata Cara Pelaksanaan Jaminan Pemerintah terhadap Kewajiban Pembayaran Bank Perkreditan Rakyat
        "b": "Kapan surat edaran No. 15/26/DPbS mulai berlaku?",  # 1 Agustus 2013
        "c": "Siapa nama dan jabatannya yang menandatangani surat dengan nomor 1/SEOJK.04/2013?",  # NURHAIDA, Kepala Eksekutif Pengawas Pasar Modal
        "d": "Saya ingin menyelenggarakan kegiatan pasar modal berikan saya nomor surat yang membahas mengenai hal ini!",  # Peraturan Pemerintah Nomor 12 Tahun 2004
        "e": "Berapa persen jaminan moneter pada tanggal 20 Agustus 1958?",  # 7,3%
        "f": "Surat edaran nomor berapa yang mengatur bank umum syariah dan unit usaha syariah?",  # 15/26/DPbS
        "g": "Apa kepanjangan dari PAPSI?",  # Pedoman Akuntansi Perbankan Syariah Indonesia
        "h": "Apa judul peraturan nomor 112/KMK.03/2001?",  # Keputusan Menteri Keuangan tentang Pemotongan Pajak Penghasil Pasal 21 atas Penghasilan berupa Uang Pesangon, Uang Tebusan Pensiun, dan Tunjangan Hari Tua atau Jaminan Hari Tua
        "i": "Saya ingin membuat sistem informasi lembaga jasa keuangan, berikan nomor regulasi dari peraturan yang membahas tentang manejemen risiko nya!",  # 4/POJK.05/2021
        "j": "Apa kepanjangan dari SWDKLLJ?",  # Sumbangan Wajib Dana Kecelakaan Lalu Lintas Jalan
        "k": "Berapa nilai SWDKLLJ dari sedan?",  # Rp. 140.000
        "l": "Apa latar belakang dari peraturan NOMOR 4/POJK.05/2021?",  # dalam bentuk list
        "m": "Apa itu LJKNB?",  # Lembaga Jasa Keuangan Non Bank
        "n": "Apakah KMK Nomor 462/KMK.04/1998 masih berlaku?",  # Tidak
        "o": "Apa itu Uang Pesangon?",  # Penghasilan yang dibayarkan oleh pemberi kerja kepada karyawan dengan nama dan dalam bentuk apapun sehubungan dengan berakhirnya masa kerja atau terjadi pemutusan hubungan kerja, termasuk uang penghargaan masa kerja dan uang ganti kerugian
        "p": "Apa itu CKPN?",  # Cadangan Kerugian Penurunan Nilai
        "q": "Kapan, dimana, dan oleh siapa surat nomor PER-06/BL/2012 ditetapkan?",  # Surat nomor PER-06/BL/2012 ditetapkan pada tanggal 22 November 2012 di Jakarta oleh Ketua Badan Pengawas Pasar Modal dan Lembaga Keuangan
        "r": "Apa kepanjangan PSAK?",  # Pernyataan Standar Akuntansi Keuangan
        "s": "Apa itu 'shahibul maal'?"  # Pemilik dana pihak ketiga
    }
    
    # Mengembalikan nilai query string berdasarkan ID, atau pesan error jika ID tidak ditemukan
    return questions.get(id, "ID tidak ditemukan")