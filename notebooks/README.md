# Colab defterleri (Faz V7)

Bu klasör, uzun GPU eğitimlerini Google Colab'da koşturmak için gereken her şeyi tutar. Kodun asıl hali depodadır;
Colab'a yalnızca bir zip kopyası gider. Sonuçlar Drive üzerinden geri gelir.

| Dosya | Ne işe yarar |
|---|---|
| `paketle.py` | Kodu ve veri önbelleğini `notebooks/paket/` altına iki zip olarak koyar |
| `v7_rapor_egitimi_indy.ipynb` | Indy rapor modelleri (seed 0–4): ön-eğitim + ince ayar + Day-0 |
| `v7_rapor_egitimi_loco.ipynb` | Loco için aynısı |
| `paket/` | Üretilen zip'ler (git'e girmez) |

İki defter aynı koddur, yalnız `PRIMAT` ayarı farklıdır. İkisi iki ayrı Colab sekmesinde aynı anda koşabilir.

## Neden Colab?

- **Paralellik:** Bilgisayarındaki GPU ayar modellerini (`runs/chain_v7_tune.sh`) eğitirken rapor modelleri Colab'da eğitilir.
- **Kararlılık:** Windows'taki CUDA Graph çöküşü (KR-043) Linux'ta beklenmiyor. Yine de defter her seed'i ayrı süreçte ve
  `--resume` ile koşar.
- **Yöntem aynı:** Config, seed'ler ve kod aynı. Colab'da git olmadığı için kod sürümü `KOD_SURUMU.json` ile her koşu
  klasörüne yazılır: commit, değişen dosyalar ve her dosyanın SHA-256'sı (KV8).

## Adım adım

### 1. Paketi oluştur (bilgisayarında, ~1 dk)

Depo kökünde, PowerShell'de:

```bash
uv run python notebooks/paketle.py
```

Çıktı `notebooks/paket/` altına iki dosya yazar:

- `spike_kod.zip`: ~0,2 MB
- `spike_veri.zip`: ~34 MB, 6 test oturumunun önbelleği

Kod değişirse yalnız kodu yeniden paketle: `uv run python notebooks/paketle.py --yalniz-kod`.

### 2. Drive'da klasörü aç

drive.google.com → Drive'ım → **Yeni → Klasör** → adı tam olarak: `Spike_colab`

### 3. Zip'leri yükle

`notebooks/paket/spike_kod.zip` ve `notebooks/paket/spike_veri.zip` dosyalarını `Drive'ım/Spike_colab/` içine sürükle.
**Zip'leri açma**, defter kendisi açar.

```
Drive'ım/
└── Spike_colab/
    ├── spike_kod.zip
    ├── spike_veri.zip
    └── runs/            ← defter oluşturur; eğitim çıktıları buraya
```

### 4. Defterleri Colab'da aç

colab.research.google.com → **Dosya → Not defteri yükle** → `notebooks/v7_rapor_egitimi_indy.ipynb`.
Loco için yeni bir sekmede aynı adımla `v7_rapor_egitimi_loco.ipynb` dosyasını yükle.

Her defterde: **Çalışma zamanı → Çalışma zamanı türünü değiştir → GPU → L4** → Kaydet.
L4 bu model için yeterli ve compute unit'i A100'den çok daha az harcar.

### 5. Önce duman denemesi (önerilir, ~5 dk)

1. 1. hücrede `DUMAN = True` yap.
2. **Çalışma zamanı → Tümünü çalıştır**. Drive izni sorulursa onayla.
3. Sonunda `BİTTİ` ve bir R² tablosu görmelisin. Duman çıktısı Drive'a yazılmaz.
4. Sonra `DUMAN = False` yap.

### 6. Asıl eğitim

`DUMAN = False` ile **Tümünü çalıştır**. 5. hücre seed seed ilerler ve her adımda şöyle bir satır yazar:

```
indy seed 0 ptft deneme 1: çıkış 0, 38.2 dk → /content/drive/MyDrive/Spike_colab/runs/2026..._v2_pretrain_v7_indy_seeds0
```

- **Sekmeyi açık tut.** Colab Pro'da tarayıcı kapanınca oturum bir süre sonra düşer.
- **Bağlantı koparsa:** defteri yeniden **Tümünü çalıştır**. Biten adımlar `[atla]` ile geçilir, yarım kalan adım
  `--resume` ile kaldığı yerden devam eder. Bir şey silmen gerekmez.
- **Süre:** primat başına birkaç saat. İki defter paralel koşarsa toplam süre en uzun primatınki kadar olur.

### 7. Sonuçları geri getir

Eğitim bitince (`BİTTİ: indy [0, 1, 2, 3, 4]`):

1. Drive'da `Spike_colab/runs` klasörüne sağ tıkla → **İndir** (Drive zip'ler).
2. Zip'i bilgisayarında `Spike/runs/colab_v7/` altına aç:

```
Spike/
└── runs/
    └── colab_v7/
        ├── 2026..._v2_pretrain_v7_indy_seeds0/
        ├── 2026..._v2_pretrain_v7_indy_seeds0/   (day0 ayrı klasörde)
        ├── ...
        ├── _bitti/
        └── log_*.txt
```

3. Claude'a "Colab eğitimi bitti" de. Test koşusu (`experiments/v2_v7.py --protocol test`) bu klasörleri `--runs` ile okur.

## Başlamadan önce bir kontrol

Rapor eğitimine başlamadan önce yerel ayar zincirinin ilk modelinin (Indy seed 100) Day-0 val R²'sine bak:
`runs/v7_tune_day0.log` içinde `"stage": "day0"` satırı, `indy_20160622_01`, `val_r2`. V6'da (augmentation'sız)
seed 100–102: Indy 0,769 / 0,770 / 0,787, Loco 0,706 / 0,719 / 0,713. Augmentation bunu belirgin düşürüyorsa
(ör. Indy < 0,74), Colab saatlerini harcamadan önce birlikte karar verelim.
