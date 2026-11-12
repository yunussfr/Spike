# Faz V8 planı: çoklu-gün omurgası, hayvan başına yalıtılmış veri bölmesi, ikinci veri seti

## Bağlam

Omurga bugün yalnız Day-0 ile eğitilip donduruluyor. Uyum olmadan (v2-A) sonraki günlerde R² −0,066'ya düşüyor.
Hipotez: omurga tek günün kanal profilini ezberliyor; **geçmişteki birkaç gerçek günü** de görürse sonraki günlerde
daha az "afallar". Bunu yalnız augmentation ile değil, gerçek günlerle test etmek istiyoruz. Aynı deneyde V7'nin
bütün bileşenleri (ReFIT, BOCPD, ölü kanal, Degenhart, beyincik, duygu uzayı) açık kalacak.

Doğrulanan veri durumu:
- Zenodo 583331: Indy 37 oturum (Day-0'dan önce 7 tane), Loco 10 oturum (eski Day-0 ilk oturum).
- Dryad xd2547dkt (CC0): Chewie 10, Mihili 6, Han 5, Lando 3 oturum. Yalnız konum var, hız yok. Dosyanın iç
  yapısı (alan adları, kutu genişliği, deneme mi sürekli mi) henüz **doğrulanmadı**.

Senin kararların: Loco'da Day-0 kaydırılacak; Dryad'dan dört maymun da kullanılacak; Indy testi 2 NeuroBench + 6 ek.

## Strateji (ve neden)

**1. Her hayvan kendi kapalı kutusu.** Her hayvan için ayrı omurga, ayrı kodlayıcı istatistiği, ayrı checkpoint,
ayrı süreç. Bir hayvanın listesine başka hayvanın dosyası girerse kod hata verir.
*Neden:* Farklı hayvanların elektrotları arasında hiçbir eşleşme yok; karıştırmak hem anlamsız hem sızıntı riski.

**2. Her hayvanın oturumları tarih sırasıyla dört role bölünür.**

| Rol | Ne için | Kural |
|---|---|---|
| ÖN (geçmiş günler) | omurganın çoklu-gün ön-eğitimi | hepsi Day-0'dan **önce** |
| Day-0 | ince ayar, W_0 çapası, tüm referanslar (BOCPD, FA, ölü kanal) | tek oturum |
| AYAR | hiperparametre seçimi | Day-0'dan sonra, testte yok |
| TEST | rapor | Day-0'dan sonra, hiçbir eğitim/ayarda yok |

*Neden:* Gerçek bir BCI'da yalnız geçmiş bilinir. Test günlerinden omurgaya hiçbir şey sızmaz. W_0 çapası
kendiliğinden "testten önceki son eğitim günü" olur.

**3. İki omurga kolu, aynı çevrimiçi deney.**
- **S (tek-gün):** yalnız Day-0 ile eğitim (bugünkü `day0` aşaması).
- **M (çoklu-gün):** ÖN + Day-0 ile ön-eğitim → Day-0'da ince ayar → W_0 çapası.
- İkisi de aynı augmentation'la eğitilir; fark yalnız gerçek geçmiş günlerdir.
- ÖN oturumları **Day-0'ın istatistikleriyle** kodlanır. *Neden:* testte Day-k de böyle kodlanıyor; omurga
  eğitimde gerçek kaymayı görsün.
- Her iki omurgada V7 koşul ailesinin **tamamı** koşar (A, R5, G7, D7, ablasyonlar, B7, P7, C7). Çevrimiçi
  bileşenlerin kodu değişmez.

**4. Dryad'a hiç ayar yapılmaz.** Hiperparametreler yalnız Indy/Loco AYAR oturumlarında seçilir, Dryad'da
olduğu gibi kullanılır. Dryad'da rol: ÖN / Day-0 / TEST.
*Neden:* Dryad böylece gerçek bir dış doğrulama olur; iki veri seti birbirinin sonucunu etkileyemez.

**5. Dryad'da hız, konumun türevinden.** Merkezî fark, yalnız kesintisiz parçaların içinde (deneme sınırını
aşmadan). NeuroBench de hızı konumun gradyanından alıyor; yöntem aynı olur.

### Oturum rolleri

**Indy** (Day-0 aynı, NeuroBench ile karşılaştırılabilir)
- ÖN: 0407_02, 0411_01, 0411_02, 0418_01, 0419_01, 0420_01, 0426_01 (2016)
- Day-0: 20160622_01 · AYAR: 0624_03, 1014_04, 0127_03 (değişmez)
- TEST birincil: 0630_01, 20170131_02 · TEST ek: 0627_01, 0916_01, 1006_02, 1027_03, 1207_02, 20170124_01
- Kalan 17 oturuma dokunulmaz (yedek).

**Loco** (Day-0 kaydı; eski V6/V7 Loco sayılarıyla doğrudan karşılaştırılamaz)
- ÖN: 0210_03, 0213_02 · Day-0: 0214_02 · AYAR: 0216_02, 0217_02, 0227_04
- TEST birincil: 0215_02, 0301_05 · TEST ek: 0228_02, 0302_02

**Dryad** (geçici; Adım 0 denetiminden sonra kesinleşir)
- Chewie: ÖN 2015 ×4 · Day-0 20160929 · TEST 1005, 1006, 1007, 1014, 1021.
  2015 ve 2016 farklı implantsa iki ayrı kutu olur (2016: ÖN 0929, 1005, 1006 · Day-0 1007 · TEST 1014, 1021).
- Mihili: ÖN 0217, 0218 · Day-0 0303 · TEST 0304, 0306, 0307
- Han: ÖN 1127, 1128 · Day-0 1201 · TEST 1204, 1207
- Lando: ÖN 0731 · Day-0 0803 · TEST 0917 (yalnız 1 geçmiş gün; en zayıf halka)

## Adımlar

### Adım 0 — Denetim (kod yazmadan önce, kapı)
`scripts/audit_sessions.py` (yeni, salt okuma) → `results/v8/audit.md`.
- Zenodo: ÖN oturumlarının kanal sayısı ve `chan_names`'i Day-0 ile aynı mı (bazı oturumlar M1+S1, 192 kanal)?
  Farklıysa yalnız M1 kanalları alınır ya da oturum ÖN'den çıkar.
- Dryad: alan adları, kutu genişliği, deneme/sürekli, birim → elektrot bilgisi, bölge (M1/PMd/S1), hedef alanları,
  konum birimi, Chewie'nin implantı.
- Yerelde iki küçük dosya yeter (indy_20160419_01 ~377 MB, Chewie_CO_20150630 ~167 MB); indirmeyi sen yaparsın.
- Kapı: kutu genişliği 4 ms değilse Dryad kendi `time.dt_ms`'iyle koşar (zaman sabitleri ms cinsinden, aynı kalır).
  Sonuçları sana tabloyla getiririm; rol tablosu orada kesinleşir.

### Adım 1 — Dokümanlar (kural önce, kod sonra)
Bu planın onayı aşağıdaki ARCHITECTURE.md değişikliklerinin onayı sayılır:
- §12.8: `multiday` aşaması; §12.9: omurga kolu S / M; yeni §12.21: oturum rolleri, hayvan yalıtımı, Dryad
  yükleyicisi ve hız türevi; §11 değişiklik kaydı. KV6'ya ek: ÖN oturumları Day-0 istatistikleriyle kodlanır.
- `docs/PhasesV2.md` Faz V8 (çıkış kriteriyle), `docs/Decision.md` KR-051, `CLAUDE.md` durum satırı.

### Adım 2 — Veri katmanı (testler önce)
- `src/v2/data.py`: `session_date` iki ad biçimini de okur; `session_roles(cfg, primate)` ve `check_roles(cfg)`
  (`check_tune_disjoint`'i genişletir): roller ayrık, ÖN < Day-0 < diğerleri, her dosyanın öneki hayvanla aynı.
  `crossday_sessions` protokolü `test | tune | ext` alır.
- `src/data/trialdata.py` (yeni): Dryad `.mat` → mevcut önbellek biçimi (`x, y, val_start, test_start` +
  `_task.pt`). Birimler elektrot başına toplanır (sabit C); denemeler zaman sırasıyla eklenir; konum mm'ye çevrilir
  (ReFIT yarıçapları mm); hedef: go işaretinden önce merkez, sonra dış hedef; bölme 80/10/10 deneme sınırında.
- `velocity_from_position(pos, dt, seg_starts, smooth_ms)`: parça içi merkezî fark; yumuşatma YAML'dan (KV8).
- `src/data/primate.py`: `load_session` / `load_task_state`, `data.trialdata_files`'taki adlar için yeni
  yükleyiciye yönlenir. Önbellek biçimi aynı olduğundan geri kalan kod değişmez.
- `scripts/download_primate.py`: yeni dosyalar için md5 (Zenodo API) `data.md5`'e; `--only pre | ext`.

### Adım 3 — Çoklu-gün eğitimi
- `experiments/v2_pretrain.py`: yeni `multiday` aşaması. `prepare_session(..., stats_from=sessions[day0])` ile ÖN
  kodlanır; `train_model(m, [ÖN…, Day-0])`; ardından Day-0 ince ayarı, `save_anchor()`,
  `multiday_<hayvan>_seed<s>.pt`. Day-k val skoru `day0` aşamasındaki gibi kaydedilir.
- `multiday.finetune: full | readout` (YAML). `readout`: omurga donuk, yalnız başlıklar Day-0'a uyar; ince ayarın
  çoklu-gün bilgisini silmesine karşı. Hangisi kalacağı AYAR oturumlarında seçilir.
- `src/v2/train.py` `TrainStep`: optimizer yalnız `requires_grad` parametreleri alır (küçük değişiklik).
- Config: `configs/v2_pretrain_v8.yaml`, `configs/v2_v8.yaml` (Indy/Loco rolleri), `configs/v2_v8_dryad.yaml`.

### Adım 4 — Çevrimiçi deney ve tablolar
- `experiments/v2_v7.py`: `--ckpt-stage day0 | multiday` (yüklenecek checkpoint), `--protocol ext`, satırlara
  `backbone: S | M`. Elektrot kaybı senaryosu Day-0 modeliyle test/ek oturumlarda da koşabilsin.
- `scripts/make_tables_v2.py`: omurga × koşul tablosu; oturum eşli farklar (M−S; D7−A'nın S ve M'deki farkı), Holm.

### Adım 5 — Colab
Mevcut akış genişler (`notebooks/README.md`, `paketle.py`, Drive `Spike_colab/`):
- `v8_veri.ipynb`: Zenodo dosyalarını Colab'da indirir, önbellek üretir, Drive'a yazar (yerelde ~12 GB indirmeye
  gerek kalmaz). Dryad önce doğrudan denenir; olmazsa dosyaları Drive'a sen koyarsın.
- `v8_egitim.ipynb`: `HAYVAN` parametresi, aşamalar `multiday` + `day0`, seed başına ayrı süreç, `--resume`.
- `v8_cevrimici.ipynb`: `v2_v7.py`'yi S ve M için koşar. L4 GPU; S ve M **aynı donanımda**.
- `paketle.py`: yeni config ve önbellekler; `KOD_SURUMU.json` aynen (KV8).

### Adım 6 — Koşu sırası
1. Yerel duman (CPU/GPU, `--epochs 1`).
2. Ayar: Indy/Loco AYAR oturumları, seed 100–102, val → `multiday.finetune` ve Loco'nun yeni Day-0'ı için ayarlar.
3. Rapor: seed 0–4, TEST birincil + ek, S ve M.
4. Dryad: ayarsız, seed 0–4.
V7 ayar koşuları önce biter (V8 onların hiperparametrelerini kullanır); kod işi bu sırada yapılabilir.

## Göreceğin çıktılar

| Çıktı | İçinde ne var | Neye bakacaksın |
|---|---|---|
| `results/v8/audit.md` | oturum başına kanal sayısı, süre, kutu, alanlar | ÖN oturumları Day-0 ile uyumlu mu |
| eğitim `table.md` | oturum × aşama (`day0`, `multiday`): val R², hızlı/yavaş oran, KV7 ✗ | Day-0 val R² iki kolda yakın mı (Indy ≈ 0,77); donuk Day-k R² M'de daha yüksek mi |
| çevrimiçi `results.json` | oturum × seed × koşul × omurga: R², `t_detect_s`, `n_dead_end`, `align_fits`, `label_quality_r2`, η̄, ᾱ, oranlar | bileşenler çalışıyor mu (BOCPD tespit süresi, ReFIT etiket kalitesi) |
| ana tablo | koşul × omurga, oturumlar üzerinden ort ± std; M−S farkı ve Holm p | asıl sonuç |
| elektrot tablosu | toparlanma süresi (Kaplan–Meier), kayıp öncesi/sonrası R² | M omurgası kanal kaybına daha dayanıklı mı |

Sonuç üç biçimde çıkabilir; sayı vaat edemem:
1. **A ve D7 ikisi de M'de yükselir** → darboğaz omurgaydı; çoklu-gün işe yarıyor.
2. **A yükselir, D7 aynı kalır** → yerel uyum zaten telafi ediyordu; çoklu-gün uyuma duyulan ihtiyacı azaltıyor.
3. **Fark yok** → kayma omurgada değil, girdi ölçeği / okumada; bu da raporlanacak geçerli bir bulgu.

## Doğrulama

- `uv run pytest` (hepsi geçmeli). Yeni testler `tests/v2/test_v8_data.py`, `tests/v2/test_v8_train.py`:
  - rol ihlalleri hata verir (çakışma, ÖN ≥ Day-0, yabancı hayvan dosyası);
  - türev: sinüs konumda analitik kosinüsle eşleşir; sabit konumda 0; parça sınırından sızma yok;
  - birim → elektrot toplamı; bölme sınırları deneme kenarında;
  - `multiday` dumanı: ÖN, Day-0 istatistikleriyle kodlanıyor; `readout` modunda omurga ağırlıkları bit bit aynı;
  - KV1/KV6 testleri değişmeden geçer.
- Dryad sağlık kontrolü: türev hızının dağılımı makul; `label_quality_r2` pozitif; Day-0 kalibrasyonunda BOCPD
  yanlış alarmı ≤ %1.
- Uçtan uca duman: bir hayvan, 1 seed, 1 epoch, `multiday` + `day0` → `v2_v7.py` iki omurgayla → tablo üretimi.

## Riskler

- Indy'nin Nisan 2016 oturumları Day-0 ile aynı kanal düzeninde olmayabilir (Adım 0 karar verir).
- Dryad kutu genişliği kaba olabilir (10–30 ms); o zaman Dryad sonuçları ayrı Δt ile, ayrı tabloda.
- Dryad'da sıralanmış birimler: birimi olmayan elektrot 0 sayar, ölü kanal dedektörünü tetikleyebilir.
- Han ve Lando büyük olasılıkla S1 (duyu korteksi) ve az oturumlu; ayrı satırda raporlanır, ana iddiaya katılmaz.
- Loco'da yalnız 2 geçmiş gün var (4 günlük aralık); etkisi küçük çıkabilir.
