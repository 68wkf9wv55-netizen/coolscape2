"""
CoolScape AI – v4 prototip (Siemens Türkiye Climathon)
Optimizasyon tabanlı karar destek: kısıtlı soğutma suyunu saat saat ısı stresine göre dağıtır.

Ne gerçek, ne temsili:
- Hava verisi GERÇEK: Open-Meteo. Kayıtlı günler için geçmiş arşiv (ERA5 tabanlı), "Yarın" için tahmin API'si.
- Isı yüzeyi GERÇEK (varsa): Landsat 8/9 yüzey sıcaklığı (+ varsa Sentinel-2 NDVI), H3 altıgenler
  (veri_hazirla.py ile bir kez üretilen isi_yuzeyi_h3.csv, repoda durur; uygulama kendisi okur).
- Hassas tesisler GERÇEK: OpenStreetMap (Overpass API), uygulama açılışta kendisi çeker, 24 saat önbellekte tutar.
- Park değerleri TEMSİLİ: nüfus, kırılgan grup, hat kapasitesi, pompa katsayısı, arıtılmış su erişimi.
- Soğutma eğrisi LİTERATÜR ÖLÇEKLİ: azami etki Paris saha ölçümlerinden (Hendel vd. 2016, ~1 °C UTCI),
  belirsizlik bandı Melbourne/Paris ölçümlerinden (0,5–2 °C), doyma Adelaide çalışmasından (Broadbent vd. 2018).
  Antalya'da yerinde doğrulanmadı; doğrulama 2027 saha pilotunun işi.

Çalıştırma:  pip install -r requirements.txt  &&  python -m streamlit run coolscape_app.py
"""

import base64
import io
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import altair as alt
import h3
import numpy as np
import pandas as pd
import pydeck as pdk
import pulp
import requests
import streamlit as st
from PIL import Image
from pythermalcomfort.models import utci

# ---------------------------------------------------------------------------
# SABİTLER VE VARSAYIMLAR
# ---------------------------------------------------------------------------
ANTALYA_LAT, ANTALYA_LON = 36.89, 30.71
SAATLER = list(range(8, 21))  # 08:00–20:00 karar ufku
GUNLER = {
    "25 Temmuz 2025 · kayıtlı sıcak hava dalgası (resmî ölçüm 46,1 °C)": ("arsiv", "2025-07-25"),
    "10 Ağustos 2025 · kayıtlı sıcak gün (MGM: Kepez 45,2 °C)": ("arsiv", "2025-08-10"),
    "Yarın · Open-Meteo tahmini": ("tahmin", None),
}
ISI_YUZEYI_CSV = "isi_yuzeyi_h3.csv"
UTCI_ESIK = 32.0          # "güçlü ısı stresi" alt sınırı
OYNATMA_ADIM_S = 1.0      # oynatmada bir saatin ekranda kalma süresi (s)
BBOX = (36.85, 30.62, 36.97, 30.80)  # güney, batı, kuzey, doğu (pilot bölge)

# UTCI ısı stresi sınıfları: eşikler resmî (Bröde vd. 2012); renkler bu uygulamanın seçimidir (YlOrRd tabanlı)
UTCI_SINIFLARI = [
    (-99, 26, "Isı stresi yok", (120, 144, 156)),
    (26, 32, "Orta", (254, 204, 92)),
    (32, 38, "Güçlü", (253, 141, 60)),
    (38, 46, "Çok güçlü", (240, 59, 32)),
    (46, 99, "Aşırı", (189, 0, 38)),
]

# Soğutma eğrisi: saha ölçümlerinden ölçeklenir (kaynaklar uygulamanın "Neden?" bölümünde)
E_REF = 1.0              # °C: sulanan noktada yaya seviyesi UTCI düşüşü, merkezi değer
                         # (Hendel vd. 2016, Paris: −0,93 ve −1,03 °C UTCI)
E_ARALIK = (0.5, 2.0)    # çarpan: 0,5–2,0 °C (Traill vd. 2024: −0,2…−2,0 °C; Parison vd. 2020: −1,9 °C)
VPD_DOYMA = 2.5          # kPa, VARSAYIM: tipik sıcak yaz öğleden sonrası; daha nemli saatlerde buharlaşma
                         # ve dolayısıyla serinlik VPD ile orantılı azalır (Penman tipi buharlaşma ilişkisi)
X0_MM = 0.5              # mm (L/m²) / saat: azalan getiri ölçeği. ~1,5 mm/saatte etkinin %95'i alınır;
                         # Broadbent vd. 2018: günde ~20 L/m² üzeri ek sulama ek serinlik getirmiyor
X0_ARALIK = (0.5, 2.0)   # çarpan: 0,25–1,0 mm duyarlılık
KIRILIM = [0, 2, 4, 6, 8, 12, 16, 24, 32]
MIN_DOZ = 2.0
CEVRE_YARICAP_M = 500   # parkın koruduğu çevre (ısı stresi bu halkadan ölçülür)
YURUME_5DK_M = 400      # kuş uçuşu ~5 dk yürüme
YURUME_10DK_M = 800     # kuş uçuşu ~10 dk yürüme
ETKI_OLCEGI_M = 150     # varsayım: park soğumasının çevreye taşınma ölçeği (e^(−d/150 m))
ETKI_KESME_M = 500      # bu mesafeden sonra etki sıfır kabul edilir

# ---------------------------------------------------------------------------
# YÖNTEMLER: aynı su bütçesi üç farklı soğutma yöntemine dağıtılabilir
# ---------------------------------------------------------------------------
# Park sulama ve yol ıslatma: E = E_REF × min(VPD/2,5; 1) × (1 − e^(−doz/x0)), doz mm cinsinden
# Sisleme: aç/kapa; açıkken E = SISLEME_E × min(VPD/2,5; 1), sabit su tüketimi
SISLEME_E = 4.4          # °C UTCI: Vanos vd. 2022 (Tempe, Arizona, kuru sıcak; güneşte sisli alan ortalaması)
SISLEME_SU_M3_SAAT = 0.12  # VARSAYIM: ~20 nozullu bir istasyon, nozul başına ~0,1 L/dk
SISLEME_ALAN_M2 = 100    # sislenen alan (~10 × 10 m)
ISLATMA_ALAN_M2 = 3000   # VARSAYIM: ~300 m cadde × 10 m kaldırım/yol
ISLATMA_KAPASITE = 5.0   # m³/saat: bir arazözün bir güzergâhta saatte boşaltabileceği (varsayım)
YONTEMLER = {
    # tür: (etiket, harita rengi, yayılım ölçeği m, kesme m)
    "park": ("Park sulama", (30, 136, 229), ETKI_OLCEGI_M, ETKI_KESME_M),
    "sisleme": ("Sisleme noktası", (0, 188, 212), 15, 50),
    "islatma": ("Yol ıslatma", (149, 117, 205), 60, 200),
}
ADAY_SISLEME, ADAY_ISLATMA = 4, 3  # sıcak nokta haritasından otomatik seçilen aday sayısı
ADAY_ARALIK_M = 1000               # adaylar birbirinden ve parklardan en az bu kadar uzak
ADAY_PARK_UZAKLIK_M = (600, 3000)  # yerleşim içinde kalsın: en yakın parka uzaklık aralığı (sezgisel)
ADAY_DOGU_SINIRI = 30.785          # havalimanı ve çevresindeki açık alanları dışarıda bırakır

# Erişim (o saatte yöntemin dokunduğu kişi sayısı) – TEMSİLİ varsayımlar, gelişmiş ayarlardan değişir
ERISIM_VARSAYILAN = {"park_carpan": 50, "sisleme": 40, "islatma": 150}
SEHIR_KIRILGAN_YUZDE = 15          # aday noktalar için temsili kırılgan nüfus oranı

# Öncelik ön ayarları: operatör yalnızca birini seçer, ayrıntı "Gelişmiş ayarlar"da
ON_AYARLAR = {
    "Dengeli": {"kirilgan_pay": 40, "w_k": 0.5},
    "Kırılgan nüfusa öncelik": {"kirilgan_pay": 60, "w_k": 1.5},
    "En çok toplam serinlik": {"kirilgan_pay": 0, "w_k": 0.0},
}

# ---------------------------------------------------------------------------
# PARK VERİSİ – pilot bölge Muratpaşa–Kepez
# Koordinatlar: Google Haritalar. Nüfus, kırılgan grup, hat kapasitesi ve arıtılmış su erişimi TEMSİLİ.
# ---------------------------------------------------------------------------
PARKLAR = pd.DataFrame({
    "park": ["Karaalioğlu Parkı", "Atatürk Parkı", "Düden Park", "Dokuma Park", "Tek Sarnıç Kent Parkı"],
    "ilce": ["Muratpaşa", "Muratpaşa", "Muratpaşa", "Kepez", "Kepez"],
    "lat": [36.87964, 36.88385, 36.85378, 36.91285, 36.90318],
    "lon": [30.70488, 30.68648, 30.78207, 30.67538, 30.63917],
    "yuzey_farki_c": [0.7, -0.3, -2.3, 2.7, 1.5],
    "yesil_skoru": [7, 6, 8, 4, 6],
    "nufus_skoru": [8, 7, 9, 5, 7],
    "hassas_yuzde": [20, 15, 25, 10, 18],
    "pompa_katsayisi": [1.2, 1.1, 1.5, 1.8, 1.3],
    "hat_kapasitesi_m3_saat": [15, 12, 20, 10, 15],
    "sulanan_alan_ha": [1.5, 1.2, 2.0, 1.0, 1.5],
    "aritilmis_su_erisimi": [False, False, True, True, False],
})
PARKLAR["tur"] = "park"


# ---------------------------------------------------------------------------
# VERİ
# ---------------------------------------------------------------------------
HAVA_DEGISKENLERI = "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation"


def _hava_tablosu(h):
    df = pd.DataFrame({
        "zaman": pd.to_datetime(h["time"]),
        "sicaklik_c": h["temperature_2m"], "bagil_nem": h["relative_humidity_2m"],
        "ruzgar_ms": h["wind_speed_10m"], "gunes_wm2": h["shortwave_radiation"],
    })
    df["saat"] = df["zaman"].dt.hour
    return df


@st.cache_data(show_spinner=False)
def hava_verisi_cek(tarih: str) -> pd.DataFrame:
    params = {"latitude": ANTALYA_LAT, "longitude": ANTALYA_LON, "start_date": tarih, "end_date": tarih,
              "hourly": HAVA_DEGISKENLERI, "wind_speed_unit": "ms", "timezone": "Europe/Istanbul"}
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=20)
    r.raise_for_status()
    return _hava_tablosu(r.json()["hourly"])


@st.cache_data(show_spinner=False, ttl=3600)
def hava_tahmini_cek(tarih: str) -> pd.DataFrame:
    params = {"latitude": ANTALYA_LAT, "longitude": ANTALYA_LON, "start_date": tarih, "end_date": tarih,
              "hourly": HAVA_DEGISKENLERI, "wind_speed_unit": "ms", "timezone": "Europe/Istanbul"}
    r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
    r.raise_for_status()
    return _hava_tablosu(r.json()["hourly"])


def csv_hava_verisi(dosya) -> pd.DataFrame:
    df = pd.read_csv(dosya)
    eksik = {"saat", "sicaklik_c", "bagil_nem", "ruzgar_ms", "gunes_wm2"} - set(df.columns)
    if eksik:
        raise ValueError(f"CSV'de eksik sütunlar: {', '.join(sorted(eksik))}")
    return df


@st.cache_data(show_spinner=False)
def isi_yuzeyi_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    df = pd.read_csv(yol)
    if not {"hex", "lst_c", "lat", "lon"} <= set(df.columns):
        return None
    df = df.dropna(subset=["lst_c"]).reset_index(drop=True)
    df["lst_fark_c"] = df["lst_c"] - df["lst_c"].median()
    return df


OSM_KATEGORILER = {
    "Okul / kreş": (r'["amenity"~"^(school|kindergarten)$"]', (245, 245, 245)),
    "Sağlık": (r'["amenity"~"^(hospital|clinic)$"]', (240, 98, 146)),
    "Huzurevi": (r'["social_facility"="nursing_home"]', (174, 213, 129)),
    "İçme suyu": (r'["amenity"="drinking_water"]', (178, 235, 242)),
}
OVERPASS_ADRESLERI = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter",
                      "https://maps.mail.ru/osm/tools/overpass/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
TESIS_CSV = "tesisler_osm.csv"  # osm_hazirla.py ile bir kez üretilip repoya konur; varsa öncelikli kullanılır


@st.cache_data(show_spinner=False)
def tesis_dosyasi_oku(yol: str, degistirilme: float):
    if not os.path.exists(yol):
        return None
    df = pd.read_csv(yol, sep=None, engine="python")
    if {"@lat", "@lon"} <= set(df.columns):  # tarayıcıdan indirilen ham Overpass CSV'si
        t = df.rename(columns={"@lat": "lat", "@lon": "lon"})
        for c in ("amenity", "social_facility", "name"):
            if c not in t:
                t[c] = None
        am, sf = t["amenity"].fillna(""), t["social_facility"].fillna("")
        t["kategori"] = np.select(
            [am.isin(["school", "kindergarten"]), am.isin(["hospital", "clinic"]),
             (sf == "nursing_home") | (am == "nursing_home"), am == "drinking_water"],
            ["Okul / kreş", "Sağlık", "Huzurevi", "İçme suyu"], default="")
        t["ad"] = t["name"].fillna(t["kategori"])
        df = t
    if not {"kategori", "lat", "lon"} <= set(df.columns):
        return None
    df = df.dropna(subset=["lat", "lon"])
    df = df[df["kategori"].isin(list(OSM_KATEGORILER))].copy()
    if "ad" not in df:
        df["ad"] = df["kategori"]
    return df[["kategori", "ad", "lat", "lon"]].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=86400)
def osm_tesisleri_cek():
    """OpenStreetMap'ten pilot bölgedeki hassas tesisler. Erişilemezse (None, hata) döner; uydurma veri yok."""
    g, b, k, d = BBOX
    parcalar = []
    for etiket, _ in OSM_KATEGORILER.values():
        parcalar.append(f"nwr{etiket}({g},{b},{k},{d});")
    parcalar.append(f'nwr["amenity"="nursing_home"]({g},{b},{k},{d});')
    sorgu = f"[out:json][timeout:25];({''.join(parcalar)});out center tags;"
    son_hata = None
    for adres in OVERPASS_ADRESLERI:
        try:
            r = requests.post(adres, data={"data": sorgu}, timeout=60,
                              headers={"User-Agent": "CoolScape-AI-prototip/0.4 (Climathon)", "Accept": "application/json"})
            r.raise_for_status()
            satirlar = []
            for e in r.json().get("elements", []):
                t = e.get("tags", {})
                lat = e.get("lat", e.get("center", {}).get("lat"))
                lon = e.get("lon", e.get("center", {}).get("lon"))
                if lat is None or lon is None:
                    continue
                if t.get("amenity") in ("school", "kindergarten"):
                    kat = "Okul / kreş"
                elif t.get("amenity") in ("hospital", "clinic"):
                    kat = "Sağlık"
                elif t.get("social_facility") == "nursing_home" or t.get("amenity") == "nursing_home":
                    kat = "Huzurevi"
                elif t.get("amenity") == "drinking_water":
                    kat = "İçme suyu"
                else:
                    continue
                satirlar.append({"kategori": kat, "ad": t.get("name", kat), "lat": lat, "lon": lon})
            return pd.DataFrame(satirlar, columns=["kategori", "ad", "lat", "lon"]).drop_duplicates(), None
        except Exception as hata:
            son_hata = f"{adres.split('/')[2]}: {type(hata).__name__}"
    return None, str(son_hata)


def parklari_guncelle(parklar, yuzey):
    """Isı yüzeyi varsa:
    - yüzey sıcaklığı farkı parkın ÇEVRESİNDEN (~500 m halka) alınır: korunacak olan çevredeki mahalledir;
    - yeşil skoru parkın KENDİSİNDEN (en yakın altıgenler) alınır: suyun buharlaşma etkisini bitki örtüsü belirler."""
    p = parklar.copy()
    p[["yuzey_farki_c", "yesil_skoru"]] = p[["yuzey_farki_c", "yesil_skoru"]].astype(float)
    p["yuzey_kaynak"] = "temsili"
    p["yesil_kaynak"] = "temsili"
    if yuzey is None:
        return p
    cozunurluk = h3.get_resolution(yuzey["hex"].iloc[0])
    aralik_m = h3.average_hexagon_edge_length(cozunurluk, unit="m") * 3 ** 0.5
    k_cevre = max(1, int(round(CEVRE_YARICAP_M / aralik_m)))
    idx = yuzey.set_index("hex")
    for i, r in p.iterrows():
        merkez = h3.latlng_to_cell(r["lat"], r["lon"], cozunurluk)
        cevre = [c for c in h3.grid_disk(merkez, k_cevre) if c in idx.index]
        yakin = [c for c in h3.grid_disk(merkez, 1) if c in idx.index]
        if cevre:
            p.at[i, "yuzey_farki_c"] = round(float(idx.loc[cevre, "lst_fark_c"].mean()), 2)
            p.at[i, "yuzey_kaynak"] = f"Landsat (~{CEVRE_YARICAP_M} m çevre)"
        if yakin and "ndvi" in idx.columns and idx.loc[yakin, "ndvi"].notna().any():
            ndvi = float(idx.loc[yakin, "ndvi"].mean())
            p.at[i, "yesil_skoru"] = round(float(np.clip(1 + ndvi / 0.8 * 9, 1, 10)), 1)
            p.at[i, "yesil_kaynak"] = "Sentinel-2 NDVI"
    return p


def aday_noktalar(yuzey, parklar):
    """Sisleme ve yol ıslatma için ADAY konumlar: ısı yüzeyinin en sıcak altıgenlerinden, yerleşim içinde kalan,
    birbirinden ve parklardan en az 1 km uzak noktalar. Kesin yer (durak, meydan, cadde) sahada belediyeyle seçilir."""
    if yuzey is None:
        return parklar.iloc[0:0].copy()
    y = yuzey[yuzey["lon"] <= ADAY_DOGU_SINIRI].copy()
    d_park = np.min([mesafe_m(r["lat"], r["lon"], y["lat"].values, y["lon"].values)
                     for _, r in parklar.iterrows()], axis=0)
    y["d_park"] = d_park
    y = y[(y["d_park"] >= ADAY_PARK_UZAKLIK_M[0]) & (y["d_park"] <= ADAY_PARK_UZAKLIK_M[1])]
    y = y.sort_values("lst_c", ascending=False)
    secilen = []
    for _, r in y.iterrows():
        if all(mesafe_m(r["lat"], r["lon"], a["lat"], a["lon"]) >= ADAY_ARALIK_M for a in secilen):
            secilen.append(r)
        if len(secilen) >= ADAY_SISLEME + ADAY_ISLATMA:
            break
    satirlar = []
    for i, r in enumerate(secilen):
        tur = "sisleme" if i < ADAY_SISLEME else "islatma"
        no = i + 1 if tur == "sisleme" else i - ADAY_SISLEME + 1
        en_yakin = parklar.iloc[int(np.argmin(mesafe_m(r["lat"], r["lon"], parklar["lat"].values,
                                                       parklar["lon"].values)))]
        satirlar.append({
            "park": f"{'Sisleme' if tur == 'sisleme' else 'Islatma'} {no} (aday)", "ilce": en_yakin["ilce"],
            "lat": r["lat"], "lon": r["lon"], "yuzey_farki_c": round(float(r["lst_fark_c"]), 2),
            "yesil_skoru": 0.0, "nufus_skoru": 0, "hassas_yuzde": SEHIR_KIRILGAN_YUZDE, "pompa_katsayisi": 1.0,
            "hat_kapasitesi_m3_saat": SISLEME_SU_M3_SAAT if tur == "sisleme" else ISLATMA_KAPASITE,
            "sulanan_alan_ha": (SISLEME_ALAN_M2 if tur == "sisleme" else ISLATMA_ALAN_M2) / 10_000,
            "aritilmis_su_erisimi": tur == "islatma", "tur": tur,
            "yuzey_kaynak": "Landsat (altıgenin kendisi)", "yesil_kaynak": "–"})
    return pd.DataFrame(satirlar)


# ---------------------------------------------------------------------------
# FİZİK
# ---------------------------------------------------------------------------
def vpd_kpa(sicaklik_c, bagil_nem):
    t = np.asarray(sicaklik_c, dtype=float)
    rh = np.asarray(bagil_nem, dtype=float)
    try:
        from metpy.calc import saturation_vapor_pressure
        from metpy.units import units
        es = saturation_vapor_pressure(t * units.degC).to("kPa").magnitude
    except Exception:
        es = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return es * (1 - rh / 100.0)


def utci_hesapla(tdb, mrt, ruzgar, nem):
    """MRT yaklaşımı: T + 0,02·SW + yüzey sıcaklığı farkı (kaba). Geçerlilik sınırına kırpılır."""
    tdb = np.asarray(tdb, dtype=float)
    mrt = np.clip(np.asarray(mrt, dtype=float), tdb - 30, tdb + 70)
    v = np.clip(np.asarray(ruzgar, dtype=float), 0.5, 17.0)
    sonuc = utci(tdb=tdb, tr=mrt, v=v, rh=np.asarray(nem, dtype=float))
    return np.asarray(getattr(sonuc, "utci", sonuc), dtype=float)


def park_saat_tablosu(hava, parklar):
    h = hava[hava["saat"].isin(SAATLER)]
    parcalar = []
    for _, p in parklar.iterrows():
        d = h.copy()
        d["park"] = p["park"]
        d["mrt_c"] = d["sicaklik_c"] + 0.02 * d["gunes_wm2"] + p["yuzey_farki_c"]
        parcalar.append(d)
    df = pd.concat(parcalar, ignore_index=True)
    df["utci_c"] = utci_hesapla(df["sicaklik_c"], df["mrt_c"], df["ruzgar_ms"], df["bagil_nem"])
    df["vpd_kpa"] = vpd_kpa(df["sicaklik_c"], df["bagil_nem"])
    df["tehlike"] = (df["utci_c"] - UTCI_ESIK).clip(lower=0).fillna(0)
    return df


def yuzey_utci(yuzey, hava_satiri):
    t, sw = hava_satiri["sicaklik_c"], hava_satiri["gunes_wm2"]
    n = len(yuzey)
    return utci_hesapla(np.full(n, t), t + 0.02 * sw + yuzey["lst_fark_c"].values,
                        np.full(n, hava_satiri["ruzgar_ms"]), np.full(n, hava_satiri["bagil_nem"]))


def kent_utci(satir):
    return float(utci_hesapla([satir["sicaklik_c"]], [satir["sicaklik_c"] + 0.02 * satir["gunes_wm2"]],
                              [satir["ruzgar_ms"]], [satir["bagil_nem"]])[0])


def utci_sinif(deger):
    for alt_s, ust_s, ad, renk in UTCI_SINIFLARI:
        if alt_s <= deger < ust_s:
            return ad, renk
    return UTCI_SINIFLARI[-1][2], UTCI_SINIFLARI[-1][3]


# ---------------------------------------------------------------------------
# OPTİMİZASYON (MILP)
# ---------------------------------------------------------------------------
def sogutma_egrisi(x_m3, emax, x0):
    """x0: m³ cinsinden azalan getiri ölçeği = X0_MM × sulanan alan."""
    return emax * (1 - np.exp(-np.asarray(x_m3, dtype=float) / np.asarray(x0, dtype=float)))


def girdileri_hazirla(tablo, noktalar, w_kirilganlik, erisim):
    """Her nokta × saat için öncelik, azami etki, azalan getiri ölçeği, kapasite ve erişim (kişi)."""
    p = noktalar.set_index("park")
    t = tablo.copy()
    t["tur"] = t["park"].map(p["tur"])
    t["kisi"] = np.select([t["tur"] == "park", t["tur"] == "sisleme"],
                          [t["park"].map(p["nufus_skoru"]) * erisim["park_carpan"], erisim["sisleme"]],
                          default=erisim["islatma"]).astype(float)
    t["oncelik"] = t["tehlike"] * t["kisi"] / 100.0 * (1 + w_kirilganlik * t["park"].map(p["hassas_yuzde"]) / 30.0)
    vpd_f = np.minimum(t["vpd_kpa"] / VPD_DOYMA, 1.0)
    t["emax"] = np.where(t["tur"] == "sisleme", SISLEME_E, E_REF) * vpd_f
    t["alan_m2"] = t["park"].map(p["sulanan_alan_ha"]) * 10_000
    t["x0"] = np.where(t["tur"] == "sisleme", SISLEME_SU_M3_SAAT / 10.0, X0_MM * t["alan_m2"] / 1000.0)
    t["kapasite"] = t["park"].map(p["hat_kapasitesi_m3_saat"]).astype(float)
    t["pompa"] = t["park"].map(p["pompa_katsayisi"])
    return t


def degisken(model, ad, alt_s=None, ust_s=None, tur="Continuous"):
    if hasattr(model, "add_variable"):  # PuLP 4.x
        return model.add_variable(ad, lowBound=alt_s, upBound=ust_s, cat=tur)
    return pulp.LpVariable(ad, lowBound=alt_s, upBound=ust_s, cat=tur)


def cozucu_sec():
    for ad in ("PULP_CBC_CMD", "HiGHS", "HiGHS_CMD", "COIN_CMD"):
        sinif = getattr(pulp, ad, None)
        if sinif is None:
            continue
        try:
            c = sinif(msg=False)
            if c.available():
                return c
        except Exception:
            continue
    raise RuntimeError("MILP çözücüsü bulunamadı (pulp==2.9.0 kurulu olmalı).")


TUM_TURLER = ("park", "sisleme", "islatma")


def optimize_et(t, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, x0_carpan=1.0,
                turler=TUM_TURLER, park_kesme=18, park_gunluk_mm=6.0):
    """Öncelik ağırlıklı UTCI düşüşünü en çoklayan MILP.
    Park/ıslatma: parçalı doğrusal azalan getiri eğrisi. Sisleme: aç/kapa, sabit su.
    Çim güvenliği: park sulaması akşam kesme saatinden sonra yok; park başına günlük üst sınır (mm)."""
    if float(t["oncelik"].sum()) <= 0:  # hiçbir saatte güçlü ısı stresi yok: sulama önerilmez
        sonuc = t.copy()
        sonuc["su_m3"] = 0.0
        return sonuc, "Optimal"
    model = pulp.LpProblem("CoolScape", pulp.LpMaximize)
    su, amac, toplam, kirilgan, park_gunluk = {}, [], [], [], {}
    for idx, r in t.iterrows():
        y = degisken(model, f"y_{idx}", tur="Binary")
        kapali = (r["tur"] not in turler or r["vpd_kpa"] < vpd_min
                  or (r["tur"] == "park" and r["saat"] >= park_kesme))
        if kapali:
            model += y == 0
        if r["tur"] == "sisleme":
            x = SISLEME_SU_M3_SAAT * y
            amac.append(r["oncelik"] * float(sogutma_egrisi(SISLEME_SU_M3_SAAT, r["emax"], r["x0"])) * y)
            su[idx] = ("sabit", y)
        else:
            segs = []
            for k in range(len(KIRILIM) - 1):
                a, b = KIRILIM[k], min(KIRILIM[k + 1], r["kapasite"])
                if b <= a:
                    break
                v = degisken(model, f"s_{idx}_{k}", alt_s=0, ust_s=float(b - a))
                x0 = r["x0"] * x0_carpan
                egim = (sogutma_egrisi(b, r["emax"], x0) - sogutma_egrisi(a, r["emax"], x0)) / (b - a)
                amac.append(r["oncelik"] * egim * v)
                if enerji_opt:
                    amac.append(-0.01 * r["pompa"] * v)
                segs.append(v)
            x = pulp.lpSum(segs)
            model += x <= r["kapasite"] * y
            model += x >= min(MIN_DOZ, r["kapasite"]) * y
            su[idx] = ("parca", segs)
            if r["tur"] == "park":
                park_gunluk.setdefault(r["park"], ([], r["alan_m2"]))[0].append(x)
        toplam.append(x)
        if r["park"] in kirilgan_parklar:
            kirilgan.append(x)
    for ad, (xs, alan) in park_gunluk.items():
        model += pulp.lpSum(xs) <= park_gunluk_mm * alan / 1000.0
    model += pulp.lpSum(amac)
    model += pulp.lpSum(toplam) <= butce
    if kirilgan and kirilgan_pay > 0:
        model += pulp.lpSum(kirilgan) >= kirilgan_pay * pulp.lpSum(toplam)
    model.solve(cozucu_sec())
    durum = pulp.LpStatus[model.status] if hasattr(pulp, "LpStatus") else str(model.status)
    sonuc = t.copy()
    degerler = []
    for i in t.index:
        tip, v = su[i]
        degerler.append(round(SISLEME_SU_M3_SAAT * (v.value() or 0), 3) if tip == "sabit"
                        else round(sum((s_.value() or 0) for s_ in v), 2))
    sonuc["su_m3"] = degerler
    return sonuc, durum


def esit_dagitim(t, butce, park_kesme=18, park_gunluk_mm=6.0):
    """Karşılaştırma: bütçe yalnızca parklara, izin verilen gündüz saatlerine eşit bölünür; nem, ısı stresi ve
    kırılganlık gözetilmez. Adil olsun diye CoolScape'le AYNI hat kapasitesi ve çim güvenliği kurallarına uyar."""
    sonuc = t.copy()
    sonuc["su_m3"] = 0.0
    izin = (sonuc["tur"] == "park") & (sonuc["saat"] < park_kesme)
    if izin.sum() == 0:
        return sonuc
    pay_ = np.minimum(butce / int(izin.sum()), sonuc.loc[izin, "kapasite"])
    tavan = (park_gunluk_mm * sonuc.loc[izin, "alan_m2"] / 1000.0) / sonuc.loc[izin].groupby("park")["saat"].transform("count")
    sonuc.loc[izin, "su_m3"] = np.minimum(pay_, tavan)
    return sonuc


def etki(sonuc, x0_carpan=1.0, emax_carpan=1.0):
    s = sonuc.copy()
    x0 = np.where(s["tur"] == "sisleme", s["x0"], s["x0"] * x0_carpan)
    s["utci_dusus_c"] = sogutma_egrisi(s["su_m3"], s["emax"] * emax_carpan, x0)
    s["su_mm"] = s["su_m3"] / (s["alan_m2"] / 1000.0)
    s["agirlikli_fayda"] = s["oncelik"] * s["utci_dusus_c"]
    s["kisi_derece_saat"] = s["kisi"] * s["utci_dusus_c"] * (s["tehlike"] > 0)
    s["utci_sonra_c"] = s["utci_c"] - s["utci_dusus_c"]
    return s


@st.cache_data(show_spinner="Su planı hesaplanıyor...")
def plan_hesapla(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, turler, park_kesme, park_mm):
    """Ayarlar her değiştiğinde otomatik çalışır (önbellekli): nominal plan, yalnız-park planı, duyarlılık."""
    kirilgan_parklar = set(kirilgan_parklar)
    ayar = dict(turler=turler, park_kesme=park_kesme, park_gunluk_mm=park_mm)
    plan, durum = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, **ayar)
    if durum != "Optimal":
        return None, None, None, durum
    esit = esit_dagitim(girdi, butce, park_kesme, park_mm)
    plan_e, esit_e = etki(plan), etki(esit)
    fayda_esit = esit_e["agirlikli_fayda"].sum()
    artislar = []
    for c in X0_ARALIK:
        p2, d2 = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt, x0_carpan=c, **ayar)
        fz = etki(esit, c)["agirlikli_fayda"].sum()
        if d2 == "Optimal" and fz > 0:
            artislar.append(etki(p2, c)["agirlikli_fayda"].sum() / fz - 1)
    yalniz_park = None
    if set(turler) != {"park"}:
        pp, dp = optimize_et(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt,
                             turler=("park",), park_kesme=park_kesme, park_gunluk_mm=park_mm)
        if dp == "Optimal" and fayda_esit > 1e-9:
            yalniz_park = etki(pp)["agirlikli_fayda"].sum() / fayda_esit - 1
    if artislar and fayda_esit > 1e-9:
        nominal = plan_e["agirlikli_fayda"].sum() / fayda_esit - 1
        artislar.append(nominal)
    sulanan = plan_e[plan_e["su_m3"] > 0.001]
    ozet = {
        "isi_stresi_var": fayda_esit > 1e-9,
        "artis": plan_e["agirlikli_fayda"].sum() / fayda_esit - 1 if fayda_esit > 1e-9 else 0.0,
        "artis_aralik": (min(artislar), max(artislar)) if artislar else None,
        "yalniz_park": yalniz_park,
        "kds": float(plan_e["kisi_derece_saat"].sum()), "kds_esit": float(esit_e["kisi_derece_saat"].sum()),
        "dusus_aralik": (float(sulanan["utci_dusus_c"].min()), float(sulanan["utci_dusus_c"].max()))
        if len(sulanan) else None,
        "su_tur": plan_e.groupby("tur")["su_m3"].sum().to_dict(),
    }
    return plan_e, esit_e, ozet, durum


def operator_komutlari(sonuc, parklar, aritilmis_senaryo):
    p = parklar.set_index("park")
    satirlar = []
    for park, g in sonuc[sonuc["su_m3"] > 0.001].sort_values("saat").groupby("park"):
        miktar = g.set_index("saat")["su_m3"]
        saatler = list(miktar.index)
        tur = p.loc[park, "tur"]
        if tur == "sisleme":
            kaynak = "İçme suyu (zorunlu, lejyonella riski)"
        elif aritilmis_senaryo and p.loc[park, "aritilmis_su_erisimi"]:
            kaynak = "Arıtılmış su (senaryo)" + (", arazöz" if tur == "islatma" else "")
        else:
            kaynak = "Arazöz, mevcut kaynak" if tur == "islatma" else "Mevcut kaynak"
        bas = onceki = saatler[0]
        for sa in saatler[1:] + [None]:
            if sa is None or sa != onceki + 1:
                satirlar.append({"Nokta": park, "Yöntem": YONTEMLER[tur][0], "Başlangıç": f"{bas:02d}:00",
                                 "Bitiş": f"{onceki + 1:02d}:00",
                                 "Su (m³)": round(float(miktar.loc[bas:onceki].sum()), 2), "Kaynak": kaynak})
                bas = sa
            onceki = sa if sa is not None else onceki
    return pd.DataFrame(satirlar).sort_values(["Başlangıç", "Nokta"]) if satirlar else pd.DataFrame()


# ---------------------------------------------------------------------------
# HARİTA: düzgün (yumuşatılmış) yüzey görüntüsü
# Altıgen başına ORTALAMA değer Python'da hesaplanır, sabit renk tablosuyla boyanır ve tek bir görüntü
# olarak çizilir. Renkler yakınlaştırmayla değişmez; lejant aynı tablodan üretilir.
# ---------------------------------------------------------------------------
SICAK_DURAKLAR = [(0.00, (255, 255, 178, 0)), (0.15, (254, 204, 92, 120)), (0.40, (253, 141, 60, 170)),
                  (0.70, (240, 59, 32, 205)), (1.00, (189, 0, 38, 230))]
SOGUK_DURAKLAR = [(0.00, (198, 219, 239, 0)), (0.08, (158, 202, 225, 150)), (0.30, (107, 174, 214, 190)),
                  (0.60, (49, 130, 189, 215)), (1.00, (8, 81, 156, 235))]
SOGUMA_UST_C = E_REF  # soğuma ölçeği sabit: 0 → 1 °C UTCI düşüşü
PIKSEL_M = 60


@st.cache_data(show_spinner=False)
def izgara_hazirla(hexler: tuple):
    """Harita görüntüsünün her pikselini bir H3 altıgenine eşler (bir kez hesaplanır)."""
    cozunurluk = h3.get_resolution(hexler[0])
    merkez = np.array([h3.cell_to_latlng(h) for h in hexler])
    pay = 0.004
    lat0, lat1 = merkez[:, 0].min() - pay, merkez[:, 0].max() + pay
    lon0, lon1 = merkez[:, 1].min() - pay, merkez[:, 1].max() + pay
    yuk = int((lat1 - lat0) * 111_000 / PIKSEL_M)
    gen = int((lon1 - lon0) * 111_000 * np.cos(np.radians((lat0 + lat1) / 2)) / PIKSEL_M)
    lats = lat1 - (np.arange(yuk) + 0.5) * (lat1 - lat0) / yuk      # üst satır = kuzey
    lons = lon0 + (np.arange(gen) + 0.5) * (lon1 - lon0) / gen
    sira = {h: i for i, h in enumerate(hexler)}
    idx = np.array([[sira.get(h3.latlng_to_cell(la, lo, cozunurluk), -1) for lo in lons] for la in lats])
    return idx, [lon0, lat0, lon1, lat1]


def _kutu_bulanik(a, r):
    if r <= 0:
        return a
    k = 2 * r + 1
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(p, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def renklendir(t, duraklar, alfa_carpan=1.0):
    konum = np.array([d[0] for d in duraklar])
    renk = np.array([d[1] for d in duraklar], dtype=float)
    rgba = np.stack([np.interp(t, konum, renk[:, i]) for i in range(4)], axis=-1)
    rgba[..., 3] *= alfa_carpan
    rgba[t < 0.03, 3] = 0
    return rgba.astype(np.uint8)


def yuzey_goruntusu(t_hex, izgara, duraklar, alfa_carpan=1.0, bulaniklik=2):
    """t_hex: altıgen başına 0–1 yoğunluk. Görüntü veri URL'si (PNG) döner."""
    idx, _ = izgara
    t = np.where(idx >= 0, np.asarray(t_hex)[np.clip(idx, 0, None)], 0.0)
    t = _kutu_bulanik(_kutu_bulanik(t, bulaniklik), bulaniklik)  # iki kez kutu bulanıklık ≈ Gauss; altıgen kenarları görünmez
    gorsel = Image.fromarray(renklendir(np.clip(t, 0, 1), duraklar, alfa_carpan), "RGBA")
    tampon = io.BytesIO()
    gorsel.save(tampon, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(tampon.getvalue()).decode()


def mesafe_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    f1, f2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((f2 - f1) / 2) ** 2 + np.cos(f1) * np.cos(f2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def mahalle_sogumasi(yuzey, parklar, plan, saat):
    """Her altıgen için o saatteki tahmini UTCI düşüşü: noktadaki düşüş × e^(−mesafe/ölçek), kesme mesafesinden sonra 0.
    Ölçek yönteme göre: park 150/500 m, ıslatma 60/200 m, sisleme 15/50 m (varsayım)."""
    sog = np.zeros(len(yuzey))
    if plan is None:
        return sog
    h = plan[plan["saat"] == saat].set_index("park")
    for _, p in parklar.iterrows():
        dusus = float(h["utci_dusus_c"].get(p["park"], 0.0))
        if dusus <= 0:
            continue
        _, _, olcek, kesme = YONTEMLER[p["tur"]]
        d = mesafe_m(p["lat"], p["lon"], yuzey["lat"].values, yuzey["lon"].values)
        sog += np.where(d <= kesme, dusus * np.exp(-d / olcek), 0.0)
    return sog


def harita_katmanlari(saat, hava, yuzey, parklar, plan, sicak_pay, mod, secenekler, tesisler):
    """mod: 'Önce' (mevcut ısı), 'Sonra' (plan uygulandıktan sonra), 'Soğuma' (planın yarattığı serinleme)."""
    katmanlar, ist = [], None
    satir = hava[hava["saat"] == saat].iloc[0]
    if yuzey is not None:
        izgara = izgara_hazirla(tuple(yuzey["hex"]))
        utci_once = yuzey_utci(yuzey, satir)
        sog = mahalle_sogumasi(yuzey, parklar, plan, saat)
        utci_sonra = utci_once - sog
        esik = float(np.nanquantile(utci_once, 1 - sicak_pay))   # sıcak nokta eşiği "önce" durumundan sabit
        ust = float(max(np.nanquantile(utci_once, 0.99), esik + 0.5))
        alan = h3.average_hexagon_area(h3.get_resolution(yuzey["hex"].iloc[0]), unit="km^2")
        ist = {"once": int(np.sum(utci_once > esik)), "sonra": int(np.sum(utci_sonra > esik)),
               "alan_km2": float(np.sum(sog >= 0.1) * alan), "azami": float(sog.max()) if len(sog) else 0.0,
               "esik": esik, "ust": ust}
        if mod == "Soğuma":
            t = np.nan_to_num(sog / SOGUMA_UST_C)
            duraklar, alfa, bul = SOGUK_DURAKLAR, 1.0, 1
        else:
            deger = utci_sonra if mod == "Sonra" else utci_once
            t = np.nan_to_num(np.clip((deger - esik) / (ust - esik), 0, 1) * (deger > esik))
            duraklar, bul = SICAK_DURAKLAR, 2
            alfa = float(np.clip((kent_utci(satir) - 26) / 20, 0.35, 1.0))  # sabah/akşam sıcak noktalar soluklaşır
        katmanlar.append(pdk.Layer("BitmapLayer", data=None, id="yuzey",
                                   image=pdk.types.String(yuzey_goruntusu(t, izgara, duraklar, alfa, bul)),
                                   bounds=izgara[1]))
    p = parklar[["park", "ilce", "lat", "lon", "tur"]].copy()
    if plan is not None:
        h = plan[plan["saat"] == saat].set_index("park")
        p["su"] = p["park"].map(h["su_m3"]).fillna(0)
        p["once"] = p["park"].map(h["utci_c"])
        p["sonra"] = p["park"].map(h["utci_sonra_c"])
        p["yaricap"] = np.where(p["tur"] == "park", np.sqrt(p["su"]) * 20 + 45, 45)
        p["renk"] = [list(YONTEMLER[t][1]) + [235] if s > 0 else [140, 140, 140, 200] for t, s in zip(p["tur"], p["su"])]
        p["bilgi"] = [f"{YONTEMLER[t][0]} · {s:.2f} m³ · UTCI {a:.1f} → {b:.1f} °C"
                      for t, s, a, b in zip(p["tur"], p["su"], p["once"], p["sonra"])]
    else:
        p["yaricap"], p["renk"], p["bilgi"] = 90, [[140, 140, 140, 200]] * len(p), ""
    p["sekil"] = [[255, 255, 255] if t == "park" else list(YONTEMLER[t][1]) for t in p["tur"]]
    p["ad"] = p["park"]
    p["kisa"] = [a if t == "park" else ("S" if t == "sisleme" else "I") + a.split()[1]
                 for a, t in zip(p["park"], p["tur"])]
    if "Yürüme mesafesi (5 ve 10 dk)" in secenekler:
        pk = p[p["tur"] == "park"]
        halka = pd.concat([pk.assign(r=YURUME_5DK_M), pk.assign(r=YURUME_10DK_M)])
        katmanlar.append(pdk.Layer("ScatterplotLayer", halka, id="yurume", get_position="[lon, lat]",
                                   get_radius="r", filled=False, stroked=True, get_line_color=[150, 150, 150, 200],
                                   line_width_min_pixels=1.5))
    if "Hassas tesisler (OSM)" in secenekler and tesisler is not None and not tesisler.empty:
        t_df = tesisler.copy()
        t_df["renk"] = t_df["kategori"].map(lambda k: list(OSM_KATEGORILER[k][1]) + [230])
        t_df["bilgi"] = t_df["kategori"]
        katmanlar.append(pdk.Layer("ScatterplotLayer", t_df, id="tesis", get_position="[lon, lat]",
                                   get_radius=30, radius_min_pixels=3, get_fill_color="renk", pickable=True))
    katmanlar.append(pdk.Layer("ScatterplotLayer", p, id="parklar", get_position="[lon, lat]",
                               get_radius="yaricap", get_fill_color="renk", pickable=True,
                               radius_min_pixels=7, stroked=True, get_line_color="sekil", line_width_min_pixels=2))
    if "Nokta adları" in secenekler:
        katmanlar.append(pdk.Layer("TextLayer", p, id="etiket", get_position="[lon, lat]", get_text="kisa",
                                   get_size=13, get_color=[255, 255, 255], get_pixel_offset=[0, -24],
                                   character_set=pdk.types.String("auto"), font_weight=600, background=True,
                                   font_family=pdk.types.String("Helvetica, Arial, sans-serif"),
                                   get_background_color=[20, 20, 20, 190], background_padding=[4, 2]))
    deck = pdk.Deck(layers=katmanlar,
                    initial_view_state=pdk.ViewState(latitude=float(p["lat"].mean()), longitude=float(p["lon"].mean()),
                                                     zoom=12.2, pitch=0),
                    tooltip={"text": "{ad}\n{bilgi}"})
    return deck, ist


def gradyan(duraklar, genislik=150):
    renkler = ", ".join(f"rgba({r},{g},{b},{max(a, 70) / 255:.2f}) {k * 100:.0f}%" for k, (r, g, b, a) in duraklar)
    return (f"<span style='display:inline-block;width:{genislik}px;height:10px;vertical-align:middle;"
            f"border-radius:2px;background:linear-gradient(to right, {renkler})'></span>")


def lejant_html(mod, ist, sicak_pay, tesis_var):
    parca = []
    if ist is not None and mod == "Soğuma":
        parca.append(f"<b>Soğuma etkisi</b> (tahmini UTCI düşüşü) &nbsp;0 °C {gradyan(SOGUK_DURAKLAR)} "
                     f"{SOGUMA_UST_C:.1f} °C")
    elif ist is not None:
        baslik = "Plan uygulanmış" if mod == "Sonra" else "Mevcut durum"
        parca.append(f"<b>{baslik}: sıcak noktalar</b> (kentin en sıcak %{sicak_pay * 100:.0f}'lik alanı) &nbsp;"
                     f"{ist['esik']:.1f} °C {gradyan(SICAK_DURAKLAR)} {ist['ust']:.1f} °C UTCI")
    else:
        parca.append("<b>Isı yüzeyi dosyası yok:</b> yalnızca parklar gösteriliyor")
    parca.append(" &nbsp;".join(f"<span style='color:rgb{YONTEMLER[k][1]}'>●</span> {YONTEMLER[k][0].lower()}"
                                for k in YONTEMLER) + " &nbsp;<span style='color:#8c8c8c'>●</span> bu saatte kapalı")
    if tesis_var:
        parca.append(" &nbsp;".join(f"<span style='color:rgb{OSM_KATEGORILER[k][1]}'>●</span> {k.lower()}"
                                    for k in OSM_KATEGORILER))
    return "<div style='font-size:0.85rem;line-height:1.9'>" + " &nbsp;·&nbsp; ".join(parca) + "</div>"


def isi_seridi_html(satir):
    u = kent_utci(satir)
    ad, (r, g, b) = utci_sinif(u)
    yazi = "#000" if (r * 0.299 + g * 0.587 + b * 0.114) > 150 else "#fff"
    return (f"<div style='background:rgb({r},{g},{b});color:{yazi};padding:6px 10px;border-radius:4px;"
            f"font-weight:600'>Kent geneli UTCI {u:.0f} °C · {ad.lower()} ısı stresi</div>")


def saat_karti(saat, hava, plan, vpd_min):
    satir = hava[hava["saat"] == saat].iloc[0]
    st.markdown(f"#### {saat:02d}:00")
    st.markdown(isi_seridi_html(satir), unsafe_allow_html=True)
    st.caption(f"Hava {satir['sicaklik_c']:.1f} °C · nem %{satir['bagil_nem']:.0f} · "
               f"rüzgâr {satir['ruzgar_ms']:.1f} m/s · güneş {satir['gunes_wm2']:.0f} W/m²")
    if plan is None:
        return
    h = plan[plan["saat"] == saat].sort_values("su_m3", ascending=False)
    sulanan = h[h["su_m3"] > 0.001]
    if sulanan.empty:
        if h["tehlike"].sum() == 0:
            st.write("Bu saatte güçlü ısı stresi yok; sulama gerekmiyor.")
        elif (h["vpd_kpa"] < vpd_min).all():
            st.write("Bu saatte sulama yok: hava nemli, su az serinletir (nem kapısı).")
        else:
            st.write("Bu saatte sulama yok: su daha verimli olacağı saatlere ayrıldı.")
    sulanan = sulanan.sort_values("agirlikli_fayda", ascending=False)
    for _, r in sulanan.head(6).iterrows():
        st.markdown(f"**{r['park']}** · {YONTEMLER[r['tur']][0].lower()} · {r['su_m3']:.2g} m³  \n"
                    f"UTCI {r['utci_c']:.1f} → {r['utci_sonra_c']:.1f} °C · "
                    f"<span style='color:gray'>{utci_sinif(r['utci_c'])[0].lower()} stres, VPD {r['vpd_kpa']:.1f} kPa</span>",
                    unsafe_allow_html=True)
    if len(sulanan) > 6:
        st.caption(f"ve {len(sulanan) - 6} nokta daha (tam liste aşağıdaki onay listesinde)")


def plan_matrisi(plan):
    d = plan[["park", "saat", "su_m3", "utci_dusus_c"]].copy()
    d["saat_et"] = d["saat"].map(lambda s: f"{s:02d}:00")
    d["etiket"] = d["su_m3"].map(lambda v: f"{v:.0f}" if v >= 0.5 else ("•" if v > 0.001 else ""))
    taban = alt.Chart(d).encode(x=alt.X("saat_et:O", title="Saat"), y=alt.Y("park:N", title=None))
    hucre = taban.mark_rect().encode(
        color=alt.Color("su_m3:Q", title="Su (m³)", scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("park:N", title="Nokta"), alt.Tooltip("saat_et:O", title="Saat"),
                 alt.Tooltip("su_m3:Q", title="Su (m³)", format=".2f"),
                 alt.Tooltip("utci_dusus_c:Q", title="UTCI düşüşü (°C)", format=".2f")])
    yazi = taban.mark_text(fontSize=11).encode(
        text="etiket:N", color=alt.condition("datum.su_m3 > 8", alt.value("white"), alt.value("black")))
    return (hucre + yazi).properties(height=40 * d["park"].nunique() + 30)


def park_ozet_tablosu(parklar, plan, tesisler):  # noqa: tüm noktalar
    satirlar = []
    for _, p in parklar.iterrows():
        g = plan[plan["park"] == p["park"]]
        saatler = g.loc[g["su_m3"] > 0.001, "saat"]
        r = {"Nokta": p["park"], "Yöntem": YONTEMLER[p["tur"]][0], "İlçe": p["ilce"],
             "Günlük su (m³)": round(float(g["su_m3"].sum()), 2),
             "Çalışma saatleri": f"{saatler.min():02d}:00–{saatler.max() + 1:02d}:00" if len(saatler) else "–",
             "En yüksek UTCI (°C)": round(float(g["utci_c"].max()), 1),
             "Erişim (kişi/saat, temsili)": int(g["kisi"].iloc[0]),
             "Kişi·°C·saat": round(float(g["kisi_derece_saat"].sum()), 0)}
        if tesisler is not None:
            d = mesafe_m(p["lat"], p["lon"], tesisler["lat"].values, tesisler["lon"].values) <= YURUME_5DK_M
            for k in ("Okul / kreş", "Sağlık", "Huzurevi"):
                r[f"{k} (400 m)"] = int(((tesisler["kategori"] == k).values & d).sum())
        satirlar.append(r)
    return pd.DataFrame(satirlar).sort_values("Günlük su (m³)", ascending=False)


# ---------------------------------------------------------------------------
# ARAYÜZ
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CoolScape AI", layout="wide", initial_sidebar_state="expanded")
for anahtar, deger in {"saat": 14, "oynuyor": False}.items():
    st.session_state.setdefault(anahtar, deger)

st.title("CoolScape AI")
st.markdown("Kentsel serinlik ve su tahsis motoru · Antalya Muratpaşa–Kepez pilotu")

# --- Girdiler: görünürde yalnızca üç karar, gerisi gelişmiş ayarlarda ---
st.sidebar.header("Günün planı")
gun = st.sidebar.selectbox("Gün", list(GUNLER))
butce = st.sidebar.number_input("Günlük su bütçesi (m³)", value=300, step=25, min_value=25,
                                help="08:00–20:00 arasında parklara, sisleme noktalarına ve yol ıslatmaya dağıtılacak toplam su.")
on_ayar = st.sidebar.radio("Öncelik", list(ON_AYARLAR), help="Hazır ayar. Ayrıntıları aşağıdan değiştirebilirsiniz.")
v = ON_AYARLAR[on_ayar]
YONTEM_ETIKET = {v_[0]: k for k, v_ in YONTEMLER.items()}
secili_yontem = st.sidebar.multiselect("Yöntemler", list(YONTEM_ETIKET), default=list(YONTEM_ETIKET),
                                       help="Sisleme ve yol ıslatma noktaları ısı haritasındaki sıcak noktalardan "
                                            "otomatik önerilen adaylardır.")
turler = tuple(sorted({YONTEM_ETIKET[x] for x in secili_yontem} | {"park"}))
with st.sidebar.expander("Gelişmiş ayarlar"):
    kirilgan_pay = st.slider("Kırılgan bölgelere asgari su payı (%)", 0, 80, v["kirilgan_pay"],
                             key=f"pay_{on_ayar}") / 100
    w_kirilganlik = st.slider("Ağırlık: kırılganlık (65+ ve 0–4 yaş)", 0.0, 2.0, v["w_k"], 0.1, key=f"wk_{on_ayar}")
    vpd_min = st.slider("Nem kapısı: en düşük VPD (kPa)", 0.0, 2.0, 0.8, 0.1,
                        help="Buhar basıncı açığı bunun altındaysa hava nemlidir, su az serinletir; o saatte sulama yapılmaz.")
    enerji_opt = st.checkbox("Pompalama enerjisini gözet", value=True)
    st.markdown("**Çim güvenliği (park sulama)**")
    park_kesme = st.slider("Akşam kesme saati", 14, 20, 18, format="%d:00",
                           help="Bu saatten sonra park sulanmaz: yaprak gece ıslak kalırsa mantar hastalığı riski artar.")
    park_mm = st.slider("Park başına günlük üst sınır (mm = L/m²)", 2.0, 12.0, 6.0, 0.5,
                        help="Çimin sıcak bir yaz gününde tükettiği suya yakın bir sınır; aşırı sulamayı önler.")
    st.markdown("**Erişim varsayımları (kişi/saat, temsili)**")
    erisim = {"park_carpan": st.slider("Park: nüfus skoru başına kişi", 10, 100, ERISIM_VARSAYILAN["park_carpan"], 5),
              "sisleme": st.slider("Sisleme noktası", 5, 200, ERISIM_VARSAYILAN["sisleme"], 5),
              "islatma": st.slider("Islatılan cadde", 20, 500, ERISIM_VARSAYILAN["islatma"], 10)}
    aritilmis_senaryo = st.checkbox("Arıtılmış su senaryosu (gelecek)", value=False,
                                    help="Altyapı kurulursa hangi parkların arıtılmış suyla beslenebileceğini gösterir.")
    sicak_pay = st.slider("Haritada gösterilecek en sıcak alan (%)", 5, 50, 20, 5) / 100

# --- Veri: kendiliğinden yüklenir ---
tur, tarih = GUNLER[gun]
if tur == "tahmin":
    tarih = (datetime.now(ZoneInfo("Europe/Istanbul")).date() + timedelta(days=1)).isoformat()
try:
    hava = hava_tahmini_cek(tarih) if tur == "tahmin" else hava_verisi_cek(tarih)
    hava_kaynak = f"Open-Meteo tahmini ({tarih})" if tur == "tahmin" else "Open-Meteo arşivi (ERA5 tabanlı)"
except Exception as hata:
    st.error(f"Open-Meteo'ya erişilemedi ({hata}). Uydurma veriyle devam edilmez. Saatlik veriyi CSV olarak "
             "yükleyebilirsiniz (sütunlar: saat, sicaklik_c, bagil_nem, ruzgar_ms, gunes_wm2).")
    yuklenen = st.file_uploader("Saatlik hava verisi (CSV)", type="csv")
    if yuklenen is None:
        st.stop()
    hava = csv_hava_verisi(yuklenen)
    hava_kaynak = "yüklenen CSV"
if not set(SAATLER) <= set(hava["saat"]):
    st.error("Hava verisinde 08:00–20:00 saatlerinin bir kısmı eksik.")
    st.stop()

yuzey = isi_yuzeyi_oku(ISI_YUZEYI_CSV, os.path.getmtime(ISI_YUZEYI_CSV) if os.path.exists(ISI_YUZEYI_CSV) else 0)
tesisler = tesis_dosyasi_oku(TESIS_CSV, os.path.getmtime(TESIS_CSV) if os.path.exists(TESIS_CSV) else 0)
if tesisler is not None:
    osm_hata, osm_kaynak = None, f"OpenStreetMap ({TESIS_CSV}), {len(tesisler)} tesis"
else:
    tesisler, osm_hata = osm_tesisleri_cek()
    osm_kaynak = (f"OpenStreetMap (canlı), {len(tesisler)} tesis" if tesisler is not None
                  else f"OpenStreetMap'e erişilemedi; {TESIS_CSV} dosyasını repoya ekleyin")
parklar_g = parklari_guncelle(PARKLAR, yuzey)
adaylar = aday_noktalar(yuzey, parklar_g)
parklar = pd.concat([parklar_g, adaylar], ignore_index=True)   # tüm müdahale noktaları
tablo = park_saat_tablosu(hava, parklar)
girdi = girdileri_hazirla(tablo, parklar, w_kirilganlik, erisim)
kirilgan_esik = parklar_g["hassas_yuzde"].median()
kirilgan_parklar = tuple(sorted(parklar.loc[parklar["hassas_yuzde"] >= kirilgan_esik, "park"]))

if yuzey is not None:
    tarih_bilgi = yuzey["landsat_tarih"].iloc[0] if "landsat_tarih" in yuzey else "?"
    yuzey_kaynak = f"Landsat yüzey sıcaklığı ({tarih_bilgi}, sabit taban harita)"
else:
    yuzey_kaynak = f"yok ({ISI_YUZEYI_CSV} bulunamadı)"
st.caption(f"Hava: {hava_kaynak} · Isı yüzeyi: {yuzey_kaynak} · Tesisler: {osm_kaynak} · "
           "Park nüfusu, kırılganlık, hat kapasitesi, sulanan alan ve erişim temsili · Etki eğrileri Paris, Melbourne, "
           "Adelaide ve Arizona saha ölçümlerine göre ölçeklendi; Antalya saha doğrulaması 2027 pilotunda.")

# --- Plan: ayar değişince kendiliğinden hesaplanır ---
plan, esit, ozet, durum = plan_hesapla(girdi, butce, kirilgan_parklar, kirilgan_pay, vpd_min, enerji_opt,
                                       turler, park_kesme, park_mm)
if plan is None:
    st.error(f"Plan bulunamadı (çözücü durumu: {durum}). Gelişmiş ayarlarda kırılgan payı veya nem kapısını gevşetin.")
    st.stop()

if not ozet["isi_stresi_var"]:
    st.info("Bu gün hiçbir saatte güçlü ısı stresi (UTCI ≥ 32 °C) beklenmiyor; soğutma sulaması önerilmiyor.")
else:
    pay = lambda d: d.loc[d["park"].isin(kirilgan_parklar), "su_m3"].sum() / max(d["su_m3"].sum(), 1e-9) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Serinlik kazancı", f"{ozet['artis'] * 100:+.0f}%", "gündüz eşit park sulamasına göre", delta_color="off",
              help="Öncelik ağırlıklı UTCI düşüşü, aynı su miktarıyla. Model simülasyonudur, saha ölçümü değildir.")
    alt_satir = []
    if ozet["artis_aralik"]:
        a, b = ozet["artis_aralik"]
        alt_satir.append(f"Belirsizlik aralığı {a * 100:+.0f}% … {b * 100:+.0f}%")
    if ozet["yalniz_park"] is not None:
        alt_satir.append(f"Yalnız park sulamasıyla {ozet['yalniz_park'] * 100:+.0f}%"
                         + (" (kırılgan pay kuralı suyu kısıtlıyor)" if ozet["yalniz_park"] < 0 else ""))
    if alt_satir:
        m1.caption(" · ".join(alt_satir))
    m2.metric("Serinlik (kişi·°C·saat)", f"{ozet['kds']:,.0f}".replace(",", "."),
              f"eşit dağıtım {ozet['kds_esit']:,.0f}".replace(",", "."), delta_color="off",
              help="Her saat, ulaşılan kişi sayısı × UTCI düşüşü toplamı (yalnızca güçlü ısı stresi saatleri). "
                   "Erişim sayıları temsilidir.")
    m3.metric("Kullanılan su", f"{plan['su_m3'].sum():.0f} m³", f"bütçe {butce} m³", delta_color="off")
    su_tur = ozet["su_tur"]
    m3.caption(" · ".join(f"{YONTEMLER[k][0]} {v_:.1f} m³" for k, v_ in su_tur.items() if v_ > 0.001))
    m4.metric("Kırılgan bölgelere giden pay", f"%{pay(plan):.0f}", f"{pay(plan) - pay(esit):+.0f} puan")
    if ozet["dusus_aralik"]:
        a, b = ozet["dusus_aralik"]
        m4.caption(f"Noktada UTCI düşüşü {a:.1f}–{b:.1f} °C")
    kullanilan = float(plan["su_m3"].sum())
    if butce - kullanilan > max(5.0, 0.05 * butce):
        neden = ("kırılgan bölgelere asgari pay kuralı ve çim güvenliği sınırları"
                 if kirilgan_pay > 0 else "çim güvenliği sınırları ve hat kapasiteleri")
        st.info(f"Bütçenin {butce - kullanilan:.0f} m³'ü kullanılmadı: {neden} nedeniyle daha fazla su faydalı biçimde "
                "dağıtılamıyor. Daha fazla su için gelişmiş ayarlarda bu sınırları gevşetebilir ya da yeni nokta "
                "ekleyebilirsiniz.")
    st.caption("Bugünkü yaygın uygulama gece sulamasıdır; gündüz ısı stresine doğrudan katkısı yok kabul edilir. "
               "Bu yüzden karşılaştırma, daha zor bir referans olan gündüz eşit park sulamasıyla yapılır; bu senaryo da "
               "aynı hat kapasitesi ve çim güvenliği kurallarına uyar.")

# --- Harita ve saat paneli ---
st.subheader("Isı haritası ve saatlik plan")
GORUNUMLER = {"Önce: mevcut ısı": "Önce", "Sonra: planla": "Sonra", "Soğuma etkisi": "Soğuma"}
KATMAN_SECENEKLERI = ["Nokta adları", "Yürüme mesafesi (5 ve 10 dk)", "Hassas tesisler (OSM)"]


def oynat_degistir():
    if st.session_state.oynuyor:
        st.session_state.oynuyor = False
    else:
        st.session_state.oynuyor = True
        st.session_state.oyun_yeni = True
        st.session_state.saat = SAATLER[0]


c_btn, c_gor, c_kat = st.columns([1, 2.2, 2.8], vertical_alignment="bottom")
c_btn.button("Durdur" if st.session_state.oynuyor else "Günü oynat", on_click=oynat_degistir,
             type="primary", key="oynat_dugmesi")
if yuzey is not None:
    mod = GORUNUMLER[c_gor.radio("Görünüm", list(GORUNUMLER), index=1, horizontal=True)]
else:
    mod = "Önce"
secenekler = c_kat.multiselect("Ek katmanlar", KATMAN_SECENEKLERI, default=KATMAN_SECENEKLERI[:2])


@st.fragment(run_every=OYNATMA_ADIM_S if st.session_state.oynuyor else None)
def zaman_paneli():
    # Saat, kaydırıcı çizilmeden ÖNCE ilerletilir: böylece kaydırıcı oynatmayla birlikte akar.
    if st.session_state.oynuyor:
        if not st.session_state.pop("oyun_yeni", False):
            st.session_state.saat = min(st.session_state.saat + 1, SAATLER[-1])
        if st.session_state.saat >= SAATLER[-1]:
            st.session_state.oynuyor = False
            st.rerun()  # zamanlayıcıyı kapatmak için tüm sayfayı yeniden çalıştır; son kare 20:00
    st.select_slider("Saat", options=SAATLER, key="saat", format_func=lambda s: f"{s:02d}:00",
                     label_visibility="collapsed")
    saat = st.session_state.saat
    deck, ist = harita_katmanlari(saat, hava, yuzey, parklar, plan, sicak_pay, mod, secenekler,
                                  tesisler if "Hassas tesisler (OSM)" in secenekler else None)
    st.markdown(lejant_html(mod, ist, sicak_pay, tesisler is not None and "Hassas tesisler (OSM)" in secenekler),
                unsafe_allow_html=True)
    c_harita, c_kart = st.columns([3, 1.25])
    c_harita.pydeck_chart(deck, height=540, key="harita")
    if ist is not None:
        if ist["alan_km2"] == 0:
            c_harita.caption(f"{saat:02d}:00 · Bu saatte sulama yok; mahallelerde soğuma etkisi yok.")
        else:
            c_harita.caption(
                f"{saat:02d}:00 · Planın soğuttuğu alan (≥0,1 °C): ~{ist['alan_km2']:.1f} km², en çok "
                f"{ist['azami']:.1f} °C · sıcak nokta alanı {ist['once']} → {ist['sonra']} altıgen. Çevreye yayılım "
                f"varsayımdır (park 150 m, ıslatma 60 m, sisleme 15 m ölçekle azalır).")
    with c_kart.container(border=True):
        saat_karti(saat, hava, plan, vpd_min)


zaman_paneli()
if "Hassas tesisler (OSM)" in secenekler and tesisler is None:
    st.caption(f"OpenStreetMap tesisleri yüklenemedi ({osm_hata}); katman gösterilmiyor.")

# --- Noktalar: neden bu nokta? ---
st.subheader("Müdahale noktaları")
st.dataframe(park_ozet_tablosu(parklar, plan, tesisler), hide_index=True)
st.caption("Sisleme ve ıslatma noktaları, ısı haritasının en sıcak altıgenlerinden yerleşim içinde kalan ve birbirinden "
           "en az 1 km uzak ADAY konumlardır. Kesin yer (durak, meydan, cadde) sahada belediyeyle seçilir.")
if tesisler is not None:
    st.caption("Tesis sayıları parkın 400 m'si (kuş uçuşu ~5 dk yürüme) içindedir; OpenStreetMap kaydı eksik olabilir. "
               "Bu sayılar şimdilik bilgi amaçlıdır, optimizasyona girmez.")

# --- Plan matrisi ve operatör onayı ---
st.subheader("Günün su planı (nokta × saat, m³; • = sisleme açık)")
st.altair_chart(plan_matrisi(plan))

st.subheader("Operatör onay listesi")
komut = operator_komutlari(plan, parklar, aritilmis_senaryo)
if komut.empty:
    st.write("Bu koşullarda sulama önerilmiyor.")
else:
    komut.insert(0, "Onay", False)
    st.data_editor(komut, hide_index=True, disabled=["Nokta", "Yöntem", "Başlangıç", "Bitiş", "Su (m³)", "Kaynak"])
    st.download_button("Planı CSV olarak indir", komut.drop(columns="Onay").to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"coolscape_plan_{tarih}.csv", mime="text/csv")

# --- Bütçe – serinlik eğrisi ---
with st.expander("Bütçe–serinlik eğrisi: ek su ne kazandırır?"):
    if st.button("Eğriyi hesapla"):
        satirlar = []
        for b in sorted({int(butce * k) for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)}):
            p_b, d_b = optimize_et(girdi, b, set(kirilgan_parklar), kirilgan_pay, vpd_min, enerji_opt,
                                   turler=turler, park_kesme=park_kesme, park_gunluk_mm=park_mm)
            if d_b == "Optimal":
                satirlar.append({"Bütçe (m³)": b, "Senaryo": "CoolScape", "Fayda": etki(p_b)["agirlikli_fayda"].sum()})
            satirlar.append({"Bütçe (m³)": b, "Senaryo": "Gündüz eşit park sulaması",
                             "Fayda": etki(esit_dagitim(girdi, b, park_kesme, park_mm))["agirlikli_fayda"].sum()})
        egri = pd.DataFrame(satirlar)
        st.altair_chart(alt.Chart(egri).mark_line(point=True).encode(
            x="Bütçe (m³):Q", y=alt.Y("Fayda:Q", title="Öncelik ağırlıklı serinlik"),
            color=alt.Color("Senaryo:N", scale=alt.Scale(range=["#1e88e5", "#999999"]))))
        st.caption("Eğri düzleştiği yerde ek su az ek serinlik getirir: bütçe kararını destekler.")

with st.expander("Neden? (teknik ayrıntı ve varsayımlar)"):
    s = st.session_state.saat
    d = plan[plan["saat"] == s][["park", "tur", "utci_c", "vpd_kpa", "kisi", "oncelik", "emax", "su_m3", "su_mm",
                                 "utci_dusus_c"]].round(2)
    d["tur"] = d["tur"].map(lambda t: YONTEMLER[t][0])
    d.columns = ["Nokta", "Yöntem", "UTCI (°C)", "VPD (kPa)", "Erişim (kişi)", "Öncelik", "Azami etki (°C)", "Su (m³)",
                 "Su (mm = L/m²)", "UTCI düşüşü (°C)"]
    st.markdown(f"**{s:02d}:00 saatinin hesap tablosu**")
    st.dataframe(d, hide_index=True)
    st.dataframe(parklar[["park", "yuzey_farki_c", "yuzey_kaynak", "yesil_skoru", "yesil_kaynak", "nufus_skoru", "hassas_yuzde"]]
                 .rename(columns={"park": "Park", "yuzey_farki_c": "Çevre yüzey farkı (°C)", "yuzey_kaynak": "Kaynak",
                                  "yesil_skoru": "Yeşil skoru", "yesil_kaynak": "Kaynak ",
                                  "nufus_skoru": "Nüfus skoru (temsili)", "hassas_yuzde": "Kırılgan % (temsili)"}),
                 hide_index=True)
    st.markdown(
        "- **Öncelik** = max(UTCI − 32, 0) × erişim (kişi) / 100 × (1 + kırılganlık ağırlığı × kırılgan % / 30)\n"
        "- **Yöntemler:** park sulama ve yol ıslatma azalan getirili eğriyle (aşağıda); sisleme aç/kapa, açıkken "
        f"{SISLEME_E} °C × min(VPD / 2,5 kPa, 1) UTCI düşüşü ve saatte {SISLEME_SU_M3_SAAT} m³ su (varsayım). "
        "Sisleme yalnızca içme suyuyla çalışır (lejyonella).\n"
        f"- **Çim güvenliği:** park sulaması {park_kesme}:00'dan sonra yapılmaz; park başına günlük en çok {park_mm} mm.\n"
        "- **Soğutma eğrisi:** UTCI düşüşü = azami etki × (1 − e^(−doz/x₀)). Doz, parkın sulanan alanına düşen su "
        "miktarıdır (mm = L/m²); x₀ = 0,5 mm/saat (duyarlılık 0,25–1,0 mm), yani ~1,5 mm/saatte etkinin %95'i alınır.\n"
        "- **Azami etki** 1,0 °C UTCI × min(VPD / 2,5 kPa, 1). Nemli saatlerde buharlaşma ve serinlik azalır. "
        "Belirsizlik bandı 0,5–2,0 °C.\n"
        "- **MILP:** öncelik ağırlıklı UTCI düşüşünü maksimize eder; kısıtlar su bütçesi, hat kapasitesi, en az doz, "
        "nem kapısı, kırılgan bölgelere asgari pay, akşam kesme saati ve günlük park üst sınırı.\n"
        "- **Karşılaştırma (gündüz eşit park sulaması):** aynı bütçe yalnızca parklara ve izin verilen gündüz saatlerine "
        "eşit bölünür, aynı kapasite ve çim güvenliği kurallarına uyar; "
        "nem, ısı stresi ve kırılganlık gözetilmez. Kazanç, aynı etki eğrileriyle hesaplanan bir simülasyondur.\n"
        "- **MRT** kaba yaklaşımdır (hava sıcaklığı + 0,02 × güneş + yüzey sıcaklığı farkı); gölge ve binalar hesaba katılmaz.\n"
        "- **Harita:** altıgen başına ortalama UTCI hesaplanır, sabit renk tablosuyla boyanır ve görsel olarak yumuşatılır. "
        "Landsat yerel sabah geçer; harita öğleden sonraki pik sıcaklığı değil, yüzeyin göreli ısınma desenini gösterir.\n"
        "- **UTCI sınıf eşikleri** resmîdir (26/32/38/46 °C); renkler bu uygulamanın seçimidir.\n"
        f"- **Yürüme halkaları** kuş uçuşu {YURUME_5DK_M} ve {YURUME_10DK_M} m'dir, yol ağına göre değildir."
    )
    st.markdown("**Etki eğrilerinin dayandığı saha ölçümleri**")
    st.dataframe(pd.DataFrame([
        ["Hendel vd. 2016, Urban Climate", "Paris, Louvre", "Kaldırım ıslatma", "−0,79 °C", "−1,03 °C", "Merkezi değer"],
        ["Hendel vd. 2016, Urban Climate", "Paris, Belleville", "Kaldırım ıslatma", "−0,60 °C", "−0,93 °C", "Merkezi değer"],
        ["Parison vd. 2020", "Paris (2013–2015)", "Kaldırım ıslatma", "−1,02 °C", "−1,93 °C", "Üst band"],
        ["Traill vd. 2024, Urban Climate", "Melbourne", "Kaldırım ıslatma (~1 mm)", "en çok −0,6 °C", "−0,2 … −2,0 °C", "Band"],
        ["Broadbent vd. 2018, Urban Climate", "Adelaide (model)", "Zemin/çim sulama", "günlük ort. en çok −2,3 °C",
         "–", "Doyma: ~20 L/m²/gün üzeri ek fayda yok"],
        ["Vanos vd. 2022, Int. J. Biometeorology", "Tempe, Arizona", "Sisleme (restoran terasları)", "–",
         "ort. −4,4 °C (gölgeyle −9,7 °C)", "Sisleme etkisi"],
    ], columns=["Çalışma", "Yer", "Uygulama", "Hava sıcaklığı", "UTCI", "Modelde kullanımı"]), hide_index=True)
    st.caption("Sınırlar: Paris ve Melbourne ölçümleri kaldırım ıslatmadır; park çim sulamasında etki farklı olabilir. "
               "Sisleme etkisi yalnızca birkaç metrelik alanda hissedilir ve Arizona'nın kuru koşulunda ölçülmüştür. "
               "Antalya'nın kuru-sıcak koşulu bu ölçümlerin çoğundan daha serttir. Sulanan alanlar temsilidir. "
               "Bu nedenle eğri yerel ölçümle doğrulanmalıdır (2027 saha pilotu).")
