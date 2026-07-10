import os
import sys
import glob
import time
import csv
import argparse
import threading
from datetime import datetime, timedelta

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import requests  # untuk notifikasi Telegram (pip install requests)
    REQUESTS_TERSEDIA = True
except ImportError:
    REQUESTS_TERSEDIA = False


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# Nilai inferensi (imgsz, conf) harus mencerminkan apa yang dilaporkan skripsi.
# Fitur opsional di bawah default NONAKTIF agar keluaran = output mentah model.
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # --- Parameter inferensi (samakan dengan skripsi) ---
    "conf_threshold": 0.5,        # Ultralytics default; sesuaikan dgn skripsi
    "imgsz": 480,                  # default training YOLOv8m; sesuaikan dgn skripsi

    # Nama kelas HARUS sama dengan label pada dataset training.
    # Jika di dataset memakai nama lain (mis. "person"), ganti di sini.
    "kelas_target": ["jerigen", "orang", "nozzle"],

    # Warna tetap per kelas target (BGR) agar konsisten antar frame/laporan.
    "warna_kelas": {
        "jerigen": (0, 140, 255),   # oranye
        "orang":   (255, 128, 0),   # biru muda
        "nozzle":  (0, 220, 0),     # hijau
    },
    "warna_default": (160, 160, 160),

    # --- Logika indikasi pengisian ilegal ---
    # Indikasi muncul bila JERIGEN + NOZZLE + ORANG terdeteksi pada frame yang
    # sama (syarat orang bisa dimatikan dengan --tanpa-orang).
    "indikasi_butuh_orang": True,
    # Jarak maksimum (px) antara pusat jerigen dan pusat nozzle agar dianggap
    # "berdekatan". Set None untuk menonaktifkan syarat jarak (cukup hadir
    # bersamaan dalam satu frame).
    "indikasi_jarak_maks": None,
    # Indikasi harus bertahan minimal N frame BERTURUT-TURUT sebelum notifikasi
    # dikirim — mencegah alarm palsu dari deteksi sesaat (1-2 frame).
    "min_frame_indikasi": 5,

    # --- NOTIFIKASI TELEGRAM (muncul di handphone) ---
    # Cara setup:
    #   1. Buka Telegram, cari @BotFather, kirim /newbot, ikuti instruksi
    #      → dapat TOKEN (contoh: "123456:ABC-xxxx")
    #   2. Cari bot barumu di Telegram, tekan START, kirim pesan apa saja
    #   3. Buka https://api.telegram.org/bot<TOKEN>/getUpdates di browser
    #      → cari "chat":{"id": 123456789} → itu CHAT_ID kamu
    # Token/chat_id bisa diisi di sini, lewat argumen CLI, atau environment
    # variable TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID (paling aman).
    "telegram_token": "",
    "telegram_chat_id": "",
    "notif_kirim_foto": True,      # kirim foto frame bersama pesan
    "notif_cooldown": 60,          # jeda minimal antar-notifikasi (detik)

    # --- Logging hasil deteksi ---
    "csv_file": "hasil_deteksi_spbu.csv",
    "interval_catat": 30,          # catat tiap N frame agar CSV tidak membludak
    "simpan_screenshot": False,    # screenshot otomatis saat ada indikasi
    "folder_screenshot": "screenshot_deteksi",

    # --- FITUR OPSIONAL (default OFF; nyalakan hanya jika ada di metodologi) ---
    "gunakan_preprocess": False,   # CLAHE + sharpen sebelum inferensi
    "gunakan_filter_ukuran": False,
    "min_box_lebar": 30, "min_box_tinggi": 30,
    "max_box_lebar": 300, "max_box_tinggi": 300,

    "gunakan_roi": False,
    "roi_x1": 100, "roi_y1": 100, "roi_x2": 700, "roi_y2": 600,
}


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMEN
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Deteksi jerigen, orang, dan nozzle di SPBU dengan YOLO "
                    "(video rekaman/kamera/gambar).")
    p.add_argument('--model', required=True,
                   help='Path model YOLO, mis. "my_model.pt"')
    p.add_argument('--source', required=True,
                   help='Sumber: file video/gambar, folder, "usb0", atau "picamera0"')
    p.add_argument('--thresh', type=float, default=CONFIG["conf_threshold"],
                   help=f'Confidence threshold (default {CONFIG["conf_threshold"]})')
    p.add_argument('--imgsz', type=int, default=CONFIG["imgsz"],
                   help=f'Ukuran inferensi (default {CONFIG["imgsz"]}). Samakan dgn skripsi.')
    p.add_argument('--resolution', default=None,
                   help='Resolusi tampilan WxH, mis. "1280x720"')
    p.add_argument('--record', action='store_true',
                   help='Rekam hasil deteksi ke output_deteksi.avi')
    p.add_argument('--no-display', action='store_true',
                   help='Jalankan tanpa jendela tampilan (untuk batch/headless)')
    p.add_argument('--tanpa-orang', action='store_true',
                   help='Indikasi cukup jerigen + nozzle saja (tanpa syarat orang)')
    p.add_argument('--jarak-maks', type=int, default=None,
                   help='[opsional] Jarak maks. (px) pusat jerigen–nozzle untuk indikasi')
    p.add_argument('--min-frame', type=int, default=CONFIG["min_frame_indikasi"],
                   help='Indikasi harus bertahan N frame berturut-turut sebelum '
                        f'notifikasi dikirim (default {CONFIG["min_frame_indikasi"]})')
    # Notifikasi Telegram
    p.add_argument('--telegram-token', default=None,
                   help='Token bot Telegram (atau set env TELEGRAM_TOKEN)')
    p.add_argument('--telegram-chat', default=None,
                   help='Chat ID Telegram tujuan (atau set env TELEGRAM_CHAT_ID)')
    p.add_argument('--notif-cooldown', type=int, default=CONFIG["notif_cooldown"],
                   help=f'Jeda minimal antar-notifikasi dlm detik (default {CONFIG["notif_cooldown"]})')
    p.add_argument('--tanpa-foto', action='store_true',
                   help='Kirim notifikasi teks saja tanpa foto frame')
    # Fitur opsional — hanya untuk yang dijelaskan di metodologi
    p.add_argument('--preprocess', action='store_true',
                   help='[opsional] Aktifkan CLAHE + sharpen sebelum inferensi')
    p.add_argument('--filter-size', action='store_true',
                   help='[opsional] Buang box di luar rentang ukuran tertentu')
    p.add_argument('--roi', action='store_true',
                   help='[opsional] Batasi deteksi pada area ROI')
    p.add_argument('--screenshot', action='store_true',
                   help='Simpan screenshot otomatis saat ada indikasi ilegal')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAS
# ─────────────────────────────────────────────────────────────────────────────
def inisialisasi_csv(nama_file):
    if not os.path.exists(nama_file):
        with open(nama_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                "No", "Tanggal", "Jam", "Nama_Objek", "Confidence_rata2(%)",
                "Jumlah_Terdeteksi", "Waktu_Video", "Lebar_Box(px)",
                "Tinggi_Box(px)", "Posisi_X", "Posisi_Y",
                "Indikasi_Ilegal", "Sumber",
            ])
        print(f"[INFO] File CSV dibuat: {nama_file}")


def catat_csv(nama_file, no, nama_obj, conf, jumlah, waktu_vid,
              lebar, tinggi, cx, cy, indikasi, sumber):
    now = datetime.now()
    with open(nama_file, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            no, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
            nama_obj, f"{conf:.1f}", jumlah, waktu_vid,
            lebar, tinggi, cx, cy,
            "YA" if indikasi else "TIDAK",
            os.path.basename(sumber),
        ])


def format_waktu(detik):
    if detik is None:
        return "-"
    td = timedelta(seconds=int(detik))
    h, r = divmod(int(td.total_seconds()), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def preprocess_frame(frame):
    """[OPSIONAL] CLAHE + sharpen. Mengubah input model — gunakan hanya bila
    dijelaskan di metodologi skripsi."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)


def gambar_roi(frame, x1, y1, x2, y2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(overlay, "ZONA DETEKSI", (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)


def lolos_filter_ukuran(lebar, tinggi):
    if not CONFIG["gunakan_filter_ukuran"]:
        return True
    terlalu_kecil = lebar < CONFIG["min_box_lebar"] or tinggi < CONFIG["min_box_tinggi"]
    terlalu_besar = lebar > CONFIG["max_box_lebar"] or tinggi > CONFIG["max_box_tinggi"]
    return not (terlalu_kecil or terlalu_besar)


def dalam_roi(cx, cy):
    if not CONFIG["gunakan_roi"]:
        return True
    return (CONFIG["roi_x1"] <= cx <= CONFIG["roi_x2"] and
            CONFIG["roi_y1"] <= cy <= CONFIG["roi_y2"])


def kelas_target_dari(classname):
    """Kembalikan nama kelas target yang cocok (mis. 'jerigen'), atau None."""
    nama = classname.lower()
    for k in CONFIG["kelas_target"]:
        if k.lower() in nama:
            return k
    return None


def warna_untuk(kelas_target, classidx):
    if kelas_target and kelas_target in CONFIG["warna_kelas"]:
        return CONFIG["warna_kelas"][kelas_target]
    return CONFIG["warna_default"]


def _kirim_telegram_worker(token, chat_id, pesan, frame_jpg):
    """Worker yang berjalan di thread terpisah agar inferensi tidak terhambat
    oleh koneksi internet yang lambat."""
    try:
        if frame_jpg is not None:
            url = f"https://api.telegram.org/bot{token}/sendPhoto"
            r = requests.post(
                url,
                data={"chat_id": chat_id, "caption": pesan, "parse_mode": "HTML"},
                files={"photo": ("deteksi.jpg", frame_jpg, "image/jpeg")},
                timeout=15)
        else:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            r = requests.post(
                url,
                data={"chat_id": chat_id, "text": pesan, "parse_mode": "HTML"},
                timeout=15)
        if r.status_code == 200:
            print("[NOTIF] Notifikasi Telegram terkirim.")
        else:
            print(f"[NOTIF] Gagal kirim (HTTP {r.status_code}): {r.text[:120]}")
    except Exception as e:
        print(f"[NOTIF] Gagal kirim notifikasi: {e}")


def kirim_notifikasi(frame, deteksi_frame, waktu_vid, sumber):
    """Susun pesan lalu kirim notifikasi Telegram secara asinkron."""
    token = CONFIG["telegram_token"]
    chat_id = CONFIG["telegram_chat_id"]
    if not token or not chat_id:
        return
    if not REQUESTS_TERSEDIA:
        print("[NOTIF] Modul 'requests' belum terpasang: pip install requests")
        return

    now = datetime.now()
    baris = []
    for k in CONFIG["kelas_target"]:
        if k in deteksi_frame:
            data = deteksi_frame[k]
            conf_avg = sum(d["conf"] for d in data) / len(data)
            baris.append(f"• {k.capitalize()}: {len(data)} (conf {conf_avg:.0f}%)")

    pesan = (
        "🚨 <b>INDIKASI PENGISIAN BBM ILEGAL</b>\n"
        f"🕐 {now.strftime('%d-%m-%Y %H:%M:%S')}\n"
        f"🎞 Waktu video: {format_waktu(waktu_vid)}\n"
        f"📹 Sumber: {os.path.basename(str(sumber))}\n"
        + "\n".join(baris) +
        "\n\nJerigen, nozzle, dan orang terdeteksi bersamaan di area SPBU."
    )

    frame_jpg = None
    if CONFIG["notif_kirim_foto"] and frame is not None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            frame_jpg = buf.tobytes()

    threading.Thread(
        target=_kirim_telegram_worker,
        args=(token, chat_id, pesan, frame_jpg),
        daemon=True).start()


def uji_koneksi_telegram():
    """Kirim pesan uji saat program mulai, agar tahu konfigurasi benar."""
    token, chat_id = CONFIG["telegram_token"], CONFIG["telegram_chat_id"]
    if not token or not chat_id:
        return False
    if not REQUESTS_TERSEDIA:
        print("[NOTIF] Modul 'requests' belum terpasang: pip install requests")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id,
                  "text": "✅ Sistem deteksi SPBU aktif. Notifikasi siap."},
            timeout=15)
        if r.status_code == 200:
            print("[NOTIF] Koneksi Telegram OK — pesan uji terkirim.")
            return True
        print(f"[NOTIF] Uji koneksi gagal (HTTP {r.status_code}): {r.text[:120]}")
    except Exception as e:
        print(f"[NOTIF] Uji koneksi gagal: {e}")
    return False


def cek_indikasi_ilegal(deteksi_frame):
    """Indikasi pengisian ilegal:
    - jerigen DAN nozzle hadir pada frame yang sama;
    - opsional: orang juga harus hadir (CONFIG['indikasi_butuh_orang']);
    - opsional: jarak pusat jerigen–nozzle <= indikasi_jarak_maks (px).
    """
    ada_jerigen = "jerigen" in deteksi_frame
    ada_nozzle = "nozzle" in deteksi_frame
    ada_orang = "orang" in deteksi_frame

    if not (ada_jerigen and ada_nozzle):
        return False
    if CONFIG["indikasi_butuh_orang"] and not ada_orang:
        return False

    jarak_maks = CONFIG["indikasi_jarak_maks"]
    if jarak_maks is None:
        return True

    for j in deteksi_frame["jerigen"]:
        for n in deteksi_frame["nozzle"]:
            jarak = np.hypot(j["cx"] - n["cx"], j["cy"] - n["cy"])
            if jarak <= jarak_maks:
                return True
    return False


def tentukan_sumber(src):
    img_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    vid_ext = ('.avi', '.mov', '.mp4', '.mkv', '.wmv')
    if os.path.isdir(src):
        return 'folder'
    if os.path.isfile(src):
        ext = os.path.splitext(src)[1].lower()
        if ext in img_ext:
            return 'image'
        if ext in vid_ext:
            return 'video'
        print(f"[ERROR] Format file '{ext}' tidak didukung.")
        sys.exit(1)
    if src.startswith('usb'):
        return 'usb'
    if src.startswith('picamera'):
        return 'picamera'
    print(f"[ERROR] Sumber input tidak valid: {src}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# PROGRAM UTAMA
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Terapkan flag ke CONFIG
    CONFIG["conf_threshold"] = args.thresh
    CONFIG["imgsz"] = args.imgsz
    CONFIG["gunakan_preprocess"] = args.preprocess
    CONFIG["gunakan_filter_ukuran"] = args.filter_size
    CONFIG["gunakan_roi"] = args.roi
    CONFIG["simpan_screenshot"] = args.screenshot
    CONFIG["indikasi_butuh_orang"] = not args.tanpa_orang
    CONFIG["min_frame_indikasi"] = args.min_frame
    if args.jarak_maks is not None:
        CONFIG["indikasi_jarak_maks"] = args.jarak_maks

    # Konfigurasi notifikasi: prioritas CLI > environment variable > CONFIG
    CONFIG["telegram_token"] = (args.telegram_token
                                or os.environ.get("TELEGRAM_TOKEN")
                                or CONFIG["telegram_token"])
    CONFIG["telegram_chat_id"] = (args.telegram_chat
                                  or os.environ.get("TELEGRAM_CHAT_ID")
                                  or CONFIG["telegram_chat_id"])
    CONFIG["notif_cooldown"] = args.notif_cooldown
    if args.tanpa_foto:
        CONFIG["notif_kirim_foto"] = False
    notif_aktif = uji_koneksi_telegram()

    # --- Cek & muat model ---
    if not os.path.exists(args.model):
        print(f"[ERROR] Model tidak ditemukan: {args.model}")
        sys.exit(1)
    model = YOLO(args.model, task='detect')
    labels = model.names

    # Peringatkan bila ada kelas target yang tidak dikenal model
    nama_kelas_model = [v.lower() for v in labels.values()]
    for k in CONFIG["kelas_target"]:
        if not any(k.lower() in n for n in nama_kelas_model):
            print(f"[PERINGATAN] Kelas target '{k}' tidak ditemukan pada model. "
                  f"Kelas model: {list(labels.values())}")

    # --- Tipe sumber ---
    source_type = tentukan_sumber(args.source)

    resize = False
    resW = resH = 0
    if args.resolution:
        resize = True
        resW, resH = (int(x) for x in args.resolution.lower().split('x'))

    # --- Siapkan sumber ---
    cap = None
    imgs_list = []
    if source_type == 'image':
        imgs_list = [args.source]
    elif source_type == 'folder':
        for f in sorted(glob.glob(os.path.join(args.source, '*'))):
            if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                imgs_list.append(f)
        if not imgs_list:
            print(f"[ERROR] Tidak ada gambar pada folder: {args.source}")
            sys.exit(1)
    elif source_type in ('video', 'usb'):
        cap_arg = args.source if source_type == 'video' else int(args.source[3:])
        cap = cv2.VideoCapture(cap_arg)
        if not cap.isOpened():
            print(f"[ERROR] Tidak dapat membuka sumber: {args.source}")
            sys.exit(1)
        if args.resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
    elif source_type == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(
            main={"format": 'RGB888', "size": (resW or 1280, resH or 720)}))
        cap.start()

    # --- Recorder ---
    recorder = None
    if args.record:
        if source_type not in ('video', 'usb', 'picamera'):
            print("[ERROR] --record hanya untuk video/kamera.")
            sys.exit(1)
        if not args.resolution:
            print("[ERROR] Tentukan --resolution untuk recording.")
            sys.exit(1)
        fps_src = 30
        if source_type == 'video' and cap is not None:
            f = cap.get(cv2.CAP_PROP_FPS)
            fps_src = f if f and f > 0 else 30
        recorder = cv2.VideoWriter('output_deteksi.avi',
                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                   fps_src, (resW, resH))

    # --- CSV & screenshot ---
    inisialisasi_csv(CONFIG["csv_file"])
    if CONFIG["simpan_screenshot"]:
        os.makedirs(CONFIG["folder_screenshot"], exist_ok=True)

    # --- Cetak konfigurasi efektif (berguna untuk lampiran skripsi) ---
    print("=" * 60)
    print("  DETEKSI JERIGEN + ORANG + NOZZLE — SPBU (YOLO)")
    print(f"  Model               : {args.model}")
    print(f"  Kelas model         : {list(labels.values())}")
    print(f"  Kelas target        : {CONFIG['kelas_target']}")
    print(f"  Sumber              : {args.source} ({source_type})")
    print(f"  Conf threshold      : {CONFIG['conf_threshold']}")
    print(f"  Imgsz inferensi     : {CONFIG['imgsz']}")
    print(f"  Indikasi butuh orang: {CONFIG['indikasi_butuh_orang']}")
    print(f"  Jarak maks indikasi : {CONFIG['indikasi_jarak_maks']}")
    print(f"  Min. frame indikasi : {CONFIG['min_frame_indikasi']}")
    print(f"  Notifikasi Telegram : {'AKTIF' if notif_aktif else 'nonaktif'}")
    if notif_aktif:
        print(f"  Cooldown notifikasi : {CONFIG['notif_cooldown']} detik")
    print(f"  Preprocessing       : {CONFIG['gunakan_preprocess']}")
    print(f"  Filter ukuran box   : {CONFIG['gunakan_filter_ukuran']}")
    print(f"  ROI                 : {CONFIG['gunakan_roi']}")
    print("  Tekan 'Q' keluar | 'S' pause | 'P' screenshot manual")
    print("=" * 60)

    # --- Variabel kontrol ---
    frame_rate_buffer, fps_avg_len = [], 100
    avg_fps = 0.0
    img_count = frame_ke = screenshot_count = 0
    no_csv = 1
    frame_dicatat = -10**9
    total_indikasi = 0
    indikasi_beruntun = 0          # frame indikasi berturut-turut
    waktu_notif_terakhir = 0.0     # untuk cooldown notifikasi
    total_notif = 0

    # ── LOOP UTAMA ──────────────────────────────────────────────────────────
    while True:
        t_start = time.perf_counter()
        frame_ke += 1

        # Ambil frame
        if source_type in ('image', 'folder'):
            if img_count >= len(imgs_list):
                print("[INFO] Semua gambar selesai diproses.")
                break
            frame = cv2.imread(imgs_list[img_count])
            img_count += 1
            if frame is None:
                continue
        elif source_type in ('video', 'usb'):
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[INFO] Sumber video/kamera selesai.")
                break
        else:  # picamera
            frame = cap.capture_array()
            if frame is None:
                break

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # Input model: pakai preprocess hanya jika diaktifkan
        frame_infer = preprocess_frame(frame.copy()) if CONFIG["gunakan_preprocess"] else frame

        if CONFIG["gunakan_roi"]:
            frame = gambar_roi(frame, CONFIG["roi_x1"], CONFIG["roi_y1"],
                               CONFIG["roi_x2"], CONFIG["roi_y2"])

        # Inferensi
        results = model(frame_infer, verbose=False,
                        imgsz=CONFIG["imgsz"], conf=CONFIG["conf_threshold"])
        detections = results[0].boxes

        # deteksi_frame: {"jerigen": [...], "orang": [...], "nozzle": [...]}
        deteksi_frame = {}
        jumlah_per_kelas = {k: 0 for k in CONFIG["kelas_target"]}

        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().reshape(-1)
            xmin, ymin, xmax, ymax = xyxy[:4].astype(int)
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = float(detections[i].conf.item())

            lebar, tinggi = xmax - xmin, ymax - ymin
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

            if not lolos_filter_ukuran(lebar, tinggi):
                continue
            if not dalam_roi(cx, cy):
                continue

            kelas = kelas_target_dari(classname)
            color = warna_untuk(kelas, classidx)

            # Gambar bounding box + label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf * 100)}%'
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ytop = max(ymin, th + 10)
            cv2.rectangle(frame, (xmin, ytop - th - 10),
                          (xmin + tw, ytop + bl - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, ytop - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            if kelas is not None:
                jumlah_per_kelas[kelas] += 1
                deteksi_frame.setdefault(kelas, []).append(
                    {"conf": conf * 100, "lebar": lebar, "tinggi": tinggi,
                     "cx": cx, "cy": cy})

        # Cek indikasi pengisian ilegal (jerigen + nozzle + orang bersamaan)
        indikasi = cek_indikasi_ilegal(deteksi_frame)
        if indikasi:
            total_indikasi += 1
            indikasi_beruntun += 1
        else:
            indikasi_beruntun = 0

        # ── NOTIFIKASI KE HANDPHONE (Telegram) ──────────────────────────
        # Dikirim bila: indikasi bertahan >= min_frame_indikasi frame
        # berturut-turut DAN sudah lewat masa cooldown.
        if (notif_aktif
                and indikasi_beruntun >= CONFIG["min_frame_indikasi"]
                and (time.time() - waktu_notif_terakhir) >= CONFIG["notif_cooldown"]):
            waktu_vid_notif = None
            if source_type == 'video' and cap is not None:
                waktu_vid_notif = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            kirim_notifikasi(frame, deteksi_frame, waktu_vid_notif, args.source)
            waktu_notif_terakhir = time.time()
            total_notif += 1

        # Screenshot otomatis saat ada indikasi pengisian ilegal
        if (indikasi and CONFIG["simpan_screenshot"]
                and (frame_ke - frame_dicatat) >= CONFIG["interval_catat"]):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                os.path.join(CONFIG["folder_screenshot"],
                             f"indikasi_{ts}_{screenshot_count:04d}.jpg"), frame)
            screenshot_count += 1

        # Catat CSV (satu baris per kelas target yang terdeteksi)
        if deteksi_frame and (frame_ke - frame_dicatat) >= CONFIG["interval_catat"]:
            waktu_vid = None
            if source_type == 'video' and cap is not None:
                waktu_vid = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            for nama_kelas, data in deteksi_frame.items():
                n = len(data)
                conf_avg = sum(d["conf"] for d in data) / n
                catat_csv(CONFIG["csv_file"], no_csv, nama_kelas, conf_avg, n,
                          format_waktu(waktu_vid),
                          int(sum(d["lebar"] for d in data) / n),
                          int(sum(d["tinggi"] for d in data) / n),
                          int(sum(d["cx"] for d in data) / n),
                          int(sum(d["cy"] for d in data) / n),
                          indikasi, args.source)
                print(f"[{no_csv}] {nama_kelas} | {n} objek | "
                      f"conf {conf_avg:.1f}% | waktu {format_waktu(waktu_vid)}"
                      f"{' | INDIKASI ILEGAL' if indikasi else ''}")
                no_csv += 1
            frame_dicatat = frame_ke

        # Overlay info per kelas
        cv2.rectangle(frame, (0, 0), (320, 115), (0, 0, 0), cv2.FILLED)
        y = 22
        if source_type in ('video', 'usb', 'picamera'):
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 22
        for k in CONFIG["kelas_target"]:
            warna = CONFIG["warna_kelas"].get(k, (0, 255, 0))
            cv2.putText(frame, f'{k.capitalize()}: {jumlah_per_kelas[k]}',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2)
            y += 22
        cv2.putText(frame, f'Tercatat: {no_csv - 1}', (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Peringatan indikasi ilegal
        if indikasi:
            cv2.putText(frame, 'INDIKASI PENGISIAN ILEGAL',
                        (frame.shape[1] - 380, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        elif deteksi_frame:
            cv2.putText(frame, 'OBJEK TERDETEKSI',
                        (frame.shape[1] - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if recorder is not None:
            recorder.write(frame)

        if not args.no_display:
            cv2.imshow('Deteksi SPBU — Jerigen/Orang/Nozzle', frame)
            key = cv2.waitKey(0 if source_type in ('image', 'folder') else 5) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("\n[INFO] Dihentikan oleh pengguna.")
                break
            elif key in (ord('s'), ord('S')):
                cv2.waitKey(0)
            elif key in (ord('p'), ord('P')):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'capture_{ts}.jpg', frame)
                print(f"[INFO] Screenshot manual: capture_{ts}.jpg")

        # Hitung FPS
        dt = time.perf_counter() - t_start
        fps_now = 1.0 / dt if dt > 0 else 0.0
        frame_rate_buffer.append(fps_now)
        if len(frame_rate_buffer) > fps_avg_len:
            frame_rate_buffer.pop(0)
        avg_fps = float(np.mean(frame_rate_buffer))

    # ── LAPORAN AKHIR ────────────────────────────────────────────────────────
    print("=" * 60)
    print("  DETEKSI SELESAI")
    print(f"  Rata-rata FPS      : {avg_fps:.2f}")
    print(f"  Total frame        : {frame_ke}")
    print(f"  Frame ber-indikasi : {total_indikasi}")
    print(f"  Notifikasi terkirim: {total_notif}")
    print(f"  Baris CSV          : {no_csv - 1}")
    print(f"  File CSV           : {CONFIG['csv_file']}")
    if CONFIG["simpan_screenshot"]:
        print(f"  Screenshot         : {screenshot_count} gambar")
    print("=" * 60)

    if cap is not None:
        if source_type == 'picamera':
            cap.stop()
        else:
            cap.release()
    if recorder is not None:
        recorder.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
