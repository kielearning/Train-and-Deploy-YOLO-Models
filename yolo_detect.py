"""
================================================================================
Deteksi Aktivitas Pengisian BBM ke Jerigen di SPBU
Universitas Sam Ratulangi (UNSRAT) - Teknik Informatika

Model    : YOLOv8m (Ultralytics)  -  1 kelas: "jerigen" (index 0)
Sumber   : VIDEO REKAMAN (sesuai ruang lingkup skripsi), gambar, atau folder.
           Mode kamera (usb/picamera) disediakan opsional, TIDAK dipakai untuk
           menghasilkan hasil eksperimen pada skripsi.
================================================================================

CATATAN KONSISTENSI SKRIPSI (penting, baca sebelum mengubah parameter)
--------------------------------------------------------------------------------
Agar kode SESUAI dengan apa yang ditulis di Bab III/IV:

  * IMGSZ dan CONF di sini HARUS sama dengan nilai yang dilaporkan di skripsi.
    (default imgsz=640 mengikuti default training YOLOv8m; ubah hanya jika
     skripsi memang melaporkan nilai lain.)

  * Preprocessing (CLAHE/sharpen), filter ukuran bounding box, dan ROI adalah
    langkah TAMBAHAN yang MENGUBAH keluaran deteksi. Semuanya NONAKTIF secara
    default sehingga keluaran = output mentah model. Aktifkan HANYA bila
    langkah tersebut benar-benar dijelaskan di metodologi skripsi.

  * Metrik evaluasi (mAP, precision, recall, confusion matrix) untuk skripsi
    sebaiknya diambil dari `model.val()` pada data uji, BUKAN dari skrip demo
    ini. Skrip ini untuk demonstrasi/visualisasi deteksi pada video.
--------------------------------------------------------------------------------

Contoh pemakaian:
  python deteksi_jerigen.py --model my_model.pt --source video_uji.mp4
  python deteksi_jerigen.py --model my_model.pt --source video_uji.mp4 --resolution 1280x720 --record
  python deteksi_jerigen.py --model my_model.pt --source folder_gambar/
"""

import os
import sys
import glob
import time
import csv
import argparse
from datetime import datetime, timedelta

import cv2
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# Nilai inferensi (imgsz, conf) harus mencerminkan apa yang dilaporkan skripsi.
# Fitur opsional di bawah default NONAKTIF agar keluaran = output mentah model.
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # --- Parameter inferensi (samakan dengan skripsi) ---
    "conf_threshold": 0.25,        # Ultralytics default; sesuaikan dgn skripsi
    "imgsz": 640,                  # default training YOLOv8m; sesuaikan dgn skripsi
    "kelas_target": ["jerigen"],   # nama kelas dari label training

    # --- Logging hasil deteksi ---
    "csv_file": "hasil_deteksi_jerigen.csv",
    "interval_catat": 30,          # catat tiap N frame agar CSV tidak membludak
    "simpan_screenshot": False,    # screenshot otomatis saat ada deteksi
    "folder_screenshot": "screenshot_jerigen",

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
        description="Deteksi jerigen di SPBU dengan YOLOv8m (video rekaman).")
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
    # Fitur opsional — hanya untuk yang dijelaskan di metodologi
    p.add_argument('--preprocess', action='store_true',
                   help='[opsional] Aktifkan CLAHE + sharpen sebelum inferensi')
    p.add_argument('--filter-size', action='store_true',
                   help='[opsional] Buang box di luar rentang ukuran tertentu')
    p.add_argument('--roi', action='store_true',
                   help='[opsional] Batasi deteksi pada area ROI')
    p.add_argument('--screenshot', action='store_true',
                   help='Simpan screenshot otomatis saat jerigen terdeteksi')
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
                "Tinggi_Box(px)", "Posisi_X", "Posisi_Y", "Sumber",
            ])
        print(f"[INFO] File CSV dibuat: {nama_file}")


def catat_csv(nama_file, no, nama_obj, conf, jumlah, waktu_vid,
              lebar, tinggi, cx, cy, sumber):
    now = datetime.now()
    with open(nama_file, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            no, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
            nama_obj, f"{conf:.1f}", jumlah, waktu_vid,
            lebar, tinggi, cx, cy, os.path.basename(sumber),
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

    # Terapkan flag opsional ke CONFIG
    CONFIG["conf_threshold"] = args.thresh
    CONFIG["imgsz"] = args.imgsz
    CONFIG["gunakan_preprocess"] = args.preprocess
    CONFIG["gunakan_filter_ukuran"] = args.filter_size
    CONFIG["gunakan_roi"] = args.roi
    CONFIG["simpan_screenshot"] = args.screenshot

    # --- Cek & muat model ---
    if not os.path.exists(args.model):
        print(f"[ERROR] Model tidak ditemukan: {args.model}")
        sys.exit(1)
    model = YOLO(args.model, task='detect')
    labels = model.names

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
    print("  DETEKSI JERIGEN SPBU — YOLOv8m")
    print(f"  Model             : {args.model}")
    print(f"  Kelas model       : {list(labels.values())}")
    print(f"  Sumber            : {args.source} ({source_type})")
    print(f"  Conf threshold    : {CONFIG['conf_threshold']}")
    print(f"  Imgsz inferensi   : {CONFIG['imgsz']}")
    print(f"  Preprocessing     : {CONFIG['gunakan_preprocess']}")
    print(f"  Filter ukuran box : {CONFIG['gunakan_filter_ukuran']}")
    print(f"  ROI               : {CONFIG['gunakan_roi']}")
    print("  Tekan 'Q' keluar | 'P' screenshot manual")
    print("=" * 60)

    # --- Variabel kontrol ---
    bbox_colors = [(68, 148, 228), (88, 159, 106), (96, 202, 231),
                   (159, 124, 168), (98, 118, 150)]
    frame_rate_buffer, fps_avg_len = [], 100
    avg_fps = 0.0
    img_count = frame_ke = screenshot_count = 0
    no_csv = 1
    frame_dicatat = -10**9

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

        object_count = 0
        deteksi_frame = {}

        for i in range(len(detections)):
            xmin, ymin, xmax, ymax = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = float(detections[i].conf.item())

            lebar, tinggi = xmax - xmin, ymax - ymin
            cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2

            if not lolos_filter_ukuran(lebar, tinggi):
                continue
            if not dalam_roi(cx, cy):
                continue

            is_target = any(k.lower() in classname.lower()
                            for k in CONFIG["kelas_target"])

            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf * 100)}%'
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ytop = max(ymin, th + 10)
            cv2.rectangle(frame, (xmin, ytop - th - 10),
                          (xmin + tw, ytop + bl - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, ytop - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            object_count += 1
            if is_target:
                deteksi_frame.setdefault(classname, []).append(
                    {"conf": conf * 100, "lebar": lebar, "tinggi": tinggi,
                     "cx": cx, "cy": cy})

        # Screenshot otomatis
        if (deteksi_frame and CONFIG["simpan_screenshot"]
                and (frame_ke - frame_dicatat) >= CONFIG["interval_catat"]):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                os.path.join(CONFIG["folder_screenshot"],
                             f"jerigen_{ts}_{screenshot_count:04d}.jpg"), frame)
            screenshot_count += 1

        # Catat CSV
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
                          args.source)
                print(f"[{no_csv}] {nama_kelas} | {n} objek | "
                      f"conf {conf_avg:.1f}% | waktu {format_waktu(waktu_vid)}")
                no_csv += 1
            frame_dicatat = frame_ke

        # Overlay info
        cv2.rectangle(frame, (0, 0), (300, 70), (0, 0, 0), cv2.FILLED)
        if source_type in ('video', 'usb', 'picamera'):
            cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f'Jerigen: {object_count}', (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'Tercatat: {no_csv - 1}', (10, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if deteksi_frame:
            cv2.putText(frame, 'JERIGEN TERDETEKSI',
                        (frame.shape[1] - 250, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if recorder is not None:
            recorder.write(frame)

        if not args.no_display:
            cv2.imshow('Deteksi Jerigen SPBU — YOLOv8m', frame)
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
    print(f"  Rata-rata FPS   : {avg_fps:.2f}")
    print(f"  Total frame     : {frame_ke}")
    print(f"  Baris CSV        : {no_csv - 1}")
    print(f"  File CSV         : {CONFIG['csv_file']}")
    if CONFIG["simpan_screenshot"]:
        print(f"  Screenshot       : {screenshot_count} gambar")
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
