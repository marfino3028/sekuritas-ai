#!/usr/bin/env bash
#
# deploy-victoria-sekuritas.sh
#
# Menjalankan D-1 s/d D-6 untuk deploy platform Victoria Sekuritas ke VPS.
# Repo: sekuritas-api (Laravel 12 + Postgres), sekuritas-frontend (Nuxt 3),
#       sekuritas-cms (Nuxt 3), sekuritas-ai (FastAPI eKYC: Nanonets OCR,
#       InsightFace, Facenox liveness ONNX).
#
# PENTING SEBELUM DIJALANKAN:
#   - Ini scaffold berdasarkan konteks yang kamu kasih, BUKAN hasil baca
#     langsung isi repo (docker-compose.yml, nama service, port asli, dsb).
#     Cek dan sesuaikan semua bagian bertanda [CEK/SESUAIKAN] di bawah
#     sebelum run di production.
#   - Script ini berhenti (checkpoint) di titik-titik kritis dan minta
#     konfirmasi manual (tekan Enter) sebelum lanjut, khususnya di D-3
#     karena kalau model AI gagal ke-download tapi tetap lanjut, service
#     bisa nyala tapi diam-diam salah.
#   - Jalankan sebagai user non-root yang sudah masuk grup `docker`.
#
# Cara pakai:
#   chmod +x deploy-victoria-sekuritas.sh
#   ./deploy-victoria-sekuritas.sh            # jalan semua tahap D1-D6
#   ./deploy-victoria-sekuritas.sh d3         # jalan cuma satu tahap
#
set -euo pipefail

# ----------------------------- KONFIGURASI -----------------------------
GITHUB_USER="marfino3028"
BASE_DIR="/opt/victoria-sekuritas"
REPOS=(sekuritas-api sekuritas-frontend sekuritas-cms sekuritas-ai)
# sekuritas-mobile sengaja TIDAK di-clone/deploy di sini (build APK terpisah)

AI_DIR="${BASE_DIR}/sekuritas-ai"
API_DIR="${BASE_DIR}/sekuritas-api"
FRONTEND_DIR="${BASE_DIR}/sekuritas-frontend"
CMS_DIR="${BASE_DIR}/sekuritas-cms"

# [CEK/SESUAIKAN] path model & health endpoint sesuai kode asli sekuritas-ai
LIVENESS_MODEL_PATH="${AI_DIR}/model/liveness/best_model_quantized.onnx"
AI_HEALTH_URL="http://localhost:8000/health"

# ----------------------------- HELPER -----------------------------
log()  { printf '\n\033[1;34m[%s]\033[0m %s\n' "$(date '+%H:%M:%S')" "$*"; }
err()  { printf '\n\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[1;32m[OK]\033[0m %s\n' "$*"; }

checkpoint() {
  # Berhenti dan minta konfirmasi manual sebelum lanjut ke tahap berikutnya.
  local msg="$1"
  echo
  echo "==================== CHECKPOINT ===================="
  echo "$msg"
  echo "======================================================"
  read -r -p "Lanjut ke tahap berikutnya? Ketik 'lanjut' untuk konfirmasi: " answer
  if [[ "$answer" != "lanjut" ]]; then
    err "Dihentikan oleh user di checkpoint. Perbaiki dulu, lalu jalankan ulang tahap ini."
    exit 1
  fi
}

require_file() {
  local path="$1" desc="$2"
  if [[ ! -s "$path" ]]; then
    err "File tidak ditemukan atau kosong: $path ($desc)"
    return 1
  fi
  ok "$desc ditemukan: $path ($(du -h "$path" | cut -f1))"
}

# ----------------------------- D-1: Persiapan VPS -----------------------------
d1_prepare_vps() {
  log "D-1: Persiapan VPS"

  log "Info sistem:"
  lsb_release -a 2>/dev/null || cat /etc/os-release
  free -h
  df -h /

  if ! command -v docker &>/dev/null; then
    err "Docker belum terinstall. Install dulu (curl -fsSL https://get.docker.com | sh), lalu jalankan ulang."
    exit 1
  fi
  ok "Docker terpasang: $(docker --version)"

  if ! docker compose version &>/dev/null; then
    err "Docker Compose plugin tidak ditemukan. Install docker-compose-plugin dulu."
    exit 1
  fi
  ok "Docker Compose terpasang: $(docker compose version)"

  if ! groups "$USER" | grep -q docker; then
    err "User $USER belum masuk grup 'docker'. Jalankan: sudo usermod -aG docker $USER, lalu re-login."
    exit 1
  fi
  ok "User $USER sudah di grup docker"

  if [[ ! -d "$BASE_DIR" ]]; then
    log "Folder $BASE_DIR belum ada, buat dengan sudo lalu chown ke user $USER..."
    sudo mkdir -p "$BASE_DIR"
    sudo chown "$USER":"$USER" "$BASE_DIR"
  elif [[ ! -w "$BASE_DIR" ]]; then
    log "Folder $BASE_DIR ada tapi tidak writable oleh $USER, chown dulu..."
    sudo chown -R "$USER":"$USER" "$BASE_DIR"
  fi
  ok "Folder kerja siap: $BASE_DIR (owner: $(stat -c '%U' "$BASE_DIR"))"

  # [CEK/SESUAIKAN] sesuaikan aturan ufw dengan kebijakan keamanan kamu
  if command -v ufw &>/dev/null; then
    log "Status firewall (ufw) saat ini:"
    sudo ufw status verbose || true
    echo "[CEK/SESUAIKAN] Pastikan hanya port 22/80/443 yang terbuka ke publik;"
    echo "service internal (postgres, sekuritas-ai, dll) sebaiknya HANYA lewat docker network internal."
  else
    echo "ufw tidak terpasang — cek firewall pakai tool lain kalau ada."
  fi

  checkpoint "Cek output di atas: versi OS/Docker/Compose, RAM & disk cukup, firewall sudah sesuai kebijakan kamu."
}

# ----------------------------- D-2: Clone semua repo -----------------------------
d2_clone_repos() {
  log "D-2: Clone semua repo"

  for repo in "${REPOS[@]}"; do
    local dest="${BASE_DIR}/${repo}"
    if [[ -d "$dest/.git" ]]; then
      log "Repo $repo sudah ada, pull update terbaru..."
      git -C "$dest" pull --ff-only
    else
      log "Clone $repo..."
      git clone "https://github.com/${GITHUB_USER}/${repo}.git" "$dest"
    fi

    if [[ -f "${dest}/.env.example" && ! -f "${dest}/.env" ]]; then
      cp "${dest}/.env.example" "${dest}/.env"
      ok "Dibuat ${dest}/.env dari .env.example"
    fi
  done

  log "Struktur folder hasil clone:"
  for repo in "${REPOS[@]}"; do
    echo " - ${BASE_DIR}/${repo}"
  done

  checkpoint "Pastikan semua repo di atas berhasil ter-clone (tidak ada error git) dan .env sudah dibuat."
}

# ----------------------------- D-3: Setup model AI -----------------------------
d3_setup_ai_models() {
  log "D-3: Setup model AI untuk sekuritas-ai (bagian paling kritis)"

  cd "$AI_DIR"

  # --- Liveness: Facenox ONNX, sudah ada script resminya ---
  log "Download model liveness (Facenox ONNX)..."
  if [[ -f "scripts/download_liveness_model.sh" ]]; then
    chmod +x scripts/download_liveness_model.sh
    ./scripts/download_liveness_model.sh
  else
    err "scripts/download_liveness_model.sh tidak ditemukan di repo. Cek ulang path/nama script."
    exit 1
  fi
  require_file "$LIVENESS_MODEL_PATH" "Model liveness (Facenox, best_model_quantized.onnx)"

  # --- OCR: Nanonets-OCR-s (GGUF) ---
  # [CEK/SESUAIKAN] Ganti bagian ini sesuai script resmi repo kalau ada
  # (misal scripts/download_ocr_model.sh). Kalau tidak ada, ini contoh
  # download manual dari HuggingFace — sesuaikan nama file GGUF & path target.
  log "Setup model OCR (Nanonets-OCR-s GGUF)..."
  if [[ -f "scripts/download_ocr_model.sh" ]]; then
    chmod +x scripts/download_ocr_model.sh
    ./scripts/download_ocr_model.sh
  else
    echo "[CEK/SESUAIKAN] Tidak ada script download OCR bawaan repo."
    echo "Cek README/kode sekuritas-ai untuk tahu path target & nama file GGUF yang benar,"
    echo "lalu download manual dari https://huggingface.co/NanoNets/Nanonets-OCR-s ke path tsb."
    echo "Lewati auto-download di sini sampai path/nama file dikonfirmasi manual."
  fi
  # require_file "${AI_DIR}/model/ocr/<nama-file>.gguf" "Model OCR Nanonets"

  # --- Face match: InsightFace ---
  # [CEK/SESUAIKAN] InsightFace kadang auto-download model saat pertama kali
  # dipanggil (bukan lewat script terpisah). Konfirmasi dulu di kode repo.
  log "Cek model face-match (InsightFace)..."
  echo "[CEK/SESUAIKAN] Baca kode sekuritas-ai untuk pastikan apakah InsightFace"
  echo "auto-download model (mis. buffalo_l) saat container start, atau perlu"
  echo "didownload manual seperti liveness. Jangan asumsikan otomatis."

  checkpoint "WAJIB: pastikan ketiga model (liveness/OCR/face-match) benar-benar ada di path yang dipakai .env sebelum lanjut ke D-4. Kalau ada yang belum, STOP dan selesaikan dulu di luar script ini."
}

# ----------------------------- D-4: Build & run sekuritas-ai -----------------------------
d4_build_run_ai() {
  log "D-4: Build & jalankan sekuritas-ai"

  cd "$AI_DIR"

  log "Isi .env yang relevan (cek manual apakah sudah benar):"
  grep -E 'OCR_ENGINE|FACE_MATCH_ENGINE|LIVENESS_ENGINE' .env || true

  # [CEK/SESUAIKAN] nama service di docker-compose.yml mungkin beda
  log "Build image sekuritas-ai..."
  docker compose build sekuritas-ai

  log "Jalankan container sekuritas-ai..."
  docker compose up -d sekuritas-ai

  log "Tunggu container startup, lalu tampilkan log..."
  sleep 5
  docker compose logs --tail=100 sekuritas-ai

  log "Test health endpoint: $AI_HEALTH_URL"
  if curl -fsS "$AI_HEALTH_URL"; then
    ok "Health endpoint merespons"
  else
    err "Health endpoint gagal diakses. Cek log di atas untuk error load model."
    exit 1
  fi

  checkpoint "Cek log startup: pastikan ketiga engine (OCR/liveness/face-match) sukses load model tanpa exception, bukan cuma container 'up'."
}

# ----------------------------- D-5: Deploy sekuritas-api -----------------------------
d5_deploy_api() {
  log "D-5: Deploy sekuritas-api (Laravel + PostgreSQL) & sambungkan ke sekuritas-ai"

  cd "$API_DIR"

  # [CEK/SESUAIKAN] service name postgres & sekuritas-ai di docker network internal
  log "Build & jalankan sekuritas-api + postgres..."
  docker compose build sekuritas-api
  docker compose up -d postgres sekuritas-api

  sleep 5

  log "Generate APP_KEY & migrate..."
  docker compose exec sekuritas-api php artisan key:generate --force
  docker compose exec sekuritas-api php artisan migrate --force

  echo "[CEK/SESUAIKAN] Pastikan .env sekuritas-api punya:"
  echo "  SEKURITAS_AI_URL=http://sekuritas-ai:8000   (nama service, bukan localhost)"
  grep -E 'SEKURITAS_AI_URL|JWT_SECRET' .env || echo "  (belum ketemu di .env — tambahkan manual)"

  checkpoint "Test satu alur eKYC end-to-end LEWAT sekuritas-api (bukan langsung ke sekuritas-ai) sebelum lanjut ke D-6."
}

# ----------------------------- D-6: Deploy frontend, CMS, reverse proxy -----------------------------
d6_deploy_frontend_cms() {
  log "D-6: Deploy frontend, CMS, reverse proxy, smoke test akhir"

  for dir in "$FRONTEND_DIR" "$CMS_DIR"; do
    cd "$dir"
    # [CEK/SESUAIKAN] pastikan API base URL di .env mengarah ke sekuritas-api yang benar
    docker compose build "$(basename "$dir")" || docker compose build
    docker compose up -d
  done

  echo "[CEK/SESUAIKAN] Setup reverse proxy (nginx/Traefik) manual di sini:"
  echo "  - Expose HANYA frontend, cms, dan sekuritas-api (kalau perlu diakses mobile) ke publik"
  echo "  - Pasang TLS (certbot / Let's Encrypt) kalau ada domain"

  checkpoint "Setelah reverse proxy & TLS beres: buka frontend, coba alur registrasi/eKYC nasabah end-to-end, dan cek login admin CMS."

  log "Ringkasan service (cek manual):"
  docker compose ps || true
}

# ----------------------------- MAIN -----------------------------
main() {
  local stage="${1:-all}"
  case "$stage" in
    d1) d1_prepare_vps ;;
    d2) d2_clone_repos ;;
    d3) d3_setup_ai_models ;;
    d4) d4_build_run_ai ;;
    d5) d5_deploy_api ;;
    d6) d6_deploy_frontend_cms ;;
    all)
      d1_prepare_vps
      d2_clone_repos
      d3_setup_ai_models
      d4_build_run_ai
      d5_deploy_api
      d6_deploy_frontend_cms
      log "Selesai D-1 s/d D-6. Cek ringkasan service di atas."
      ;;
    *)
      err "Tahap tidak dikenal: $stage (pilihan: d1 d2 d3 d4 d5 d6 all)"
      exit 1
      ;;
  esac
}

main "$@"