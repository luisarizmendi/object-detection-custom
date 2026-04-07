#!/usr/bin/env bash
# build-all.sh — build all service images found in subdirectories
#
# Discovers build.sh scripts automatically. Runs _bases_/ first (if present),
# then all other first-level directories that contain a build.sh.
# By default: builds all services for the host architecture, push.
#
# Usage:
#   ./build-all.sh                            # build all services, local only
#   ./build-all.sh --no-push                  # build all but do not push
#   ./build-all.sh --cross                    # also cross-build the other arch
#   ./build-all.sh --registry quay.io/myorg   # override registry
#   ./build-all.sh --prod-tag release-1.2     # use a custom stable manifest tag
#   ./build-all.sh --force-manifest-reset     # ignore existing remote arches
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Args: forward everything to per-service build.sh ─────────────────────────
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  FORWARD_ARGS+=("$1"); shift
done

# ── Logging helpers ───────────────────────────────────────────────────────────
_sep()     { printf '%72s\n' '' | tr ' ' '═'; }
_banner()  { echo ""; _sep; printf "  %s\n" "$1"; _sep; }
_svc_hdr() { echo ""; printf '  ┌─ Service: %s\n' "$1"; printf '  └%s\n' "$(printf '%68s' '' | tr ' ' '─')"; }
_ok()      { printf "\n  ✓  %s\n" "$1"; }
_info()    { printf "  ℹ  %s\n"   "$1"; }
_warn()    { printf "  ⚠  %s\n"   "$1"; }

# ── QEMU pre-flight (only relevant when --cross is forwarded) ─────────────────
for arg in "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"; do
  if [[ "$arg" == "--cross" ]]; then
    if ! ls /proc/sys/fs/binfmt_misc/qemu-aarch64 &>/dev/null; then
      _warn "QEMU binfmt_misc not found — cross-build will likely fail."
      _warn "  sudo dnf install qemu-user-static   # Fedora/RHEL"
      _warn "  sudo apt install qemu-user-static    # Debian/Ubuntu"
    fi
    break
  fi
done

# ── Discover services ─────────────────────────────────────────────────────────
# Collect base services (from _bases_/) and regular services separately
BASE_SERVICES=()
REGULAR_SERVICES=()

if [[ -d "${SCRIPT_DIR}/_bases_" ]]; then
  for dir in "${SCRIPT_DIR}/_bases_"/*/; do
    [[ -f "${dir}build.sh" ]] && BASE_SERVICES+=("$dir")
  done
fi

for dir in "${SCRIPT_DIR}"/*/; do
  dir="${dir%/}"
  [[ "$(basename "$dir")" == "_bases_" ]] && continue
  [[ -f "${dir}/build.sh" ]] && REGULAR_SERVICES+=("$dir")
done

ALL_SERVICES=("${BASE_SERVICES[@]}" "${REGULAR_SERVICES[@]}")

# ── Summary ───────────────────────────────────────────────────────────────────
_banner "build-all.sh"

_push_mode="push enabled"
for a in "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"; do [[ "$a" == "--no-push" ]] && _push_mode="push disabled"; done
_info "Push     : ${_push_mode}"

if [[ ${#BASE_SERVICES[@]} -gt 0 ]]; then
  _info "Bases    : ${BASE_SERVICES[*]}"
fi
_info "Services : ${#ALL_SERVICES[@]} found"
_info "Forward  : ${FORWARD_ARGS[*]+"${FORWARD_ARGS[*]}"}"

# ── Build each service ────────────────────────────────────────────────────────
FAILED=()

run_build() {
  local dir=$1
  local name
  name="$(basename "$dir")"

  _svc_hdr "$name"

  if (
    cd "$dir"
    chmod +x build.sh
    ./build.sh "${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}"
  ); then
    _ok "${name} — done"
  else
    _warn "${name} — FAILED (continuing with remaining services)"
    FAILED+=("$name")
  fi
}

if [[ ${#BASE_SERVICES[@]} -gt 0 ]]; then
  _banner "Building base images (_bases_/)"
  for dir in "${BASE_SERVICES[@]}"; do
    run_build "$dir"
  done
fi

_banner "Building remaining images"
for dir in "${REGULAR_SERVICES[@]}"; do
  run_build "$dir"
done

# ── Final report ──────────────────────────────────────────────────────────────
_banner "Build summary"

if [[ ${#FAILED[@]} -eq 0 ]]; then
  _ok "All ${#ALL_SERVICES[@]} service(s) built successfully."
else
  _warn "Failed services: ${FAILED[*]}"
  echo ""
  exit 1
fi

echo ""