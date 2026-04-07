#!/usr/bin/env bash
set -e

# --- Defaults ---
REGISTRY="quay.io/luisarizmendi"
CROSS_BUILD=false
PUSH=true
FORCE_MANIFEST_RESET=false

# --- Parse args ---
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --cross)
      CROSS_BUILD=true
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --no-push)
      PUSH=false
      FORWARD_ARGS+=("$1")
      shift
      ;;
    --registry)
      REGISTRY="$2"
      FORWARD_ARGS+=("$1" "$2")
      shift 2
      ;;
    --force-manifest-reset)
      FORCE_MANIFEST_RESET=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# --- Get script directory name as IMAGE_NAME ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="$(basename "$SCRIPT_DIR")"

# --- Detect host architecture ---
### force arm64
HOST_ARCH="arm64" 

cd "$SCRIPT_DIR/src"

echo "========================================"
echo "Image: ${REGISTRY}/${IMAGE_NAME}"
echo "Host arch: ${HOST_ARCH}"
echo "Cross build: ${CROSS_BUILD}"
echo "Push: ${PUSH}"
echo "Force manifest reset: ${FORCE_MANIFEST_RESET}"
echo "========================================"
echo ""

# --- Build images ---
BUILT_IMAGES=()

build_image() {
  local arch=$1
  local tag="${REGISTRY}/${IMAGE_NAME}:${arch}"

  # Remove any stale local image/manifest-list under this tag before building.
  # A previous run (or a pull) may have left a corrupted manifest-list entry
  # in local storage; podman build will fail trying to overwrite it.
  # "manifest rm" handles the manifest-list case; "rmi --force" handles a
  # plain image; both are silenced — it's fine if neither exists yet.
  podman manifest rm "$tag" 2>/dev/null || true
  podman rmi --force  "$tag" 2>/dev/null || true

  echo "→ Building for $arch..."
  podman build --platform "linux/${arch}" -t "$tag" .
  BUILT_IMAGES+=("$tag")
}

build_image "$HOST_ARCH"

if [[ "$CROSS_BUILD" == true ]]; then
  if [[ "$HOST_ARCH" == "amd64" ]]; then
    build_image "arm64"
  elif [[ "$HOST_ARCH" == "arm64" ]]; then
    build_image "amd64"
  fi
fi

# --- Manifest names ---
MANIFEST_PROD="${REGISTRY}/${IMAGE_NAME}:prod"
MANIFEST_LATEST="${REGISTRY}/${IMAGE_NAME}:latest"

# --- Architectures built this run ---
BUILT_ARCHES=()
for img in "${BUILT_IMAGES[@]}"; do
  BUILT_ARCHES+=("$(basename "$img" | sed 's/.*://')")
done

# --- All known architectures ---
ALL_ARCHES=("amd64" "arm64")

# ---------------------------------------------------------------------------
# push_manifest <manifest_tag>
#
# 1. Push the freshly built arch-specific images to the registry.
# 2. For every arch we did NOT build, try to pull its arch-specific tag from
#    the registry (e.g. :arm64). If it exists, it goes into the manifest too.
# 3. Create a fresh local manifest, add everything, push.
# ---------------------------------------------------------------------------
push_manifest() {
  local manifest=$1

  # Clean up any stale local manifest
  podman manifest rm  "$manifest" 2>/dev/null || true
  podman rmi --force  "$manifest" 2>/dev/null || true
  podman manifest create "$manifest"

  if [[ "$FORCE_MANIFEST_RESET" == false ]]; then
    # Pull arch-specific images we did NOT build this run
    for arch in "${ALL_ARCHES[@]}"; do
      local already_built=false
      for built_arch in "${BUILT_ARCHES[@]}"; do
        [[ "$built_arch" == "$arch" ]] && already_built=true && break
      done

      if [[ "$already_built" == false ]]; then
        local arch_tag="${REGISTRY}/${IMAGE_NAME}:${arch}"
        echo "  → Pulling existing ${arch} image from registry: ${arch_tag}"
        if podman pull --platform "linux/${arch}" "${arch_tag}" 2>/dev/null; then
          echo "  → Adding pulled ${arch} to manifest"
          podman manifest add "$manifest" "$arch_tag"
        else
          echo "  → No existing ${arch} image found in registry — skipping"
        fi
      fi
    done
  fi

  # Add the freshly built images
  for img in "${BUILT_IMAGES[@]}"; do
    local arch
    arch=$(basename "$img" | sed 's/.*://')
    echo "  → Adding freshly built ${arch} to manifest"
    podman manifest add "$manifest" "$img"
  done

  echo "  Manifest contents:"
  podman manifest inspect "$manifest" \
    | python3 -c "
import sys, json
for m in json.load(sys.stdin).get('manifests', []):
    arch   = m.get('platform', {}).get('architecture', '?')
    digest = m.get('digest', '')
    print(f'    {arch}  →  {digest[:19]}...')
"
}

# --- Push logic ---
if [[ "$PUSH" == true ]]; then
  echo ""
  echo "→ Pushing arch-specific images..."
  for img in "${BUILT_IMAGES[@]}"; do
    podman push "$img"
  done

  echo ""
  echo "→ Building and pushing manifest: prod"
  push_manifest "$MANIFEST_PROD"
  podman manifest push --rm "$MANIFEST_PROD"

  echo ""
  echo "→ Building and pushing manifest: latest"
  push_manifest "$MANIFEST_LATEST"
  podman manifest push --rm "$MANIFEST_LATEST"

  echo ""
  echo "✅ Done."
else
  echo ""
  echo "⚠️  Push disabled. Skipping manifest + push steps."
fi