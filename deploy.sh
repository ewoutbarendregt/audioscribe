#!/usr/bin/env bash
# Deploy AudioScribe to the Trustable VPS fleet (staging + prod).
#
# AudioScribe runs as a profile-gated service inside the trustable
# docker-compose stack on each VPS. Its release cycle is independent of the
# trustable web/api deploys: the trustable workflows never enable the
# "audioscribe" profile, so they never touch this container. This script is
# the ONLY thing that builds, ships, and restarts AudioScribe.
#
# Usage:
#   ./deploy.sh              # build, push, deploy to BOTH staging and prod
#   ./deploy.sh staging      # staging only
#   ./deploy.sh prod         # prod only
#   TAG=v1.2.0 ./deploy.sh   # deploy a specific image tag (default: latest)
#
# Prerequisites:
#   - docker login ghcr.io  (echo $GHCR_PAT | docker login ghcr.io -u ewoutbarendregt --password-stdin)
#   - SSH access to each target. By default this uses the SSH config aliases
#     'trustable-staging' and 'trustable-prod' (which already encode the deploy
#     user + key). Override with STAGING_SSH_HOST / PROD_SSH_HOST if needed.
#   - /opt/trustable/audioscribe.env present on each target VPS (see below)
#
# First-time setup on EACH VPS (staging and prod), run once.
# The env file is the single source of truth for the key — nothing else reads
# it. Keep a backup wherever you normally store secrets.
#   ssh trustable-staging   # then repeat for trustable-prod
#   cat > /opt/trustable/audioscribe.env << 'EOF'
#   GEMINI_API_KEY=<paste your Gemini key>
#   API_TOKEN=<generate with: openssl rand -hex 32>   # legacy/scripted access only
#   # --- interactive sign-in (emailed one-time code) ---
#   ALLOWED_EMAILS=you@example.com,colleague@example.com   # seeds the user table once
#   SMTP_HOST=smtp.resend.com
#   SMTP_PORT=587
#   SMTP_USER=resend
#   SMTP_PASS=<Resend API key>
#   MAIL_FROM=audioscribe@trustable.nl      # domain must be verified in Resend
#   COOKIE_PATH=/projects/audioscribe       # cookie scope behind the Caddy prefix
#   EOF
#   chmod 640 /opt/trustable/audioscribe.env
#
# Sessions AND the list of permitted users live in a SQLite database on the
# audioscribe_data volume (see docker-compose.audioscribe.yml), so sign-ins and
# accounts survive redeploys. ALLOWED_EMAILS only seeds that list on a brand-new
# database; afterwards manage users with:
#   docker compose ... exec audioscribe python auth.py add|remove|list <email>

set -euo pipefail

IMAGE="ghcr.io/ewoutbarendregt/audioscribe"
TAG="${TAG:-latest}"
TARGET="${1:-both}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SSH targets — default to the ~/.ssh/config aliases (user + key baked in).
STAGING_SSH_HOST="${STAGING_SSH_HOST:-trustable-staging}"
PROD_SSH_HOST="${PROD_SSH_HOST:-trustable-prod}"

# ---------------------------------------------------------------------------
# Resolve target hosts
# ---------------------------------------------------------------------------
declare -a HOSTS=()
case "$TARGET" in
  staging) HOSTS+=("$STAGING_SSH_HOST") ;;
  prod)    HOSTS+=("$PROD_SSH_HOST") ;;
  both)
    HOSTS+=("$STAGING_SSH_HOST")
    HOSTS+=("$PROD_SSH_HOST")
    ;;
  *)
    echo "Unknown target '$TARGET'. Use: staging | prod | both" >&2
    exit 1
    ;;
esac

echo "=== AudioScribe deploy: ${IMAGE}:${TAG} → ${TARGET} ==="
echo ""

# ---------------------------------------------------------------------------
# 1. Build (for the VPS architecture) and push, in one step
#    The Trustable VPSes are x86_64, so we always target linux/amd64 — this
#    matters when building from an arm64 (Apple Silicon) machine.
# ---------------------------------------------------------------------------
PLATFORM="${PLATFORM:-linux/amd64}"
echo "→ Building (${PLATFORM}) and pushing ${IMAGE}:${TAG} to GHCR..."
docker buildx build --platform "${PLATFORM}" -t "${IMAGE}:${TAG}" --push "${SCRIPT_DIR}"
echo ""

# ---------------------------------------------------------------------------
# 3. Roll out to each target host
# ---------------------------------------------------------------------------
deploy_to_host() {
  local host="$1"
  echo "→ Deploying to ${host}..."
  # Ship the AudioScribe-only compose overlay (declares the /data volume that holds
  # the auth database). It lives here rather than in trustable's docker-compose.yml
  # because that file is owned and overwritten by trustable's own deploy.
  scp -q "${SCRIPT_DIR}/docker-compose.audioscribe.yml" "${host}:/opt/trustable/"
  ssh "${host}" bash -s -- "${TAG}" << 'REMOTE'
set -euo pipefail
TAG="$1"
cd /opt/trustable

if [ ! -f /opt/trustable/audioscribe.env ]; then
  echo "ERROR: /opt/trustable/audioscribe.env is missing on this host." >&2
  echo "       Create it before deploying (see deploy.sh header)." >&2
  exit 1
fi

# --profile audioscribe scopes every command to JUST this service, so the
# trustable web/api/db/caddy containers are never touched by this deploy.
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.audioscribe.yml"
AUDIOSCRIBE_IMAGE_TAG="${TAG}" $COMPOSE --profile audioscribe pull audioscribe
AUDIOSCRIBE_IMAGE_TAG="${TAG}" $COMPOSE --profile audioscribe up -d audioscribe
docker image prune -f

sleep 3
$COMPOSE --profile audioscribe ps audioscribe
REMOTE
  echo "   ✓ ${host} done."
  echo ""
}

for host in "${HOSTS[@]}"; do
  deploy_to_host "$host"
done

echo "=== Deploy complete ==="
case "$TARGET" in
  staging) echo "   https://staging.trustable.nl/projects/audioscribe/" ;;
  prod)    echo "   https://trustable.nl/projects/audioscribe/" ;;
  both)
    echo "   https://staging.trustable.nl/projects/audioscribe/"
    echo "   https://trustable.nl/projects/audioscribe/"
    ;;
esac
