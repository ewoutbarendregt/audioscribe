#!/usr/bin/env bash
# Deploy AudioScribe from this Mac to the Trustable servers (staging + prod).
# There is no GitHub Actions deploy: this script is the only thing that builds,
# ships and restarts AudioScribe.
#
# AudioScribe runs as a profile-gated service inside the trustable
# docker-compose stack (/opt/trustable) on each server. Trustable's own deploys
# never enable the "audioscribe" profile, so they never touch this container.
#
# Usage:
#   ./deploy.sh [staging|prod|both] [options]   # default target: both
#     --ref <git-ref>   deploy this commit/branch/tag (default: origin/master)
#     --dry-run         build the image and check the server; change nothing on it
#     --yes, -y         skip the confirmation prompt
#
# What it does:
#   1. git fetch; check out <ref> into a temporary worktree, so uncommitted local
#      work is never deployed and an old ref really builds the old code
#   2. image tag = v<APP_VERSION> from that ref's main.py (never "latest")
#   3. preflight per host: ssh works, audioscribe.env exists, show the running tag
#      and whether the compose overlay changes
#   4. build linux/amd64 with docker buildx (--load, no registry push)
#   5. per host: docker save | gzip | ssh docker load, copy the compose overlay,
#      upload the server steps as a file and run them with `ssh -n`; the run only
#      counts if it prints REMOTE-DEPLOY-OK, then the running image tag is verified
#
# Hosts (~/.ssh/config aliases, user deploy):
#   prod    = trustable-contabo (Contabo, 144.91.87.103; moved from Hetzner
#             trustable-prod on 2026-11-24). Override with PROD_SSH_HOST.
#   staging = trustable-staging. Override with STAGING_SSH_HOST.
#
# First-time setup on EACH VPS (staging and prod), run once.
# The env file is the single source of truth for the key — nothing else reads
# it. Keep a backup wherever you normally store secrets.
#   ssh trustable-staging   # then repeat for trustable-contabo (prod)
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
usage() { sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

TARGET="both"
case "${1:-}" in
  staging|prod|both) TARGET="$1"; shift ;;
esac
REF="origin/master"; DRY_RUN=0; YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --ref) REF="${2:?--ref needs a value}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) YES=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown option: $1 (target must come first: staging | prod | both)" >&2; usage 1 ;;
  esac
done
if [ -n "${TAG:-}" ]; then
  echo "TAG is no longer supported: the tag comes from the deployed ref's APP_VERSION." >&2
  echo "To deploy an older version, use --ref <commit or tag of that version>." >&2
  exit 1
fi

IMAGE="ghcr.io/ewoutbarendregt/audioscribe"
PLATFORM="${PLATFORM:-linux/amd64}"   # the servers are x86_64; matters on Apple Silicon
REMOTE_DIR="/opt/trustable"
STAGING_SSH_HOST="${STAGING_SSH_HOST:-trustable-staging}"
PROD_SSH_HOST="${PROD_SSH_HOST:-trustable-contabo}"   # Contabo since 2026-11-24 (was trustable-prod on Hetzner)

declare -a HOSTS=() URLS=()
case "$TARGET" in
  staging) HOSTS=("$STAGING_SSH_HOST"); URLS=("https://staging.trustable.nl/projects/audioscribe/") ;;
  prod)    HOSTS=("$PROD_SSH_HOST");    URLS=("https://trustable.nl/projects/audioscribe/") ;;
  both)    HOSTS=("$STAGING_SSH_HOST" "$PROD_SSH_HOST")
           URLS=("https://staging.trustable.nl/projects/audioscribe/" "https://trustable.nl/projects/audioscribe/") ;;
esac

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
# Short path: the ssh ControlPath socket must stay under ~104 characters on macOS.
WORK="$(mktemp -d /tmp/asdeploy.XXXXXX)"

# One multiplexed ssh connection per host: sshd on Contabo resets bursts of new
# connections, so every ssh/scp below reuses the master connection.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=auto
          -o "ControlPath=$WORK/cm-%C" -o ControlPersist=300)
rssh() { ssh "${SSH_OPTS[@]}" "$@"; }
rscp() { scp -q "${SSH_OPTS[@]}" "$@"; }

cleanup() {
  for h in "${HOSTS[@]}"; do ssh "${SSH_OPTS[@]}" -O exit "$h" >/dev/null 2>&1 || true; done
  git -C "$REPO_ROOT" worktree remove --force "$WORK/src" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

# ── 1. Source ──────────────────────────────────────────────────────────────────
log "Fetching and resolving $REF"
git -C "$REPO_ROOT" fetch --quiet origin
SHA="$(git -C "$REPO_ROOT" rev-parse --verify --quiet "$REF^{commit}")" || die "Unknown ref: $REF"
git -C "$REPO_ROOT" worktree add --quiet --detach "$WORK/src" "$SHA"
SRC="$WORK/src"
APP_VERSION="$(sed -n 's/^APP_VERSION = "\(.*\)"$/\1/p' "$SRC/main.py")"
[ -n "$APP_VERSION" ] || die "No APP_VERSION found in main.py at $REF"
TAG="v${APP_VERSION}"
echo "AudioScribe ${IMAGE}:${TAG} ← $REF (${SHA:0:7}): $(git -C "$REPO_ROOT" log -1 --format=%s "$SHA")"
echo "Target: $TARGET (${HOSTS[*]})$([ "$DRY_RUN" = 1 ] && echo ', DRY RUN')"

# ── 2. Preflight ───────────────────────────────────────────────────────────────
log "Preflight"
command -v docker >/dev/null || die "docker not found"
docker buildx version >/dev/null 2>&1 || die "docker buildx not available"
declare -a PREV=()
for host in "${HOSTS[@]}"; do
  rssh "$host" true || die "Cannot ssh to $host"
  rssh "$host" "test -f $REMOTE_DIR/audioscribe.env" \
    || die "$REMOTE_DIR/audioscribe.env is missing on $host. Create it first (see the deploy.sh header)."
  rssh "$host" "test -f $REMOTE_DIR/docker-compose.yml && test -f $REMOTE_DIR/.env" \
    || die "$REMOTE_DIR/docker-compose.yml or .env missing on $host"
  running="$(rssh "$host" "docker inspect --format '{{.Config.Image}}' trustable-audioscribe-1 2>/dev/null" || true)"
  PREV+=("${running:-none}")
  echo "$host: running ${running:-nothing}; .env $(rssh "$host" "grep -m1 '^AUDIOSCRIBE_IMAGE_TAG=' $REMOTE_DIR/.env" || echo 'has no AUDIOSCRIBE_IMAGE_TAG')"
  if rssh "$host" "cat $REMOTE_DIR/docker-compose.audioscribe.yml 2>/dev/null" | cmp -s - "$SRC/docker-compose.audioscribe.yml"; then
    echo "$host: docker-compose.audioscribe.yml unchanged"
  else
    echo "$host: docker-compose.audioscribe.yml will change:"
    diff -u <(rssh "$host" "cat $REMOTE_DIR/docker-compose.audioscribe.yml 2>/dev/null" || true) \
      "$SRC/docker-compose.audioscribe.yml" | sed 's/^/    /' || true
  fi
done

# ── 3. Build ───────────────────────────────────────────────────────────────────
log "Building ${IMAGE}:${TAG} (${PLATFORM})"
docker buildx build --platform "$PLATFORM" --load -t "${IMAGE}:${TAG}" "$SRC"

if [ "$DRY_RUN" = 1 ]; then
  log "Dry run: image built, nothing changed on ${HOSTS[*]}"
  exit 0
fi

if [ "$YES" != 1 ]; then
  printf '\nDeploy %s (%s) to %s? [y/N] ' "$TAG" "$REF" "$TARGET"
  read -r answer
  [ "$answer" = "y" ] || [ "$answer" = "Y" ] || die "Aborted"
fi

# ── 4. Server steps ────────────────────────────────────────────────────────────
# Uploaded as a file and run with `ssh -n`, never piped via `bash -s`/heredoc:
# a stdin-reading command would swallow the rest of the script while ssh still
# exits 0. The end marker catches any run that stops early.
cat > "$WORK/remote.sh" <<'REMOTE'
set -euo pipefail
TAG="$1"
cd /opt/trustable
# --profile audioscribe scopes every command to JUST this service, so the
# trustable web/api/db/caddy containers are never touched by this deploy.
COMPOSE="docker compose -f docker-compose.yml -f docker-compose.audioscribe.yml --profile audioscribe"

mv docker-compose.audioscribe.yml.new docker-compose.audioscribe.yml
# --pull never: the image was just loaded onto this host.
AUDIOSCRIBE_IMAGE_TAG="$TAG" $COMPOSE up -d --pull never audioscribe < /dev/null

# Persist the tag in .env so a later `docker compose up` (reboot, manual recreate)
# keeps this version instead of falling back to :latest.
if grep -q '^AUDIOSCRIBE_IMAGE_TAG=' .env; then
  sed -i "s/^AUDIOSCRIBE_IMAGE_TAG=.*/AUDIOSCRIBE_IMAGE_TAG=${TAG}/" .env
else
  printf 'AUDIOSCRIBE_IMAGE_TAG=%s\n' "$TAG" >> .env
fi

echo "Waiting for audioscribe to become healthy"
timeout 90 sh -c "until $COMPOSE ps audioscribe < /dev/null | grep -q '(healthy)'; do sleep 3; done"
mkdir -p .deploy
echo "$(date -u +%FT%TZ) audioscribe $TAG" >> .deploy/history.log
docker image prune -f >/dev/null
$COMPOSE ps audioscribe < /dev/null
echo "REMOTE-DEPLOY-OK"
REMOTE

for i in "${!HOSTS[@]}"; do
  host="${HOSTS[$i]}"
  log "Deploying ${TAG} to ${host}"
  echo "Uploading image"
  docker save "${IMAGE}:${TAG}" | gzip -1 | rssh "$host" 'gunzip | docker load'
  rscp "$SRC/docker-compose.audioscribe.yml" "$host:$REMOTE_DIR/docker-compose.audioscribe.yml.new"
  rssh -n "$host" "mkdir -p $REMOTE_DIR/.deploy"
  rscp "$WORK/remote.sh" "$host:$REMOTE_DIR/.deploy/audioscribe-remote.sh"
  rssh -n "$host" "bash $REMOTE_DIR/.deploy/audioscribe-remote.sh '$TAG' < /dev/null" | tee "$WORK/remote-$i.log"
  grep -q '^REMOTE-DEPLOY-OK$' "$WORK/remote-$i.log" || die "Server steps on $host did not finish (see output above)"

  running="$(rssh -n "$host" "docker inspect --format '{{.Config.Image}}' trustable-audioscribe-1")"
  [ "$running" = "${IMAGE}:${TAG}" ] || die "$host runs '$running', expected ${IMAGE}:${TAG}"
  echo "✓ $host runs $running (was ${PREV[$i]})"
  curl -fsS -m 15 "${URLS[$i]}health" && echo || echo "WARNING: ${URLS[$i]}health did not answer" >&2
done

log "Deployed ${TAG} to ${TARGET}"
printf '   %s\n' "${URLS[@]}"
echo "Rollback: ./deploy.sh $TARGET --ref <commit of the previous version>   (previous: ${PREV[*]})"
