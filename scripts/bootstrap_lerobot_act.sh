#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat >&2 <<'EOF'
usage: scripts/bootstrap_lerobot_act.sh MODE --run-dir PATH [--repo-root PATH] [--supporting-evidence PATH ...]

MODE is one of: audit, compile-lock, sync, smoke.
The current R1.4 lock is dependency-security blocked, so sync and smoke exit
before creating .venv-lerobot. compile-lock only creates the resolver artifact.
EOF
  exit 2
}

MODE="${1-}"
case "$MODE" in
  audit|compile-lock|sync|smoke) shift ;;
  *) usage ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || exit 2
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd -P)" || exit 2
EMB="$DEFAULT_REPO"
RUN_DIR=""
SUPPORTING_EVIDENCE=()
while (($#)); do
  case "$1" in
    --repo-root)
      (($# >= 2)) || usage
      EMB="$2"
      shift 2
      ;;
    --run-dir)
      (($# >= 2)) || usage
      RUN_DIR="$2"
      shift 2
      ;;
    --supporting-evidence)
      (($# >= 2)) || usage
      SUPPORTING_EVIDENCE+=("$2")
      shift 2
      ;;
    *) usage ;;
  esac
done

EMB="$(cd -- "$EMB" && pwd -P)" || exit 2
[[ -n "$RUN_DIR" ]] || usage
RUN_DIR="$(/usr/bin/realpath -m -- "$RUN_DIR")" || exit 2
[[ "$RUN_DIR" == "$EMB/runs/m3/"* ]] || {
  printf '%s\n' 'run directory must be below $EMB/runs/m3/' >&2
  exit 2
}
[[ -f "$EMB/README.md" && -d "$EMB/.git" ]] || {
  printf '%s\n' 'repo root must contain README.md and .git' >&2
  exit 2
}

LR_PREFIX="$EMB/.venv-lerobot"
LR_BOOTSTRAP_PY=/usr/bin/python3.12
VERIFIER="$EMB/scripts/verify_lerobot_environment.py"
CANONICAL="$EMB/src/policy/canonical_json.py"
REQUIREMENTS="$EMB/requirements.lerobot-act.in"
LOCK="$EMB/requirements.lerobot-act.lock.txt"
[[ -x "$LR_BOOTSTRAP_PY" && -f "$VERIFIER" && -f "$CANONICAL" && -f "$REQUIREMENTS" ]] || {
  printf '%s\n' 'required bootstrap input is missing' >&2
  exit 2
}
"$LR_BOOTSTRAP_PY" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 2)' || exit 2

mkdir -p -- "$EMB/runs/m3" || exit 2
mkdir -- "$RUN_DIR" || {
  printf 'run directory already exists or cannot be created: %s\n' "$RUN_DIR" >&2
  exit 2
}

RECEIPT="$RUN_DIR/lerobot_smoke_receipt.json"

publish_preflight_blocked() {
  local failure_stage="$1" reason_code="$2" message="$3"
  local failure_log="$RUN_DIR/${failure_stage}.stderr.log"
  local draft="$RUN_DIR/.lerobot_smoke_receipt.draft.json"
  local -a receipt_evidence draft_command

  (set -o noclobber; printf '%s\n' "$message" >"$failure_log") || exit 70
  receipt_evidence=("${SUPPORTING_EVIDENCE[@]}" "$failure_log")
  draft_command=(
    "$LR_BOOTSTRAP_PY" "$VERIFIER" preflight-blocked-receipt-draft
    --repository "$EMB"
    --expected-prefix "$LR_PREFIX"
    --failure-stage "$failure_stage"
    --reason-code "$reason_code"
    --data-free-kib "${data_free_kib:--1}"
    --root-free-kib "${root_free_kib:--1}"
    --output "$draft"
  )
  for evidence_path in "${receipt_evidence[@]}"; do
    draft_command+=(--supporting-evidence "$evidence_path")
  done
  env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/local/bin:/usr/bin:/bin \
    "${draft_command[@]}" || exit 70
  env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/local/bin:/usr/bin:/bin \
    "$LR_BOOTSTRAP_PY" "$CANONICAL" publish-id \
      --input "$draft" --identity-field receipt_id --output "$RECEIPT" || exit 70
  rm -- "$draft" || exit 70
  (set -o noclobber
   sha256sum "$RECEIPT" >"$RUN_DIR/lerobot_smoke_receipt.sha256") || exit 70
  printf 'LeRobot ACT isolate is BLOCKED (%s); receipt: %s\n' \
    "$reason_code" "$RECEIPT"
  exit 3
}

data_free_kib=-1
root_free_kib=-1
if ! data_free_kib="$(/usr/bin/df -Pk -- "$EMB" | /usr/bin/awk 'NR == 2 {print $4}')" ||
   [[ ! "$data_free_kib" =~ ^[0-9]+$ ]]; then
  data_free_kib=-1
  publish_preflight_blocked initial_space_preflight df_unreadable \
    'could not obtain the initial data-disk free-space snapshot'
fi
if ! root_free_kib="$(/usr/bin/df -Pk -- / | /usr/bin/awk 'NR == 2 {print $4}')" ||
   [[ ! "$root_free_kib" =~ ^[0-9]+$ ]]; then
  root_free_kib=-1
  publish_preflight_blocked initial_space_preflight df_unreadable \
    'could not obtain the initial root-disk free-space snapshot'
fi
if ((data_free_kib < 20 * 1024 * 1024)); then
  publish_preflight_blocked initial_space_preflight insufficient_data_space \
    'initial data-disk free space is below 20 GiB'
fi
if ((root_free_kib < 3 * 1024 * 1024)); then
  publish_preflight_blocked initial_space_preflight insufficient_root_space \
    'initial root-disk free space is below 3 GiB'
fi

for ignored_dir in \
  "$LR_PREFIX" \
  "$EMB/cache/uv-lerobot" \
  "$EMB/cache/pip-lerobot" \
  "$EMB/cache/tmp-lerobot" \
  "$EMB/cache/xdg-lerobot" \
  "$EMB/hf" \
  "$EMB/cache/torch" \
  "$EMB/cache/cuda-lerobot" \
  "$EMB/cache/ccache-lerobot" \
  "$EMB/cache/pycache-lerobot" \
  "$RUN_DIR"; do
  git -C "$EMB" check-ignore -q "$ignored_dir/" || {
    publish_preflight_blocked initial_ignore_preflight directory_not_ignored \
      "required directory is not ignored: $ignored_dir/"
  }
done

if [[ "$MODE" == compile-lock && ! -e "$LOCK" ]]; then
  UV_BIN="$(command -v uv)" || publish_preflight_blocked resolver_preflight \
    dependency_input_missing 'uv executable is unavailable'
  COMPILE_STDOUT="$RUN_DIR/resolver.stdout.log"
  COMPILE_STDERR="$RUN_DIR/resolver.stderr.log"
  COMPILE_CODE="$RUN_DIR/resolver.exit-code.txt"
  CANDIDATE_LOCK="$RUN_DIR/requirements.lerobot-act.lock.candidate.txt"
  set +e
  (set -o noclobber
   env -u VIRTUAL_ENV -u PYTHONPATH \
     PATH=/usr/local/bin:/usr/bin:/bin \
     "UV_CACHE_DIR=$EMB/cache/uv-lerobot" \
     "PIP_CACHE_DIR=$EMB/cache/pip-lerobot" \
     "TMPDIR=$EMB/cache/tmp-lerobot" \
     "XDG_CACHE_HOME=$EMB/cache/xdg-lerobot" \
     UV_PYTHON_DOWNLOADS=never \
     "$UV_BIN" pip compile "$REQUIREMENTS" \
       --python "$LR_BOOTSTRAP_PY" --torch-backend cu128 \
       --generate-hashes --emit-index-url --emit-index-annotation \
       --custom-compile-command 'scripts/bootstrap_lerobot_act.sh compile-lock' \
       --output-file "$CANDIDATE_LOCK" \
       >"$COMPILE_STDOUT" 2>"$COMPILE_STDERR")
  compile_rc=$?
  set -u
  if ! (set -o noclobber; printf '%s\n' "$compile_rc" >"$COMPILE_CODE"); then
    SUPPORTING_EVIDENCE+=("$COMPILE_STDOUT" "$COMPILE_STDERR")
    publish_preflight_blocked resolver_evidence \
      resolver_incomplete_or_indeterminate \
      'resolver exit-code evidence could not be published'
  fi
  SUPPORTING_EVIDENCE+=("$COMPILE_STDOUT" "$COMPILE_STDERR" "$COMPILE_CODE")
  (set -o noclobber
   sha256sum "$COMPILE_STDOUT" "$COMPILE_STDERR" "$COMPILE_CODE" \
     >"$RUN_DIR/resolver.logs.sha256") || publish_preflight_blocked \
       resolver_evidence resolver_incomplete_or_indeterminate \
       'resolver log hashes could not be published'
  SUPPORTING_EVIDENCE+=("$RUN_DIR/resolver.logs.sha256")
  ((compile_rc == 0)) || publish_preflight_blocked resolver \
    resolver_incomplete_or_indeterminate \
    "uv lock resolver exited nonzero: $compile_rc"
  ln -- "$CANDIDATE_LOCK" "$LOCK" || publish_preflight_blocked resolver_publish \
    resolver_incomplete_or_indeterminate \
    'resolved lock could not be published with no-clobber semantics'
fi

[[ -f "$LOCK" ]] || {
  publish_preflight_blocked dependency_input_preflight dependency_input_missing \
    'compiled LeRobot lock is missing'
}

AUDIT_STDOUT="$RUN_DIR/dependency-audit.stdout.log"
AUDIT_STDERR="$RUN_DIR/dependency-audit.stderr.log"
AUDIT_CODE="$RUN_DIR/dependency-audit.exit-code.txt"
DEPENDENCY_EVIDENCE="$RUN_DIR/dependency-security-evidence.json"
set +e
(set -o noclobber
 env -u VIRTUAL_ENV -u PYTHONPATH \
   PATH=/usr/local/bin:/usr/bin:/bin \
   "$LR_BOOTSTRAP_PY" "$VERIFIER" audit-lock \
     --requirements "$REQUIREMENTS" --lock "$LOCK" \
     --output "$DEPENDENCY_EVIDENCE" \
     >"$AUDIT_STDOUT" 2>"$AUDIT_STDERR")
audit_rc=$?
set -u
if ! (set -o noclobber; printf '%s\n' "$audit_rc" >"$AUDIT_CODE"); then
  SUPPORTING_EVIDENCE+=("$AUDIT_STDOUT" "$AUDIT_STDERR")
  publish_preflight_blocked dependency_audit_evidence \
    dependency_audit_incomplete \
    'dependency-audit exit-code evidence could not be published'
fi
SUPPORTING_EVIDENCE+=("$AUDIT_STDOUT" "$AUDIT_STDERR" "$AUDIT_CODE")
(set -o noclobber
 sha256sum "$AUDIT_STDOUT" "$AUDIT_STDERR" "$AUDIT_CODE" \
   >"$RUN_DIR/dependency-audit.logs.sha256") || publish_preflight_blocked \
     dependency_audit_evidence dependency_audit_incomplete \
     'dependency-audit log hashes could not be published'
SUPPORTING_EVIDENCE+=("$RUN_DIR/dependency-audit.logs.sha256")
if [[ "$audit_rc" == 0 ]]; then
  publish_preflight_blocked dependency_security_audit \
    security_protocol_revision_required \
    'lock cleared the frozen security gate, but no reviewed protocol revision enables sync'
fi
if [[ "$audit_rc" != 3 || ! -s "$DEPENDENCY_EVIDENCE" ]]; then
  publish_preflight_blocked dependency_security_audit dependency_audit_incomplete \
    "dependency audit was incomplete or unexpected: exit $audit_rc"
fi

DRAFT="$RUN_DIR/.lerobot_smoke_receipt.draft.json"
draft_command=(
  "$LR_BOOTSTRAP_PY" "$VERIFIER" blocked-receipt-draft
  --repository "$EMB"
  --expected-prefix "$LR_PREFIX"
  --dependency-evidence "$DEPENDENCY_EVIDENCE"
  --data-free-kib "$data_free_kib"
  --root-free-kib "$root_free_kib"
  --output "$DRAFT"
)
for evidence_path in "${SUPPORTING_EVIDENCE[@]}"; do
  draft_command+=(--supporting-evidence "$evidence_path")
done
env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/local/bin:/usr/bin:/bin \
  "${draft_command[@]}" || exit 70

env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/local/bin:/usr/bin:/bin \
  "$LR_BOOTSTRAP_PY" "$CANONICAL" publish-id \
    --input "$DRAFT" --identity-field receipt_id --output "$RECEIPT" || exit 70
rm -- "$DRAFT" || exit 70
(set -o noclobber
 sha256sum "$RECEIPT" >"$RUN_DIR/lerobot_smoke_receipt.sha256") || exit 70

printf 'LeRobot ACT isolate is BLOCKED by dependency security; receipt: %s\n' "$RECEIPT"
exit 3
