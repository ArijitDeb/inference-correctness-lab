#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Create a git worktree of the vLLM clone for one tree under test, and link
# the clone's precompiled build artifacts (*.so, _version.py, vendored
# third_party dirs) into it, so `PYTHONPATH=trees/<name>` imports a working
# vLLM from that tree. The kernels were built for the clone's base commit;
# every tree must sit on that same base so the Python and the kernels agree.
#
#   ./setup_trees.sh main            # trees/main at $BASE_REF
#   ./setup_trees.sh pr              # trees/pr-53298: PR #53298 merged onto $BASE_REF
#   ./setup_trees.sh fix <branch>    # trees/fix on the fix branch (created from $BASE_REF)
set -euo pipefail

CLONE=${VLLM_CLONE:-$HOME/Projects/vllm}
BASE_REF=${BASE_REF:-main-04462732}
HERE=$(cd "$(dirname "$0")" && pwd)
TREES="$HERE/trees"
mkdir -p "$TREES"

link_artifacts() {
  local wt=$1 exclude artifacts
  artifacts=$(git -C "$CLONE" ls-files --others --ignored --exclude-standard --directory -- vllm/ \
    | grep -v '__pycache__' || true)
  if [ -z "$artifacts" ]; then
    echo "!! no build artifacts in $CLONE/vllm: install it first" >&2
    echo "!! (VLLM_USE_PRECOMPILED=1 uv pip install -e . from the clone)" >&2
    exit 1
  fi
  # The links must never be committed. .gitignore's "dir/" patterns don't
  # match a symlink, so each linked path also goes into info/exclude (shared
  # by the clone and all its worktrees; the clone ignores these paths anyway).
  exclude="$(git -C "$wt" rev-parse --path-format=absolute --git-common-dir)/info/exclude"
  mkdir -p "$(dirname "$exclude")"
  touch "$exclude"
  while read -r entry; do
    local path=${entry%/}
    grep -qxF "/$path" "$exclude" || echo "/$path" >> "$exclude"
    if [ -e "$wt/$path" ] || [ -L "$wt/$path" ]; then
      continue  # tracked in this tree, or already linked
    fi
    mkdir -p "$(dirname "$wt/$path")"
    ln -s "$CLONE/$path" "$wt/$path"
  done <<< "$artifacts"
}

case ${1:-} in
  main)
    wt="$TREES/main"
    [ -d "$wt" ] || git -C "$CLONE" worktree add --detach "$wt" "$BASE_REF"
    ;;
  pr)
    wt="$TREES/pr-53298"
    if [ ! -d "$wt" ]; then
      git -C "$CLONE" worktree add -B olf/pr-53298-on-main "$wt" pr-53298
      if ! git -C "$wt" merge --no-edit "$BASE_REF"; then
        # Leave nothing behind: a half-built tree would be reused as "ready".
        git -C "$wt" merge --abort || true
        git -C "$CLONE" worktree remove --force "$wt"
        git -C "$CLONE" branch -D olf/pr-53298-on-main
        echo "!! PR #53298 does not merge cleanly onto $BASE_REF." >&2
        exit 1
      fi
    fi
    git -C "$wt" merge-base --is-ancestor pr-53298 HEAD \
      || { echo "!! $wt does not contain pr-53298" >&2; exit 1; }
    ;;
  fix)
    # A real branch, not a detached checkout: this is where the fix is
    # written and committed. Created from $BASE_REF if it doesn't exist.
    ref=${2:?usage: ./setup_trees.sh fix <branch>}
    wt="$TREES/fix"
    if [ ! -d "$wt" ]; then
      if git -C "$CLONE" show-ref --verify --quiet "refs/heads/$ref"; then
        git -C "$CLONE" worktree add "$wt" "$ref"
      else
        git -C "$CLONE" worktree add -b "$ref" "$wt" "$BASE_REF"
      fi
    fi
    ;;
  *)
    sed -n '4,12p' "$0"; exit 2
    ;;
esac

# Every tree must contain $BASE_REF, or its Python won't match the kernels.
git -C "$wt" merge-base --is-ancestor "$BASE_REF" HEAD \
  || { echo "!! $wt is not based on $BASE_REF; kernels would not match." >&2; exit 1; }

link_artifacts "$wt"
echo "ready: $wt ($(git -C "$wt" log --oneline -1))"
