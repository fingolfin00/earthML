#!/usr/bin/env bash
set -euo pipefail

#######################################
# User parameters
#######################################

# Base directory containing the experiment folders
BASE_DIR="/users_home/cmcc/jd19424/work/test-ML/experiments_earthML_weather"

# Pattern matching the source experiment folders
SRC_GLOB="exp_weather_t2m*_mseloss"

# Token to replace in source folder names
SRC_SUFFIX="mseloss"

# Derived experiment suffixes
NEW_SUFFIXES=(
  "maskedmseloss"
  "variancenormalizedmseloss_spatial"
  "variancenormalizedmseloss_geochannel"
  "heterobiascorrectionloss_spatial_usefirstinput"
)

# Datasets to centralize
DATASETS=(
  "test_input.zarr"
  "test_target.zarr"
  "train_input.zarr"
  "train_target.zarr"
)

# Where to store shared data:
#   inside_src -> BASE_DIR/<src>/<DATA_SUBDIR>
#   central    -> CENTRAL_DATA_ROOT/<src>/<DATA_SUBDIR>
DATA_MODE="central"

# Shared subdirectory name
DATA_SUBDIR="data"

# Used only if DATA_MODE="central"
CENTRAL_DATA_ROOT="/users_home/cmcc/jd19424/work/test-ML/shared_experiment_data"

# rsync options
RSYNC_OPTS=(-a)

# 1 = print commands only, 0 = execute
DRY_RUN=0

# 1 = create destination folders if missing
CREATE_DST_DIRS=1

# 1 = replace existing dataset entries with links
FORCE_REPLACE_LINKS=1

#######################################
# Helpers
#######################################

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[DRY] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

msg() {
  printf '%s\n' "$*"
}

abspath() {
  python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

relpath() {
  python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$1" "$2"
}

get_data_dir() {
  local src="$1"
  case "$DATA_MODE" in
    inside_src)
      printf '%s\n' "$BASE_DIR/$src/$DATA_SUBDIR"
      ;;
    central)
      printf '%s\n' "$CENTRAL_DATA_ROOT/$src/$DATA_SUBDIR"
      ;;
    *)
      printf 'Error: unsupported DATA_MODE=%s\n' "$DATA_MODE" >&2
      exit 1
      ;;
  esac
}

#######################################
# Main
#######################################

cd "$BASE_DIR"
shopt -s nullglob

for src in $SRC_GLOB; do
  [[ -d "$src" ]] || continue
  msg "Processing source: $src"

  src_abs="$BASE_DIR/$src"
  data_dir="$(get_data_dir "$src")"

  run mkdir -p "$data_dir"

  #
  # Step 1: centralize datasets into data_dir
  #         and link them back inside the source folder
  #
  for ds in "${DATASETS[@]}"; do
    src_ds="$src_abs/$ds"
    shared_ds="$data_dir/$ds"

    # If dataset is still physically in the source root, move/copy it to shared location
    if [[ -d "$src_ds" && ! -L "$src_ds" ]]; then
      if [[ ! -e "$shared_ds" ]]; then
        run mkdir -p "$shared_ds"
        run rsync "${RSYNC_OPTS[@]}" "$src_ds/" "$shared_ds/"
      fi
      run rm -rf "$src_ds"
    fi

    # Link from source root -> shared dataset
    src_link_target="$(relpath "$shared_ds" "$src_abs")"
    if [[ "$FORCE_REPLACE_LINKS" -eq 1 ]]; then
      run rm -rf "$src_ds"
    fi
    run ln -sfn "$src_link_target" "$src_ds"
  done

  #
  # Step 2: link datasets into derived experiment folders
  #
  for new_suffix in "${NEW_SUFFIXES[@]}"; do
    dst="${src/$SRC_SUFFIX/$new_suffix}"

    if [[ "$dst" == "$src" ]]; then
      msg "  Warning: replacement did not change name: $src"
      continue
    fi

    dst_abs="$BASE_DIR/$dst"

    if [[ ! -d "$dst_abs" ]]; then
      if [[ "$CREATE_DST_DIRS" -eq 1 ]]; then
        run mkdir -p "$dst_abs"
      else
        msg "  Destination missing, skipping: $dst"
        continue
      fi
    fi

    for ds in "${DATASETS[@]}"; do
      dst_ds="$dst_abs/$ds"
      dst_link_target="$(relpath "$data_dir/$ds" "$dst_abs")"

      if [[ "$FORCE_REPLACE_LINKS" -eq 1 ]]; then
        run rm -rf "$dst_ds"
      fi

      run ln -sfn "$dst_link_target" "$dst_ds"
    done
  done
done
