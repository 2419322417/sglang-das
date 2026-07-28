#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 2 || ! "$1" =~ ^(run|preflight|cleanup)$ || "$2" != --role=* ]]; then
  echo "Usage: $0 run|preflight|cleanup --role=prefill|decode" >&2
  exit 2
fi

COMMAND="$1"
ROLE="${2#--role=}"
if [[ "${ROLE}" != "prefill" && "${ROLE}" != "decode" ]]; then
  echo "Invalid HCU PD role: ${ROLE}" >&2
  exit 2
fi

IMAGE="${HCU_PD_IMAGE:?HCU_PD_IMAGE is required}"
CHECKOUT="${HCU_PD_CHECKOUT:-${GITHUB_WORKSPACE:-$PWD}}"
RUN_ID="${GITHUB_RUN_ID:?GITHUB_RUN_ID is required}"
ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
CONTROLLER_NAME="ci_sglang_hcu_pd_controller_${RUN_ID}_${ATTEMPT}_${ROLE}"

if [[ ! -f "${CHECKOUT}/scripts/ci/hcu/hcu_ci_pd.py" ]]; then
  echo "Invalid HCU PD checkout: ${CHECKOUT}" >&2
  exit 2
fi
if [[ ! -S /var/run/docker.sock || ! -x /usr/bin/docker ]]; then
  echo "Docker socket or static Docker CLI is unavailable." >&2
  exit 2
fi

docker rm -f "${CONTROLLER_NAME}" >/dev/null 2>&1 || true

DOCKER_GROUP="$(stat -c '%g' /var/run/docker.sock)"
CONTROLLER_ARGS=(
  --rm
  --name "${CONTROLLER_NAME}"
  --user "$(id -u):$(id -g)"
  --group-add "${DOCKER_GROUP}"
  --privileged
  --network host
  --uts host
  --ipc host
  -v /var/run/docker.sock:/var/run/docker.sock
  -v /usr/bin/docker:/usr/bin/docker:ro
  -v "${CHECKOUT}:${CHECKOUT}:ro"
  -v /ci_public:/ci_public
  -v /sys:/sys:ro
  -w "${CHECKOUT}"
)

for path in /public /public4 /opt/hyhal; do
  if [[ -e "${path}" ]]; then
    CONTROLLER_ARGS+=(-v "${path}:${path}:ro")
  fi
done

for device in /dev/kfd /dev/dri /dev/infiniband; do
  if [[ -e "${device}" ]]; then
    CONTROLLER_ARGS+=(-v "${device}:${device}")
  fi
done

FORWARDED_ENV=(
  GITHUB_REPOSITORY
  GITHUB_RUN_ID
  GITHUB_RUN_ATTEMPT
  RUNNER_NAME
  HCU_PD_SHA
  HCU_PD_TARGET_REF
  HCU_PD_IMAGE
  HCU_PD_IMAGE_ID
  HCU_PD_CHECKOUT
  HCU_PD_PREFILL_IP
  HCU_PD_DECODE_IP
  HCU_PD_LOCAL_IP
  HCU_PD_PEER_IP
  HCU_PD_LOCAL_IFNAME
  HCU_PD_LOCAL_IB_DEVICE
  HCU_PD_GID_INDEX
  HCU_PD_SHARED_ROOT
  HCU_PD_WHEEL_ROOT
  HCU_PD_SHARED_GID
  HCU_PD_PEER_TIMEOUT
  HCU_PD_SERVICE_TIMEOUT
  HCU_PD_HEARTBEAT_TIMEOUT
  HCU_PD_COMPLETION_TIMEOUT
  SGLANG_HCU_MINIMAX_M27_MODEL
)

for variable in "${FORWARDED_ENV[@]}"; do
  if [[ -v "${variable}" ]]; then
    CONTROLLER_ARGS+=(-e "${variable}=${!variable}")
  fi
done

exec docker run "${CONTROLLER_ARGS[@]}" \
  "${IMAGE}" \
  python3 "${CHECKOUT}/scripts/ci/hcu/hcu_ci_pd.py" \
  "${COMMAND}" --role "${ROLE}"
