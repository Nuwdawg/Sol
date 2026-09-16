set -u
fail() { echo "ERROR {\"stage\": \"launcher\", \"message\": \"$1\"}"; sleep infinity; }
mkdir -p /workspace/exp097 /workspace/code || fail "cannot create workspace"
test -f /workspace/exp097/pod_start_epoch || date +%s > /workspace/exp097/pod_start_epoch
export SOL_POD_START_EPOCH="$(cat /workspace/exp097/pod_start_epoch)"
echo "STAGE {\"name\": \"launcher\", \"pod\": \"${RUNPOD_POD_ID:-unknown}\"}"
export PIP_BREAK_SYSTEM_PACKAGES=1
command -v python >/dev/null 2>&1 || ln -sf "$(command -v python3)" /usr/local/bin/python
command -v pip >/dev/null 2>&1 || ln -sf "$(command -v pip3)" /usr/local/bin/pip
pip install -q --no-cache-dir --break-system-packages kaggle==2.2.4 || fail "kaggle cli install failed"
cd /workspace/code || fail "no code directory"
test -f sol_exp097_code.payload || kaggle datasets download cosmicwitness/sol-exp097-code -p /workspace/code --unzip -q || fail "code dataset download failed"
echo "${SOL_CODE_SHA256}  sol_exp097_code.payload" | sha256sum -c - || fail "code tarball SHA256 mismatch"
tar -xzf sol_exp097_code.payload || fail "code tarball extraction failed"
echo "STAGE {\"name\": \"code_verified\", \"sha256\": \"${SOL_CODE_SHA256}\"}"
exec bash experiments/exp_097_runpod_yolo/pod/bootstrap.sh 2>&1
