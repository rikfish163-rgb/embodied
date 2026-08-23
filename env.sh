# source 我 —— 每次开工第一件事；不会改变当前工作目录
_EMB_SOURCE="${BASH_SOURCE[0]}"
export EMB="$(cd -- "$(dirname -- "$_EMB_SOURCE")" && pwd)"
unset _EMB_SOURCE

export UV_CACHE_DIR="$EMB/cache/uv"
export HF_HOME="$EMB/hf"
export TORCH_HOME="$EMB/cache/torch"
export MUJOCO_GL=egl
export MENAGERIE="$EMB/menagerie"
export PYTHONPATH="$EMB/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPYCACHEPREFIX="$EMB/cache/pycache"

if [[ -f "$EMB/venv/bin/activate" ]]; then
    source "$EMB/venv/bin/activate"
elif [[ -f "$EMB/.venv/bin/activate" ]]; then
    source "$EMB/.venv/bin/activate"
else
    echo "No virtual environment found. Run: uv sync --group test" >&2
    return 1
fi
