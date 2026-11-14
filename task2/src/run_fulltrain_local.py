"""Run task2_fulltrain.ipynb locally, with persistent logs and per-epoch resume."""

import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone

TASK = Path(__file__).resolve().parent
TASK = TASK.parent if TASK.name == "src" else TASK  # 位于 src/ 时回到任务根目录
OUT = TASK / "task2_fulltrain_resnet34_seed42"
OUT.mkdir(exist_ok=True)
os.chdir(TASK)
os.environ["TORCH_HOME"] = str(TASK / ".torch")
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)


def status(state, **extra):
    payload = dict(
        state=state, pid=os.getpid(), updated_at=datetime.now(timezone.utc).isoformat(), **extra
    )
    temporary = OUT / "local_status.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(OUT / "local_status.json")


def main():
    # Windows releases the lock on process exit, including interruption.
    import msvcrt

    with (OUT / "local_run.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("已有本地全量训练进程运行，请勿重复启动。") from None
        status("starting")
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("未检测到 CUDA，停止以免意外在 CPU 上训练。")
            notebook = json.loads(
                (TASK / "notebooks" / "task2_fulltrain.ipynb").read_text(encoding="utf-8")
            )
            scope = {"__name__": "__main__"}
            stages = {
                1: "configuration",
                3: "data_scan",
                5: "model_setup",
                7: "training",
                9: "prediction_and_export",
            }
            for index, stage in stages.items():
                status("running", stage=stage)
                print(f"\n===== {stage} =====", flush=True)
                exec(
                    compile(notebook["cells"][index]["source"], f"fulltrain_cell_{index}", "exec"),
                    scope,
                )
                if index == 1:
                    scope["DATA_DIR"] = TASK / "2026task2/fer_data/fer_data"
                    scope["ROOT"] = TASK
                    scope["OUT"] = OUT
            submission = OUT / "submission_fulltrain.csv"
            if not submission.is_file():
                raise RuntimeError("预测阶段结束但未生成提交文件。")
            status("completed", submission=str(submission))
            print("本地全量训练与预测已完成：", submission, flush=True)
        except BaseException as exc:
            status(
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)
            )
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
