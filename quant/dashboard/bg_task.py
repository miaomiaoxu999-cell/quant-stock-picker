"""通用后台任务管理 — 让 LLM 流式调用和耗时分析在后台线程执行，切页不中断"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import streamlit as st

from quant.llm.client import SiliconFlowClient, LLMConfig, LLMError


# ==================== 数据结构 ====================

@dataclass
class BgTask:
    task_id: str
    status: str = "running"  # running | completed | failed | cancelled
    result: Any = None
    error: str = ""
    text: str = ""  # 流式文本累积
    thread: threading.Thread | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


def _get_tasks() -> dict[str, BgTask]:
    if "_bg_tasks" not in st.session_state:
        st.session_state["_bg_tasks"] = {}
    return st.session_state["_bg_tasks"]


# ==================== 核心 API ====================

def bg_llm_stream(
    task_id: str,
    llm_config: LLMConfig,
    messages: list[dict],
    *,
    retry_key: str = "btn_retry",
) -> str | None:
    """后台 LLM 流式调用。

    - 首次调用：启动 daemon 线程
    - 运行中：显示已累积文本 + 光标 + 中断按钮，轮询刷新
    - 完成：返回完整文本
    - 失败：显示错误 + 重试按钮，返回 None
    - 取消：显示已中断 + 部分结果，返回 None
    """
    tasks = _get_tasks()
    task = tasks.get(task_id)

    # 首次：启动线程
    if task is None:
        task = BgTask(task_id=task_id)

        def worker():
            try:
                client = SiliconFlowClient(llm_config)
                for chunk in client.chat_stream(messages):
                    if task.cancel_event.is_set():
                        task.status = "cancelled"
                        return
                    task.text += chunk
                task.result = task.text
                task.status = "completed"
            except LLMError as e:
                task.error = str(e)
                task.status = "failed"
            except Exception as e:
                task.error = f"LLM 调用异常: {e}"
                task.status = "failed"

        t = threading.Thread(target=worker, daemon=True)
        task.thread = t
        tasks[task_id] = task
        t.start()
        # 短暂等待让首批 chunk 到达
        time.sleep(0.3)

    # 渲染当前状态
    if task.status == "running":
        # 显示已累积文本
        if task.text:
            st.markdown(task.text + " **|**")
        else:
            st.caption("AI 正在思考...")
        if st.button("中断", key=f"cancel_{task_id}"):
            task.cancel_event.set()
            task.status = "cancelled"
            st.rerun()
        # 轮询
        time.sleep(1)
        st.rerun()

    if task.status == "completed":
        result = task.result
        return result

    if task.status == "failed":
        st.error(task.error)
        if st.button("重试", key=f"{retry_key}_{task_id}"):
            tasks.pop(task_id, None)
            st.rerun()
        return None

    if task.status == "cancelled":
        st.warning("已中断")
        if task.text:
            with st.expander("查看部分结果"):
                st.markdown(task.text)
        if st.button("重新开始", key=f"restart_{task_id}"):
            tasks.pop(task_id, None)
            st.rerun()
        return None

    return None


def bg_run(
    task_id: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any | None:
    """后台执行任意函数。

    - 运行中：显示 spinner + 中断按钮
    - 完成：返回函数返回值
    - 失败/取消：同 bg_llm_stream
    """
    tasks = _get_tasks()
    task = tasks.get(task_id)

    if task is None:
        task = BgTask(task_id=task_id)

        def worker():
            try:
                task.result = fn(*args, **kwargs, _cancel_event=task.cancel_event)
                if task.cancel_event.is_set():
                    task.status = "cancelled"
                else:
                    task.status = "completed"
            except TypeError:
                # fn 不接受 _cancel_event 参数，重试不传
                try:
                    task.result = fn(*args, **kwargs)
                    task.status = "completed"
                except Exception as e:
                    task.error = str(e)
                    task.status = "failed"
            except Exception as e:
                task.error = str(e)
                task.status = "failed"

        t = threading.Thread(target=worker, daemon=True)
        task.thread = t
        tasks[task_id] = task
        t.start()

    if task.status == "running":
        st.caption("正在分析中...")
        if st.button("中断", key=f"cancel_{task_id}"):
            task.cancel_event.set()
            task.status = "cancelled"
            st.rerun()
        time.sleep(1.5)
        st.rerun()

    if task.status == "completed":
        return task.result

    if task.status == "failed":
        st.error(task.error)
        if st.button("重试", key=f"retry_{task_id}"):
            tasks.pop(task_id, None)
            st.rerun()
        return None

    if task.status == "cancelled":
        st.warning("已中断")
        if st.button("重新开始", key=f"restart_{task_id}"):
            tasks.pop(task_id, None)
            st.rerun()
        return None

    return None


# ==================== 辅助函数 ====================

def has_task(task_id: str) -> bool:
    """检查指定任务是否存在（未清理）"""
    return task_id in _get_tasks()


def clear_task(task_id: str) -> None:
    """清除已完成/失败/取消的任务"""
    tasks = _get_tasks()
    task = tasks.get(task_id)
    if task and task.status != "running":
        tasks.pop(task_id, None)


def get_running_count() -> int:
    """当前运行中的后台任务数"""
    tasks = _get_tasks()
    return sum(1 for t in tasks.values() if t.status == "running")


def render_running_indicator() -> None:
    """在侧边栏显示运行中任务数"""
    n = get_running_count()
    if n > 0:
        st.sidebar.info(f"{n} 个分析任务运行中")
