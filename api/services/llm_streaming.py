"""SSE streaming wrapper around SiliconFlowClient.chat_stream()."""

from __future__ import annotations

from typing import Generator

from quant.llm.client import SiliconFlowClient, LLMConfig, LLMError


def stream_sse(
    config: LLMConfig,
    messages: list[dict],
) -> Generator[dict, None, None]:
    """Yield SSE event dicts from LLM streaming response.

    Event types (aligned with frontend SSEEvent type):
      - {"type": "chunk", "content": "token"}
      - {"type": "done", "content": "full accumulated text"}
      - {"type": "error", "message": "error description"}
    """
    client = SiliconFlowClient(config)
    full_text = ""

    try:
        for token in client.chat_stream(messages):
            full_text += token
            yield {"type": "chunk", "content": token}
        yield {"type": "done", "content": full_text}
    except LLMError as e:
        yield {"type": "error", "message": str(e)}
    except Exception as e:
        yield {"type": "error", "message": f"LLM streaming failed: {e}"}
