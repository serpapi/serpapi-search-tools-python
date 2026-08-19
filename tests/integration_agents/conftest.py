from __future__ import annotations

import json
import re
import socket
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request


@dataclass
class RecordingSearchClient:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def search(self, params: dict[str, Any]) -> Mapping[str, Any] | str:
        call = dict(params)
        self.calls.append(call)
        primary_key = {
            "google_news": "news_results",
            "google_maps": "local_results",
            "google_images": "images_results",
            "google_shopping": "shopping_results",
            "youtube": "video_results",
            "google_hotels": "properties",
            "google_flights": "best_flights",
            "google_travel_explore": "destinations",
        }.get(str(call["engine"]), "organic_results")
        if call.get("output") == "md":
            heading = {
                "news_results": "News Results",
                "local_results": "Local Results",
                "images_results": "Images Results",
                "shopping_results": "Shopping Results",
                "video_results": "Video Results",
                "properties": "Properties",
                "best_flights": "Best Flights",
                "destinations": "Destinations",
            }.get(primary_key, "Organic Results")
            return (
                "---\nsearch_metadata:\n  status: Success\n---\n\n"
                f"## {heading}\n\n"
                "| Title | Link |\n"
                "| --- | --- |\n"
                f"| Fake result for {call} | https://example.com/result |\n"
            )
        return {
            "search_metadata": {"status": "Success"},
            "search_parameters": call,
            primary_key: [
                {"title": f"Fake result for {call}", "link": "https://example.com/result"}
            ],
        }


@dataclass
class FakeOpenAIServer:
    url: str
    requests: list[dict[str, Any]]


@pytest.fixture
def serpapi_client() -> RecordingSearchClient:
    return RecordingSearchClient()


@pytest.fixture
def fake_openai_server() -> Iterator[FakeOpenAIServer]:
    app = FastAPI()
    requests: list[dict[str, Any]] = []

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": "gpt-4o-mini", "object": "model", "owned_by": "fake"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        body = await request.json()
        requests.append(body)
        messages = body.get("messages", [])

        if _has_tool_result(messages) or _has_react_observation(messages):
            return _text_response(body, "final answer using search results")

        requested_name, arguments = _requested_call(messages)
        tools = body.get("tools") or []
        if tools:
            _assert_requested_schema(tools, requested_name, arguments)
            return _tool_call_response(body, requested_name, arguments)

        prompt = " ".join(_message_text(message) for message in messages).lower()
        if "thought:" in prompt or "current task:" in prompt:
            return _text_response(
                body,
                "Thought: I should use the requested tool.\n"
                f"Action: {requested_name}\n"
                f"Action Input: {json.dumps(arguments)}",
            )
        return _text_response(body, "final answer using search results")

    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 5
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("Timed out waiting for fake OpenAI server to start.")
        time.sleep(0.01)

    yield FakeOpenAIServer(url=f"http://127.0.0.1:{port}", requests=requests)

    server.should_exit = True
    thread.join(timeout=5)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _requested_call(messages: list[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    text = "\n".join(_message_text(message) for message in messages)
    matches = re.findall(r"SERPAPI_TOOL_CALL:\s*(\{[^\n]+\})", text)
    if not matches:
        raise AssertionError("Prompt is missing a SERPAPI_TOOL_CALL JSON marker.")
    payload = json.loads(matches[-1])
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict):
        raise AssertionError(f"Invalid SERPAPI_TOOL_CALL marker: {payload!r}")
    return name, arguments


def _assert_requested_schema(
    tools: list[Mapping[str, Any]],
    requested_name: str,
    arguments: Mapping[str, Any],
) -> None:
    schemas: dict[str, Mapping[str, Any]] = {}
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            schemas[function["name"]] = function.get("parameters", {})
        elif isinstance(tool.get("name"), str):
            schemas[tool["name"]] = tool.get("parameters", tool.get("input_schema", {}))

    if requested_name not in schemas:
        raise AssertionError(
            f"Requested tool {requested_name!r} was not serialized. Available: {sorted(schemas)}"
        )
    schema = schemas[requested_name]
    if schema.get("additionalProperties") not in {None, False}:
        raise AssertionError(f"Tool {requested_name!r} must not serialize an open object schema.")
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    missing_properties = set(arguments).difference(properties)
    if missing_properties:
        raise AssertionError(
            f"Tool {requested_name!r} schema is missing argument properties: "
            f"{sorted(missing_properties)}"
        )
    missing_required = set(schema.get("required", [])).difference(arguments)
    if missing_required:
        raise AssertionError(
            f"Tool call for {requested_name!r} omits required fields: {sorted(missing_required)}"
        )


def _has_tool_result(messages: list[Mapping[str, Any]]) -> bool:
    return any(message.get("role") in {"tool", "function"} for message in messages)


def _has_react_observation(messages: list[Mapping[str, Any]]) -> bool:
    user_messages = [
        _message_text(message).lower() for message in messages if message.get("role") == "user"
    ]
    if not user_messages:
        return False
    latest = user_messages[-1]
    return "observation:" in latest and "serpapi_tool_call:" in latest


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return ""


def _tool_call_response(
    body: Mapping[str, Any],
    tool_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": "chatcmpl_fake_tool",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o-mini"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{tool_name}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _text_response(body: Mapping[str, Any], text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl_fake_final",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "gpt-4o-mini"),
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
