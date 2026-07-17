from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from collections.abc import Iterable
from typing import Any
from unittest.mock import patch

from optional_dependencies import import_optional

from serpapi_search_tools import (
    flights_search,
    hotels_search,
    images_search,
    maps_search,
    news_search,
    shopping_search,
    travel_explore_search,
    videos_search,
    web_search,
)

SUPPORTED_AGENT_PROVIDERS: tuple[str, ...] = (
    "openai-agents",
    "pydantic-ai",
    "langchain",
    "langgraph",
    "agno",
    "smolagents",
    "crewai",
    "autogen",
    "haystack",
    "llamaindex",
    "google-adk",
)

PROVIDER_IMPORTS = {
    "openai-agents": "agents",
    "pydantic-ai": "pydantic_ai",
    "langchain": "langchain.agents",
    "langgraph": "langgraph",
    "agno": "agno.agent",
    "smolagents": "smolagents",
    "crewai": "crewai",
    "autogen": "autogen_agentchat.agents",
    "haystack": "haystack.components.agents",
    "llamaindex": "llama_index.core.agent.workflow",
    "google-adk": "google.adk.agents",
}

TOP_AGENT_PROVIDERS: tuple[str, ...] = (
    "openai-agents",
    "pydantic-ai",
    "langchain",
)


def prompt_for_tool(name: str, arguments: dict[str, Any]) -> str:
    marker = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
    return (
        f"Call the {name} tool exactly once with the supplied arguments, then answer in one "
        f"short sentence. SERPAPI_TOOL_CALL: {marker}"
    )


def prompt_for_engine(engine: str, search_query: str = "coffee") -> str:
    """Compatibility helper while provider-specific integration cases migrate."""

    return prompt_for_tool("web_search", {"query": search_query, "engine": engine})


def run_agent_provider(
    provider: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None = None,
    serpapi_api_key: str | None = None,
) -> Any:
    if provider == "openai-agents":
        return _run_openai_agents(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    if provider == "pydantic-ai":
        return _run_pydantic_ai(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    if provider in {"langchain", "langgraph"}:
        return _run_langchain(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    if provider == "agno":
        return _run_agno(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    if provider == "smolagents":
        return _run_smolagents(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    if provider == "crewai":
        return _run_crewai(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    if provider == "autogen":
        return _run_autogen(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    if provider == "haystack":
        return _run_haystack(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    if provider == "llamaindex":
        return _run_llamaindex(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    if provider == "google-adk":
        return _run_google_adk(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
    raise ValueError(f"Unsupported agent provider: {provider}")


def _tools(
    provider: str,
    *,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> list[Any]:
    factories = {
        "web_search": web_search,
        "news_search": news_search,
        "maps_search": maps_search,
        "images_search": images_search,
        "shopping_search": shopping_search,
        "videos_search": videos_search,
        "hotels_search": hotels_search,
        "flights_search": flights_search,
        "travel_explore_search": travel_explore_search,
    }
    kwargs: dict[str, Any] = {"provider": provider}
    if client is not None:
        kwargs["client"] = client
    if serpapi_api_key is not None:
        kwargs["api_key"] = serpapi_api_key
    return [factories[name](**kwargs) for name in tool_names]


def _google_adk_litellm_model(model: str) -> str:
    if "/" in model and model.split("/", 1)[0] in {
        "ai21",
        "anthropic",
        "azure",
        "azure_ai",
        "bedrock",
        "cohere",
        "databricks",
        "deepseek",
        "fireworks_ai",
        "groq",
        "mistral",
        "ollama",
        "ollama_chat",
        "openai",
        "together_ai",
        "vertex_ai",
    }:
        return model
    return f"openai/{model}"


def _run_openai_agents(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    agents = import_optional("agents")
    openai = import_optional("openai")

    chat_model = agents.OpenAIChatCompletionsModel(
        model=model,
        openai_client=openai.AsyncOpenAI(api_key=api_key, base_url=base_url),
    )
    agent = agents.Agent(
        name="search-agent",
        instructions="Use the search tool when the user asks for search.",
        model=chat_model,
        tools=_tools(
            "openai-agents",
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        ),
    )

    async def run_agent() -> Any:
        return await agents.Runner.run(
            agent,
            prompt,
            run_config=agents.RunConfig(tracing_disabled=True),
        )

    return asyncio.run(run_agent())


def _run_pydantic_ai(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    pydantic_ai = import_optional("pydantic_ai")
    openai_models = import_optional("pydantic_ai.models.openai")
    openai_providers = import_optional("pydantic_ai.providers.openai")

    provider = openai_providers.OpenAIProvider(api_key=api_key, base_url=base_url)
    chat_model = openai_models.OpenAIChatModel(model, provider=provider)
    agent = pydantic_ai.Agent(
        chat_model,
        instructions="Use the web_search tool when the user asks for search.",
        tools=_tools(
            "pydantic-ai",
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        ),
    )
    return agent.run_sync(prompt)


def _run_langchain(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    agents = import_optional("langchain.agents")
    langchain_openai = import_optional("langchain_openai")

    chat_model = langchain_openai.ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
    agent = agents.create_agent(
        model=chat_model,
        tools=_tools(
            provider,
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        ),
    )
    return agent.invoke({"messages": [{"role": "user", "content": prompt}]})


def _run_agno(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    agno_agent = import_optional("agno.agent")
    agno_openai = import_optional("agno.models.openai")

    chat_model = agno_openai.OpenAIChat(id=model, api_key=api_key, base_url=base_url)
    agent = agno_agent.Agent(
        model=chat_model,
        instructions="Use tools for search.",
        tools=_tools(
            "agno",
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        ),
        telemetry=False,
    )
    return agent.run(prompt)


def _run_smolagents(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    smolagents = import_optional("smolagents")

    chat_model = smolagents.OpenAIServerModel(
        model_id=model,
        api_base=base_url,
        api_key=api_key,
        client_kwargs={"timeout": 120, "max_retries": 0},
    )
    agent = smolagents.ToolCallingAgent(
        model=chat_model,
        tools=_tools(
            "smolagents",
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        ),
        max_steps=3,
    )
    return agent.run(prompt)


def _run_crewai(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    with patch.dict(
        os.environ,
        {
            "CREWAI_TESTING": "true",
            "CREWAI_DISABLE_TELEMETRY": "true",
        },
    ):
        crewai = import_optional("crewai")

        llm = crewai.LLM(model=model, api_key=api_key, base_url=base_url, provider="openai")
        tools = _tools(
            "crewai",
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        )
        for tool in tools:
            if hasattr(tool, "result_as_answer"):
                tool.result_as_answer = True
        agent = crewai.Agent(
            role="Search agent",
            goal="Search with SerpApi",
            backstory="You use tools when asked to search.",
            llm=llm,
            tools=tools,
            max_iter=2,
            verbose=False,
        )
        task = crewai.Task(
            description=prompt,
            expected_output="A concise answer.",
            agent=agent,
        )
        crew = crewai.Crew(agents=[agent], tasks=[task], verbose=False, tracing=False)
        return crew.kickoff()


def _run_autogen(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    autogen_agents = import_optional("autogen_agentchat.agents")
    autogen_openai = import_optional("autogen_ext.models.openai")

    async def run_agent() -> Any:
        model_client = autogen_openai.OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "family": "unknown",
                "structured_output": False,
            },
        )
        try:
            agent = autogen_agents.AssistantAgent(
                "search_agent",
                model_client=model_client,
                tools=_tools(
                    "autogen",
                    tool_names=tool_names,
                    client=client,
                    serpapi_api_key=serpapi_api_key,
                ),
                reflect_on_tool_use=True,
            )
            return await agent.run(task=prompt)
        finally:
            await model_client.close()

    return asyncio.run(run_agent())


def _run_haystack(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    haystack_agents = import_optional("haystack.components.agents")
    haystack_generators = import_optional("haystack.components.generators.chat")
    haystack_dataclasses = import_optional("haystack.dataclasses")
    haystack_auth = import_optional("haystack.utils.auth")

    generator = haystack_generators.OpenAIChatGenerator(
        api_key=haystack_auth.Secret.from_token(api_key),
        model=model,
        api_base_url=base_url,
    )
    agent = haystack_agents.Agent(
        chat_generator=generator,
        tools=_tools(
            "haystack",
            tool_names=tool_names,
            client=client,
            serpapi_api_key=serpapi_api_key,
        ),
        max_agent_steps=3,
    )
    return agent.run(messages=[haystack_dataclasses.ChatMessage.from_user(prompt)])


def _run_llamaindex(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    llama_agent = import_optional("llama_index.core.agent.workflow")
    llama_core_llms = import_optional("llama_index.core.llms")
    llama_openai = import_optional("llama_index.llms.openai")

    class OpenAICompatible(llama_openai.OpenAI):
        @property
        def _tokenizer(self) -> Any | None:
            return None

        @property
        def metadata(self) -> Any:
            return llama_core_llms.LLMMetadata(
                context_window=8192,
                num_output=-1,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
                system_role=llama_core_llms.MessageRole.SYSTEM,
            )

    async def run_agent() -> Any:
        llm = OpenAICompatible(
            model=model,
            api_key=api_key,
            api_base=base_url,
            temperature=0,
        )
        agent = llama_agent.FunctionAgent(
            tools=_tools(
                "llamaindex",
                tool_names=tool_names,
                client=client,
                serpapi_api_key=serpapi_api_key,
            ),
            llm=llm,
            initial_tool_choice="required",
            streaming=False,
            timeout=120,
        )
        return await agent.run(prompt)

    return asyncio.run(run_agent())


def _run_google_adk(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    tool_names: Iterable[str],
    client: Any | None,
    serpapi_api_key: str | None,
) -> Any:
    adk_agents = import_optional("google.adk.agents")
    adk_runners = import_optional("google.adk.runners")
    adk_sessions = import_optional("google.adk.sessions")
    genai_types = import_optional("google.genai.types")

    if importlib.util.find_spec("litellm") is not None:
        adk_lite_llm = import_optional("google.adk.models.lite_llm")
        chat_model = adk_lite_llm.LiteLlm(
            model=_google_adk_litellm_model(model),
            api_base=base_url,
            api_key=api_key,
            temperature=0,
            max_tokens=256,
        )
    else:
        adk_base_llm = import_optional("google.adk.models.base_llm")
        adk_llm_response = import_optional("google.adk.models.llm_response")

        class ScriptedToolCallingLlm(adk_base_llm.BaseLlm):
            """Minimal deterministic ADK model for interpreter compatibility tests."""

            async def generate_content_async(self, llm_request: Any, stream: bool = False) -> Any:
                del stream
                parts = [part for content in llm_request.contents for part in (content.parts or [])]
                if any(getattr(part, "function_response", None) for part in parts):
                    yield adk_llm_response.LlmResponse(
                        content=genai_types.Content(
                            role="model",
                            parts=[
                                genai_types.Part.from_text(text="The requested search completed.")
                            ],
                        )
                    )
                    return

                prompt_text = "\n".join(part.text for part in parts if getattr(part, "text", None))
                marker = prompt_text.rsplit("SERPAPI_TOOL_CALL: ", maxsplit=1)[1]
                requested_call = json.loads(marker)
                yield adk_llm_response.LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[
                            genai_types.Part.from_function_call(
                                name=requested_call["name"],
                                args=requested_call["arguments"],
                            )
                        ],
                    )
                )

        chat_model = ScriptedToolCallingLlm(model=model)

    async def run_agent() -> Any:
        agent = adk_agents.Agent(
            name="search_agent",
            model=chat_model,
            instruction="Use the web_search tool when the user asks for search.",
            tools=_tools(
                "google-adk",
                tool_names=tool_names,
                client=client,
                serpapi_api_key=serpapi_api_key,
            ),
        )
        app_name = "serpapi_search_tools_tests"
        user_id = "test-user"
        session_id = "test-session"
        session_service = adk_sessions.InMemorySessionService()
        maybe_session = session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if asyncio.iscoroutine(maybe_session):
            await maybe_session

        runner = adk_runners.Runner(
            agent=agent,
            app_name=app_name,
            session_service=session_service,
        )
        new_message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )
        final_event = None
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message,
        ):
            if event.is_final_response():
                final_event = event
        return final_event

    return asyncio.run(run_agent())
