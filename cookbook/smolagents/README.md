# smolagents purchase research assistant

This ToolCallingAgent combines product listings, images, and video
demonstrations to prepare a purchase shortlist. It writes the result to
`cookbook-output/smolagents-purchase-research.md`.

## Original source

Inspired by smolagents'
[official multiple-tools example](https://github.com/huggingface/smolagents/blob/main/examples/multiple_tools.py).

This adaptation was **enhanced with SerpApi** by replacing demonstration tools
with typed shopping, image, and YouTube search, giving the agent three
complementary evidence surfaces for one purchase decision.

## Run

Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --isolated --no-project --with 'serpapi-search-tools[smolagents]' --with python-dotenv cookbook/smolagents/main.py
```

Optional environment variables: `OPENAI_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.

