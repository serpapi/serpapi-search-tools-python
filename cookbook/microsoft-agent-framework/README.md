# Microsoft Agent Framework technology due diligence

This agent discovers current warehouse-robotics developments, verifies material
claims, and writes a due-diligence memo to
`cookbook-output/microsoft-agent-framework-due-diligence.md`.

## Original source

Inspired by Microsoft Agent Framework's official Python
[tool-use example](https://github.com/microsoft/Agent-Framework-Samples/blob/main/00.ForBeginners/04-tool-use/code_samples/python-agent-framework-ghmodel-tools.ipynb).

This adaptation was **enhanced with SerpApi** by replacing the example's local
demonstration function with typed web and news search tools, then applying the
tool loop to a source-backed decision artifact.

Microsoft Agent Framework is the successor recommended by Microsoft for
AutoGen and Semantic Kernel agent applications.

## Run

Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --with '.[microsoft-agent-framework]' \
  cookbook/microsoft-agent-framework/main.py
```

Optional environment variables: `OPENAI_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.
