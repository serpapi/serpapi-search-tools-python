# Haystack weekly industry newsletter

This Haystack agent searches, verifies, and deduplicates current industry
stories before writing a compact newsletter to
`cookbook-output/haystack-industry-newsletter.md`.

## Original source

Inspired by the official Haystack Cookbook
[newsletter agent](https://github.com/deepset-ai/haystack-cookbook/blob/main/notebooks/newsletter-agent.ipynb).

This adaptation was **enhanced with SerpApi** by providing separate typed news
and web tools so the editor can discover timely stories and verify their
background independently.

## Run

Set `SERPAPI_API_KEY` and `OPENAI_API_KEY` in the repository-root `.env`, then:

```bash
uv run --isolated --no-project --with 'serpapi-search-tools[haystack]' --with python-dotenv cookbook/haystack/main.py
```

Optional environment variables: `OPENAI_MODEL`, `COOKBOOK_PROMPT`, and
`COOKBOOK_OUTPUT_DIR`.

