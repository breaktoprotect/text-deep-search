# 🐾 Teddy Search (`text-deep-search`)

**Teddy Search** is a local semantic search tool built for professionals who need to quickly extract insights from structured documents like `.xlsx`, `.csv`, and more.

Designed for enterprise users working with corpora of ~<10k records — ideal for cybersecurity, GRC, or other specialized data analysis needs.

---

## ⚙️ Optional: `.env` Overrides

You may create a `.env` file at the root of the project to override default behavior:

```shell
TEDDY_SEARCH_DEFAULT_MODEL="MiniLM-L6-v2"  # Must match one of the supported model keys
