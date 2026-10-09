# TechHaven Customer Support AI (Demo)

Single CrewAI agent, Gemini, Streamlit, FAISS RAG, simulated Excel orders and human escalation tickets.

## GitHub → Streamlit Cloud (no local test required)

1. Download and unzip this project; upload the **contents** of the project folder to the root of a new GitHub repository.
2. In Streamlit Community Cloud, select the repository, branch `main`, and main file `app.py`.
3. Open **Advanced settings → Secrets** and paste the following, replacing placeholders:

```toml
GEMINI_API_KEY = "your-real-api-key"
GEMINI_MODEL = "gemini-2.5-flash-lite"
ADMIN_PASSWORD = "choose-a-long-unique-password"
```

4. Deploy. On first startup the embedding model downloads and creates `faiss.index` from `chunks.json` if missing. This can take several minutes. The model ID is configurable; use only a model available to your Gemini API account.
5. Ask a policy question, or test order `ORD-2026-1001` with the corresponding phone number from `orders.xlsx`. Ask for a human to create a pending ticket.

## Files

- `app.py` — Streamlit + single CrewAI agent and three tools
- `chunks.json` — 7 fictional company-policy chunks with metadata
- `metadata.json` — model, dimension and index configuration
- `orders.xlsx` — 60 fictional orders
- `knowledge/*.txt` — original fictional source documents
- `requirements.txt` — deployment dependencies
- `.streamlit/secrets.toml.example` — secrets template (never commit real secrets)
- `build_index.py` — optional standalone index builder

## Index

`faiss.index` is **generated automatically on first deployment**, not included in this ZIP. It must be built with the exact `sentence-transformers/all-MiniLM-L6-v2` model (384 dimensions). Do not substitute random vectors. If replacing the supplied chunks with your own knowledge, also replace the index and update metadata. An existing index is validated for dimension and chunk count.

## Limitations (important)

- Tickets are **session-only**, not persistent, not visible across different sessions, and not actually delivered to a support team. Add Supabase/Postgres or another persistent ticket service before using this with real customers.
- Orders are fictional. The Excel file is a read-only demo database. Do not commit real customer phone numbers or addresses to a public repository.
- Session chat history disappears after Streamlit session reset; no cross-device memory.
- `requirements.txt` specifies compatible major-version ranges, not a tested lockfile. API/library changes may require adjustments.
- FAISS similarity retrieval is approximate semantic matching; low relevance or missing facts should trigger escalation.
- Streamlit Cloud's free resources may be constrained by model downloads and CrewAI dependencies.
- Avoid publishing an admin dashboard containing customer data without proper authentication, database authorization, and audit logging.
