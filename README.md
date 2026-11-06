# RAG local/híbrido con LangChain, Qdrant y Ollama

Sistema RAG modular en Python 3.11+ con:

- **LangChain v0.3+ / LCEL** (`RunnableParallel`, `RunnableBranch`, pipes).
- **Qdrant** embebido en disco o como servidor HTTP, con distancia coseno.
- **Ollama** para LLM y embeddings locales.
- **Proveedores comerciales** opcionales (OpenAI / Gemini).
- **CLI** con Typer y salida Rich (Markdown, tablas y paneles).
- **Citas** de archivo, página y fragmento en cada respuesta.

## Estructura

```text
rag-local/
├── app/
│   ├── cli.py                 # Comandos Typer + Rich
│   ├── config.py              # Settings con pydantic-settings
│   ├── exceptions.py          # Errores de dominio
│   ├── core/
│   │   ├── factory.py         # ModelFactory (LLM + embeddings)
│   │   └── vector_store.py    # Cliente Qdrant y colección
│   └── services/
│       ├── ingestion.py       # PDF/TXT/MD → chunks → Qdrant
│       └── rag.py             # Cadena LCEL + citas
├── data/
│   ├── documents/             # Corpus de entrada
│   └── qdrant_storage/        # Persistencia local (se crea sola)
├── .env.example
├── requirements.txt
└── README.md
```

## 1. Instalación

```bash
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 2. Preparar Ollama

Instala Ollama desde <https://ollama.com/download> y descarga los modelos:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

Verifica que responde:

```bash
curl http://localhost:11434/api/tags
```

> `nomic-embed-text` genera vectores de 768 dimensiones. Si cambias de modelo
> de embeddings, reindexa con `--recreate`.

## 3. Qdrant

### Opción A — local en disco (por defecto)

No requiere Docker. En `.env`:

```env
QDRANT_MODE=local_path
QDRANT_PATH=./data/qdrant_storage
```

### Opción B — servidor Qdrant

```bash
docker run --rm -p 6333:6333 -p 6334:6334 \
  -v "${PWD}/data/qdrant_server:/qdrant/storage" \
  qdrant/qdrant
```

En `.env`:

```env
QDRANT_MODE=server
QDRANT_URL=http://localhost:6333
```

## 4. Ingestión

```bash
python -m app.cli ingest --path ./data/documents/
```

Formatos soportados: `.pdf`, `.txt`, `.md`, `.markdown`.

- Los PDF se procesan página a página extrayendo texto y tablas en Markdown.
- Chunks de 500 caracteres con 50 de solape.
- Metadatos conservados: `source`, `page`, `chunk_id`, `chunk_index`.

Reindexar desde cero (por cambio de modelo o de metadatos):

```bash
python -m app.cli ingest --path ./data/documents/ --recreate
```

## 5. Consultas

Pregunta puntual:

```bash
python -m app.cli ask "¿Cómo se configura Qdrant en modo servidor?"
```

Con más contexto recuperado:

```bash
python -m app.cli ask "¿Qué modelos recomienda la guía?" --top-k 6
```

Sesión interactiva:

```bash
python -m app.cli chat
```

Comandos dentro del chat: `/help`, `/clear`, `/exit`.

Estado de la configuración y la colección:

```bash
python -m app.cli info
```

## 6. Configuración principal

| Variable | Valores | Default |
| --- | --- | --- |
| `APP_ENV` | `local` / `cloud` | `local` |
| `LLM_PROVIDER` | `ollama` / `openai` / `gemini` | `ollama` |
| `EMBEDDING_PROVIDER` | igual que arriba (opcional) | usa `LLM_PROVIDER` |
| `OLLAMA_BASE_URL` | URL | `http://localhost:11434` |
| `OLLAMA_NUM_CTX` | entero (tokens) | `4096` |
| `LLM_MODEL` | nombre del modelo | `llama3.2` |
| `EMBEDDING_MODEL` | nombre del modelo | `nomic-embed-text` |
| `TEMPERATURE` | `0.0`–`2.0` | `0.1` |
| `TOP_K` | entero | `4` |
| `MIN_RELEVANCE_SCORE` | `-1.0`–`1.0` (opcional) | desactivado |
| `CHUNK_SIZE` | entero (caracteres) | `500` |
| `CHUNK_OVERLAP` | entero (caracteres) | `50` |
| `QDRANT_MODE` | `local_path` / `server` | `local_path` |
| `QDRANT_PATH` | ruta | `./data/qdrant_storage` |
| `QDRANT_URL` | URL | `http://localhost:6333` |
| `COLLECTION_NAME` | nombre | `rag_documents` |
| `MAX_HISTORY_TURNS` | entero | `6` |

### Usar OpenAI o Gemini

```env
# OpenAI para chat y embeddings
APP_ENV=cloud
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

```env
# Gemini para chat, embeddings locales de Ollama
APP_ENV=cloud
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
GOOGLE_API_KEY=...
```

## 7. Cómo funciona la generación

La cadena LCEL hace lo siguiente:

1. `RunnableParallel` calcula en paralelo la pregunta, el historial y los
   documentos recuperados por el retriever.
2. `.assign(context=...)` serializa los fragmentos como bloques
   `<documento fuente="..." pagina="..." fragmento="...">`.
3. Si `MIN_RELEVANCE_SCORE` está definido, los fragmentos con similitud coseno
   inferior se descartan antes de construir el prompt.
4. `RunnableBranch` evita llamar al LLM si no hay documentos y devuelve una
   respuesta determinista de “sin contexto”.
5. El prompt obliga a citar `[Fuente: archivo, pág. N, fragmento ID]` y a no
   usar conocimiento externo.
6. La CLI muestra la respuesta en Markdown y una tabla con las fuentes.

Las citas se respaldan además con la tabla de fuentes, que se construye con los
metadatos reales devueltos por Qdrant (no con texto generado por el modelo).

## 8. Solución de problemas

| Síntoma | Causa / solución |
| --- | --- |
| `La colección 'rag_documents' no existe` | Ejecuta primero `ingest`. |
| `Connection refused` contra Ollama | Inicia `ollama serve` y revisa `OLLAMA_BASE_URL`. |
| HTTP 501 `does not support embeddings` | Tu servidor Ollama tiene el endpoint de embeddings deshabilitado o usa un modelo generativo. Actualiza/reinicia Ollama con soporte de embeddings y usa `nomic-embed-text`. |
| Error de dimensión al indexar | Cambiaste de modelo de embeddings: usa `--recreate`. |
| Qdrant local bloqueado | Otro proceso (chat/ask) tiene abierto el mismo `QDRANT_PATH`. |
| PDF sin texto | Es un escaneo; aplica OCR antes de indexarlo. |
| Respuesta “No encuentro esa información…” | El corpus no contiene el dato o `TOP_K` es muy bajo. |

## 9. Limitaciones actuales

- La búsqueda es densa (similitud coseno); no hay búsqueda híbrida densa+sparse.
- El historial del chat no reescribe la consulta de recuperación; sirve para
  resolver referencias conversacionales, no para reformular preguntas complejas.
- La extracción de tablas depende de la detección de PyMuPDF; tablas muy
  irregulares pueden requerir OCR/parseo especializado.
- Qdrant en `local_path` está pensado para un único proceso; para concurrencia,
  usa `QDRANT_MODE=server`.
