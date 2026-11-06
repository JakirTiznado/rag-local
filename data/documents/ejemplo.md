# Guía de ejemplo: RAG local

Este documento existe para probar la ingestión y las citas del sistema.

## Componentes

El sistema combina tres piezas principales:

| Componente | Función |
| --- | --- |
| LangChain LCEL | Compone recuperación, prompt y generación con pipes. |
| Qdrant | Almacena embeddings y permite búsqueda por similitud de coseno. |
| Ollama | Ejecuta modelos locales de chat y embeddings. |

## Flujo de una consulta

1. El usuario escribe una pregunta en la CLI.
2. La pregunta se convierte en un vector con el modelo de embeddings.
3. Qdrant devuelve los fragmentos más similares.
4. Los fragmentos se insertan en el prompt dentro de `<contexto>`.
5. El LLM responde citando archivo, página y fragmento.

## Configuración de Qdrant

En modo `local_path`, Qdrant persiste en `./data/qdrant_storage` y no requiere
un servidor. En modo `server`, la URL por defecto es `http://localhost:6333`.

## Modelos recomendados para Ollama

- Chat: `llama3.2`
- Embeddings: `nomic-embed-text`

## Limitación conocida

Los PDF escaneados sin capa de texto requieren un paso previo de OCR.
