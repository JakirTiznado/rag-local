"""Configuración centralizada de la aplicación.

Las variables se cargan desde el entorno del proceso y, si existe, desde el
archivo ``.env`` del directorio de trabajo. ``pydantic-settings`` se encarga
del tipado, los valores por defecto y la validación cruzada.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Proveedores de modelos soportados por :class:`app.core.factory.ModelFactory`.
Provider = Literal["ollama", "openai", "gemini"]

#: Modos de conexión con Qdrant.
QdrantMode = Literal["local_path", "server"]

#: Entornos lógicos admitidos.
AppEnv = Literal["local", "cloud"]


def _is_secret_set(value: SecretStr | None) -> bool:
    """Comprueba si un secreto tiene contenido útil.

    Args:
        value: Secreto opcional cargado desde la configuración.

    Returns:
        ``True`` si el secreto existe y no está vacío tras aplicar ``strip``.
    """
    return value is not None and bool(value.get_secret_value().strip())


class Settings(BaseSettings):
    """Configuración tipada del sistema RAG.

    Attributes:
        app_env: Entorno lógico de ejecución (``local`` o ``cloud``).
        llm_provider: Proveedor del modelo de chat.
        embedding_provider: Proveedor de embeddings. Si es ``None`` se reutiliza
            ``llm_provider``, lo que permite combinar un LLM comercial con
            embeddings locales (o viceversa) mediante ``EMBEDDING_PROVIDER``.
        ollama_base_url: URL del servidor Ollama.
        ollama_num_ctx: Ventana de contexto de Ollama en tokens.
        llm_model: Identificador del modelo de chat.
        embedding_model: Identificador del modelo de embeddings.
        temperature: Temperatura del LLM; baja por defecto para priorizar
            fidelidad al contexto recuperado.
        top_k: Número de fragmentos recuperados por consulta.
        min_relevance_score: Umbral mínimo de similitud coseno; si es ``None``
            no se filtra por score.
        chunk_size: Tamaño objetivo de cada fragmento, en caracteres.
        chunk_overlap: Solape entre fragmentos consecutivos, en caracteres.
        qdrant_mode: ``local_path`` para Qdrant embebido en disco o ``server``
            para un servidor Qdrant por red.
        qdrant_path: Ruta de persistencia local de Qdrant.
        qdrant_url: URL del servidor Qdrant.
        qdrant_api_key: API key opcional para Qdrant Cloud o servidores
            protegidos.
        qdrant_timeout: Timeout de red de Qdrant, en segundos.
        collection_name: Nombre de la colección vectorial.
        openai_api_key: API key de OpenAI cuando se usa ese proveedor.
        google_api_key: API key de Google Gemini cuando se usa ese proveedor.
        max_history_turns: Turnos de conversación recordados en modo ``chat``.
        log_level: Nivel de logging de la aplicación.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: AppEnv = Field(default="local", description="Entorno lógico de ejecución.")
    llm_provider: Provider = Field(default="ollama", description="Proveedor del LLM de chat.")
    embedding_provider: Provider | None = Field(
        default=None,
        description="Proveedor de embeddings; si se omite, se usa LLM_PROVIDER.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL base del servidor Ollama.",
    )
    ollama_num_ctx: int = Field(
        default=4096,
        ge=512,
        description="Ventana de contexto de Ollama en tokens.",
    )
    llm_model: str = Field(default="llama3.2", description="Modelo de chat.")
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Modelo de embeddings.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperatura del LLM (0.0 = determinista).",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=50,
        description="Fragmentos recuperados por consulta.",
    )
    min_relevance_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Umbral mínimo de similitud coseno; None desactiva el filtro.",
    )
    chunk_size: int = Field(
        default=500,
        ge=100,
        description="Tamaño de chunk en caracteres.",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Solape entre chunks en caracteres.",
    )
    qdrant_mode: QdrantMode = Field(
        default="local_path",
        description="Modo de conexión con Qdrant.",
    )
    qdrant_path: str = Field(
        default="./data/qdrant_storage",
        description="Ruta de persistencia local de Qdrant.",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="URL del servidor Qdrant.",
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="API key opcional de Qdrant.",
    )
    qdrant_timeout: int = Field(
        default=30,
        gt=0,
        description="Timeout de red de Qdrant en segundos.",
    )
    collection_name: str = Field(
        default="rag_documents",
        min_length=1,
        description="Nombre de la colección de Qdrant.",
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="API key de OpenAI.",
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        description="API key de Google Gemini.",
    )
    max_history_turns: int = Field(
        default=6,
        ge=0,
        le=50,
        description="Turnos de historial conservados en el chat.",
    )
    log_level: str = Field(default="INFO", description="Nivel de logging.")

    @property
    def resolved_embedding_provider(self) -> Provider:
        """Devuelve el proveedor de embeddings efectivo.

        Returns:
            ``embedding_provider`` si está definido; en caso contrario,
            ``llm_provider``.
        """
        return self.embedding_provider or self.llm_provider

    @property
    def qdrant_location(self) -> str:
        """Devuelve la ruta o URL efectiva de Qdrant.

        Returns:
            La ruta local o la URL del servidor según ``qdrant_mode``.
        """
        return self.qdrant_path if self.qdrant_mode == "local_path" else self.qdrant_url

    def resolved_qdrant_path(self) -> Path:
        """Resuelve la ruta local de Qdrant a un path absoluto.

        Returns:
            Ruta absoluta expandida para el almacenamiento embebido.
        """
        return Path(self.qdrant_path).expanduser().resolve()

    @model_validator(mode="after")
    def _validate_configuration(self) -> Self:
        """Valida combinaciones de configuración que podrían fallar en runtime.

        Returns:
            La propia instancia validada.

        Raises:
            ValueError: Si el solape es mayor o igual que el tamaño de chunk,
                faltan credenciales para un proveedor comercial o la URL de
                Qdrant está vacía en modo servidor.
        """
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP debe ser menor que CHUNK_SIZE.")

        if self.llm_provider == "openai" and not _is_secret_set(self.openai_api_key):
            raise ValueError("OPENAI_API_KEY es obligatoria cuando LLM_PROVIDER=openai.")
        if self.llm_provider == "gemini" and not _is_secret_set(self.google_api_key):
            raise ValueError("GOOGLE_API_KEY es obligatoria cuando LLM_PROVIDER=gemini.")

        embedding_provider = self.resolved_embedding_provider
        if embedding_provider == "openai" and not _is_secret_set(self.openai_api_key):
            raise ValueError(
                "OPENAI_API_KEY es obligatoria cuando EMBEDDING_PROVIDER=openai."
            )
        if embedding_provider == "gemini" and not _is_secret_set(self.google_api_key):
            raise ValueError(
                "GOOGLE_API_KEY es obligatoria cuando EMBEDDING_PROVIDER=gemini."
            )

        if self.qdrant_mode == "server" and not self.qdrant_url.strip():
            raise ValueError("QDRANT_URL es obligatoria cuando QDRANT_MODE=server.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Carga y cachea la configuración de la aplicación.

    Returns:
        Instancia única de :class:`Settings` para el proceso actual.

    Raises:
        pydantic.ValidationError: Si alguna variable de entorno no cumple el
            esquema o falta una credencial requerida.
    """
    return Settings()
