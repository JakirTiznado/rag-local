"""Fábrica de modelos de lenguaje y embeddings.

La importación de cada integración se hace de forma perezosa: un despliegue
100 % Ollama no necesita cargar los SDK de OpenAI o Gemini, y los errores de
dependencias faltantes se convierten en mensajes accionables.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import SecretStr

from app.config import Provider, Settings
from app.exceptions import ConfigurationError


class ModelFactory:
    """Crea LLMs y embeddings según las variables de entorno.

    Attributes:
        _settings: Configuración validada de la aplicación.
    """

    def __init__(self, settings: Settings) -> None:
        """Inicializa la fábrica.

        Args:
            settings: Configuración validada de la aplicación.
        """
        self._settings: Settings = settings

    def create_llm(self, *, temperature: float | None = None) -> BaseChatModel:
        """Instancia el modelo de chat configurado.

        Args:
            temperature: Temperatura explícita. Si es ``None`` se usa la de
                ``Settings`` (0.1 por defecto).

        Returns:
            Un ``BaseChatModel`` compatible con LCEL.

        Raises:
            ConfigurationError: Si el proveedor no está soportado, falta una
                credencial o falta la dependencia de integración.
        """
        settings = self._settings
        provider: Provider = settings.llm_provider

        # En RAG la fidelidad al contexto importa más que la creatividad.
        # Una temperatura baja (0.1) reduce respuestas inventadas y hace que
        # las citas sean más estables entre ejecuciones.
        effective_temperature = (
            settings.temperature if temperature is None else temperature
        )
        if not 0.0 <= effective_temperature <= 2.0:
            raise ConfigurationError(
                "La temperatura debe estar entre 0.0 y 2.0; "
                f"se recibió {effective_temperature}."
            )

        if provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:  # pragma: no cover - depende del entorno
                raise ConfigurationError(
                    "Falta 'langchain-ollama'. Instálalo con: "
                    "pip install langchain-ollama"
                ) from exc
            return ChatOllama(
                model=settings.llm_model,
                base_url=settings.ollama_base_url,
                temperature=effective_temperature,
                # El contexto por defecto de Ollama (2048) puede truncar el
                # prompt RAG + historial. 4096 deja margen para varios chunks
                # de 500 caracteres sin disparar el uso de memoria.
                num_ctx=settings.ollama_num_ctx,
            )

        if provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:  # pragma: no cover - depende del entorno
                raise ConfigurationError(
                    "Falta 'langchain-openai'. Instálalo con: "
                    "pip install langchain-openai"
                ) from exc
            return ChatOpenAI(
                model=settings.llm_model,
                api_key=self._require_secret(settings.openai_api_key, "OPENAI_API_KEY"),
                temperature=effective_temperature,
            )

        if provider == "gemini":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:  # pragma: no cover - depende del entorno
                raise ConfigurationError(
                    "Falta 'langchain-google-genai'. Instálalo con: "
                    "pip install langchain-google-genai"
                ) from exc
            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=self._require_secret(
                    settings.google_api_key, "GOOGLE_API_KEY"
                ).get_secret_value(),
                temperature=effective_temperature,
            )

        raise ConfigurationError(f"Proveedor de LLM no soportado: {provider!r}.")

    def create_embeddings(self) -> Embeddings:
        """Instancia el modelo de embeddings configurado.

        Returns:
            Un ``Embeddings`` compatible con LangChain.

        Raises:
            ConfigurationError: Si el proveedor no está soportado, falta una
                credencial o falta la dependencia de integración.
        """
        settings = self._settings
        provider: Provider = settings.resolved_embedding_provider

        if provider == "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings
            except ImportError as exc:  # pragma: no cover - depende del entorno
                raise ConfigurationError(
                    "Falta 'langchain-ollama'. Instálalo con: "
                    "pip install langchain-ollama"
                ) from exc
            return OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
            )

        if provider == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError as exc:  # pragma: no cover - depende del entorno
                raise ConfigurationError(
                    "Falta 'langchain-openai'. Instálalo con: "
                    "pip install langchain-openai"
                ) from exc
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=self._require_secret(settings.openai_api_key, "OPENAI_API_KEY"),
            )

        if provider == "gemini":
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
            except ImportError as exc:  # pragma: no cover - depende del entorno
                raise ConfigurationError(
                    "Falta 'langchain-google-genai'. Instálalo con: "
                    "pip install langchain-google-genai"
                ) from exc
            return GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=self._require_secret(
                    settings.google_api_key, "GOOGLE_API_KEY"
                ),
            )

        raise ConfigurationError(
            f"Proveedor de embeddings no soportado: {provider!r}."
        )

    @staticmethod
    def _require_secret(value: SecretStr | None, env_var: str) -> SecretStr:
        """Valida un secreto y lo devuelve sin exponerlo como texto plano.

        Args:
            value: Secreto opcional.
            env_var: Nombre de la variable de entorno, usado en el error.

        Returns:
            El ``SecretStr`` validado, listo para pasarlo al SDK.

        Raises:
            ConfigurationError: Si el secreto no existe o está vacío.
        """
        if value is None or not value.get_secret_value().strip():
            raise ConfigurationError(
                f"Falta {env_var}. Defínela en el entorno o en el archivo .env."
            )
        return value
