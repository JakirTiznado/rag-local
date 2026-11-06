"""Gestión del cliente Qdrant y del almacén vectorial de LangChain.

Se crea la colección de forma explícita para fijar la distancia de coseno y el
tamaño real del embedding antes de indexar. Así se evita depender de los
valores por defecto de la integración y se obtienen errores claros cuando el
modelo de embeddings cambia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import SecretStr
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import Settings
from app.exceptions import CollectionNotFoundError, ConfigurationError


@dataclass(frozen=True, slots=True)
class CollectionStats:
    """Resumen del estado de una colección de Qdrant.

    Attributes:
        exists: Indica si la colección existe.
        points_count: Número de puntos indexados, si está disponible.
        status: Estado reportado por Qdrant (por ejemplo ``green``).
        vector_size: Dimensión del vector configurado.
    """

    exists: bool
    points_count: int | None = None
    status: str | None = None
    vector_size: int | None = None


class VectorStoreFactory:
    """Construye clientes y almacenes ``QdrantVectorStore``.

    Attributes:
        _settings: Configuración validada de la aplicación.
    """

    def __init__(self, settings: Settings) -> None:
        """Inicializa la fábrica de almacenamiento vectorial.

        Args:
            settings: Configuración validada de la aplicación.
        """
        self._settings: Settings = settings

    def create_client(self) -> QdrantClient:
        """Crea un cliente Qdrant local o remoto según la configuración.

        Returns:
            Cliente sincrónico de Qdrant.

        Raises:
            ConfigurationError: Si no se puede preparar la ruta local o crear
                el cliente del servidor.
        """
        settings = self._settings

        if settings.qdrant_mode == "local_path":
            path = settings.resolved_qdrant_path()
            try:
                path.mkdir(parents=True, exist_ok=True)
                return QdrantClient(
                    path=str(path),
                    timeout=settings.qdrant_timeout,
                )
            except Exception as exc:
                raise ConfigurationError(
                    "No se pudo abrir Qdrant en modo local_path "
                    f"({path}): {exc}"
                ) from exc

        try:
            return QdrantClient(
                url=settings.qdrant_url,
                api_key=self._optional_secret(settings.qdrant_api_key),
                timeout=settings.qdrant_timeout,
            )
        except Exception as exc:
            raise ConfigurationError(
                "No se pudo crear el cliente Qdrant en modo server "
                f"({settings.qdrant_url}): {exc}"
            ) from exc

    def ensure_collection(
        self,
        client: QdrantClient,
        embeddings: Embeddings,
        *,
        recreate: bool = False,
    ) -> None:
        """Garantiza que la colección exista con distancia de coseno.

        Args:
            client: Cliente Qdrant ya inicializado.
            embeddings: Modelo usado para calcular la dimensión del vector.
            recreate: Si es ``True``, elimina la colección existente antes de
                crearla de nuevo. Útil al cambiar de modelo de embeddings.

        Raises:
            ConfigurationError: Si falla la comunicación con Qdrant, no se
                puede calcular la dimensión o la creación es rechazada.
        """
        name = self._settings.collection_name

        try:
            exists = client.collection_exists(collection_name=name)
        except Exception as exc:
            raise ConfigurationError(
                f"No se pudo comprobar la colección '{name}': {exc}"
            ) from exc

        vector_size: int | None = None
        if recreate and exists:
            # Se calcula la dimensión ANTES de borrar: si el proveedor de
            # embeddings está caído, no se destruyen los datos existentes.
            vector_size = self._embedding_dimension(embeddings)
            try:
                client.delete_collection(collection_name=name)
            except Exception as exc:
                raise ConfigurationError(
                    f"No se pudo recrear la colección '{name}': {exc}"
                ) from exc
            exists = False

        if exists:
            # Si la colección ya existe, se comprueba que su dimensión coincida
            # con el modelo actual. Sin esto, un cambio de embedding produciría
            # búsquedas silenciosamente incorrectas o un error opaco al indexar.
            self._validate_embedding_dimension(client, embeddings)
            return

        if vector_size is None:
            vector_size = self._embedding_dimension(embeddings)

        try:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise ConfigurationError(
                f"No se pudo crear la colección '{name}' "
                f"(dimensión {vector_size}): {exc}"
            ) from exc

    def build_store(
        self,
        client: QdrantClient,
        embeddings: Embeddings,
        *,
        validate_dimension: bool = True,
    ) -> QdrantVectorStore:
        """Abre un almacén vectorial sobre una colección existente.

        Args:
            client: Cliente Qdrant ya inicializado.
            embeddings: Modelo de embeddings que se usará en las búsquedas.
            validate_dimension: Si es ``True``, verifica que la dimensión de la
                colección coincida con la del modelo. La ingestión lo desactiva
                porque ``ensure_collection`` ya hizo la validación.

        Returns:
            Instancia de ``QdrantVectorStore`` lista para recuperar.

        Raises:
            CollectionNotFoundError: Si la colección aún no fue creada.
            ConfigurationError: Si falla la consulta a Qdrant o la dimensión del
                embedding no coincide con la colección.
        """
        name = self._settings.collection_name
        try:
            exists = client.collection_exists(collection_name=name)
        except Exception as exc:
            raise ConfigurationError(
                f"No se pudo comprobar la colección '{name}': {exc}"
            ) from exc

        if not exists:
            raise CollectionNotFoundError(
                f"La colección '{name}' no existe. Ejecuta primero: "
                "python -m app.cli ingest --path ./data/documents/"
            )

        if validate_dimension:
            self._validate_embedding_dimension(client, embeddings)

        # validate_collection_config=False evita que QdrantVectorStore repita
        # la comprobación de dimensión que ya hicimos (y su embedding extra).
        return QdrantVectorStore(
            client=client,
            collection_name=name,
            embedding=embeddings,
            distance=Distance.COSINE,
            validate_collection_config=False,
        )

    def collection_stats(self, client: QdrantClient) -> CollectionStats:
        """Obtiene un resumen de la colección configurada.

        Args:
            client: Cliente Qdrant ya inicializado.

        Returns:
            Estado, número de puntos y dimensión de la colección. Si no existe,
            devuelve ``CollectionStats(exists=False)``.

        Raises:
            ConfigurationError: Si Qdrant no responde a la consulta.
        """
        name = self._settings.collection_name
        try:
            if not client.collection_exists(collection_name=name):
                return CollectionStats(exists=False)

            info = client.get_collection(collection_name=name)
        except Exception as exc:
            raise ConfigurationError(
                f"No se pudo consultar la colección '{name}': {exc}"
            ) from exc

        status_value = getattr(info.status, "value", str(info.status))
        return CollectionStats(
            exists=True,
            points_count=info.points_count,
            status=str(status_value).lower(),
            vector_size=self._extract_vector_size(info.config.params.vectors),
        )

    def _validate_embedding_dimension(
        self,
        client: QdrantClient,
        embeddings: Embeddings,
    ) -> None:
        """Comprueba que el modelo y la colección usen la misma dimensión.

        Args:
            client: Cliente Qdrant ya inicializado.
            embeddings: Modelo de embeddings configurado.

        Raises:
            ConfigurationError: Si la dimensión no coincide o no se puede
                determinar. El mensaje indica cómo reindexar.
        """
        collection_size = self._collection_vector_size(client)
        embedding_size = self._embedding_dimension(embeddings)
        if collection_size != embedding_size:
            name = self._settings.collection_name
            raise ConfigurationError(
                f"La colección '{name}' usa vectores de {collection_size} "
                f"dimensiones, pero el modelo '{self._settings.embedding_model}' "
                f"produce {embedding_size}. Reindexa con: "
                "python -m app.cli ingest --path ./data/documents/ --recreate"
            )

    def _collection_vector_size(self, client: QdrantClient) -> int:
        """Obtiene la dimensión configurada en la colección.

        Args:
            client: Cliente Qdrant ya inicializado.

        Returns:
            Tamaño del primer vector de la colección.

        Raises:
            ConfigurationError: Si Qdrant falla o la configuración no permite
                inferir la dimensión.
        """
        name = self._settings.collection_name
        try:
            info = client.get_collection(collection_name=name)
        except Exception as exc:
            raise ConfigurationError(
                f"No se pudo consultar la configuración de '{name}': {exc}"
            ) from exc

        vector_size = self._extract_vector_size(info.config.params.vectors)
        if vector_size is None:
            raise ConfigurationError(
                f"No se pudo determinar la dimensión de la colección '{name}'."
            )
        return vector_size

    @staticmethod
    def _embedding_dimension(embeddings: Embeddings) -> int:
        """Calcula la dimensión real del modelo de embeddings.

        Se incrusta una frase corta de prueba. Es una sola llamada por arranque
        y evita asumir dimensiones por proveedor o por nombre de modelo.

        Args:
            embeddings: Modelo de embeddings.

        Returns:
            Dimensión del vector generado.

        Raises:
            ConfigurationError: Si el proveedor falla o devuelve un vector
                vacío.
        """
        try:
            probe_vector = embeddings.embed_query("dimensión del embedding")
        except Exception as exc:
            raise ConfigurationError(
                "No se pudo calcular la dimensión del embedding. "
                "Verifica el modelo y la conexión con el proveedor: "
                f"{exc}"
            ) from exc

        vector_size = len(probe_vector)
        if vector_size <= 0:
            raise ConfigurationError(
                "El modelo de embeddings devolvió un vector vacío."
            )
        return vector_size

    @staticmethod
    def _extract_vector_size(vectors: Any) -> int | None:
        """Extrae la dimensión de una configuración de vectores de Qdrant.

        Args:
            vectors: Configuración nominal o con nombre devuelta por Qdrant.

        Returns:
            La dimensión del primer vector, o ``None`` si no se puede inferir.
        """
        if isinstance(vectors, VectorParams):
            return vectors.size
        if isinstance(vectors, dict):
            first = next(iter(vectors.values()), None)
            return getattr(first, "size", None)
        return getattr(vectors, "size", None)

    @staticmethod
    def _optional_secret(value: SecretStr | None) -> str | None:
        """Convierte un secreto opcional en texto o ``None``.

        Args:
            value: Secreto opcional.

        Returns:
            El valor sin espacios, o ``None`` si no está definido o está vacío.
        """
        if value is None:
            return None
        raw = value.get_secret_value().strip()
        return raw or None
