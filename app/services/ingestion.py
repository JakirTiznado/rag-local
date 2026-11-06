"""Ingestión de documentos PDF, TXT y Markdown.

El flujo es: descubrir archivos, extraer texto (y tablas en PDF) por página,
fragmentar con ``RecursiveCharacterTextSplitter`` enriqueteciendo metadatos y
hacer *upsert* en Qdrant con identificadores deterministas. Los IDs estables
permiten reindexar sin duplicar puntos cuando el contenido no cambia.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from app.config import Settings
from app.core.vector_store import VectorStoreFactory
from app.exceptions import IngestionError

# PyMuPDF 1.28+ imprime una recomendación sobre `pymupdf_layout` la primera
# vez que se detectan tablas. Se desactiva para no contaminar la salida de la
# CLI; la detección nativa sigue funcionando. En versiones antiguas no existe.
_disable_layout_recommendation = getattr(pymupdf, "no_recommend_layout", None)
if callable(_disable_layout_recommendation):
    _disable_layout_recommendation()

#: Extensiones aceptadas por el servicio de ingestión.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md", ".markdown"})

#: Directorios que nunca se recorren durante el descubrimiento.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "qdrant_storage",
        "qdrant_server",
    }
)


@dataclass(frozen=True, slots=True)
class FileReport:
    """Resultado de procesar un archivo.

    Attributes:
        source: Ruta absoluta del archivo.
        documents: Unidades extraídas (normalmente páginas).
        chunks: Fragmentos generados.
        error: Mensaje de error si el archivo falló; ``None`` si fue exitoso.
    """

    source: Path
    documents: int
    chunks: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Indica si el archivo se procesó sin errores.

        Returns:
            ``True`` cuando ``error`` es ``None``.
        """
        return self.error is None


@dataclass(frozen=True, slots=True)
class IngestionReport:
    """Resumen de una ejecución de ingestión.

    Attributes:
        files: Resultados por archivo.
        total_chunks: Total de fragmentos indexados.
        elapsed_seconds: Duración total de la operación.
        collection_name: Colección de Qdrant utilizada.
    """

    files: tuple[FileReport, ...]
    total_chunks: int
    elapsed_seconds: float
    collection_name: str

    @property
    def successful_files(self) -> int:
        """Cuenta los archivos procesados correctamente.

        Returns:
            Número de archivos sin error.
        """
        return sum(1 for report in self.files if report.ok)

    @property
    def failed_files(self) -> int:
        """Cuenta los archivos que fallaron.

        Returns:
            Número de archivos con error.
        """
        return len(self.files) - self.successful_files


class IngestionService:
    """Extrae, fragmenta e indexa documentos en Qdrant.

    Attributes:
        _settings: Configuración validada.
        _embeddings: Modelo de embeddings usado para calcular la dimensión y
            vectorizar los fragmentos.
        _factory: Fábrica de clientes y almacenes Qdrant.
        _splitter: Divisor recursivo configurado con tamaño y solape.
    """

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings,
        vector_store_factory: VectorStoreFactory | None = None,
    ) -> None:
        """Inicializa el servicio de ingestión.

        Args:
            settings: Configuración validada de la aplicación.
            embeddings: Modelo de embeddings.
            vector_store_factory: Fábrica opcional; se crea una por defecto.
        """
        self._settings: Settings = settings
        self._embeddings: Embeddings = embeddings
        self._factory: VectorStoreFactory = (
            vector_store_factory or VectorStoreFactory(settings)
        )

        # RecursiveCharacterTextSplitter trabaja en caracteres. Se mantienen
        # separadores semánticos (párrafo, línea, frase) para que los cortes
        # no rompan ideas cuando sea evitable. 500/50 equilibra precisión de
        # citas (chunks pequeños) con contexto suficiente para el LLM.
        self._splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        )

    def ingest(self, path: Path, *, recreate: bool = False) -> IngestionReport:
        """Indexa un archivo o todos los documentos de un directorio.

        Args:
            path: Archivo o directorio de entrada.
            recreate: Si es ``True``, elimina la colección antes de indexar.
                Necesario al cambiar el modelo de embeddings o tras modificar
                el esquema de metadatos.

        Returns:
            Informe con el resultado por archivo y el total de fragmentos.

        Raises:
            IngestionError: Si la ruta no existe, no contiene archivos
                soportados o falla el *upsert* en Qdrant.
            ConfigurationError: Si Qdrant no está disponible o la colección no
                puede crearse.
        """
        started_at = time.perf_counter()
        files = self._discover_files(path)

        chunks: list[Document] = []
        reports: list[FileReport] = []

        for file_path in files:
            try:
                documents = self._load_file(file_path)
                file_chunks = self._chunk_documents(documents)
            except Exception as exc:  # noqa: BLE001 - se reporta por archivo
                # Un archivo corrupto no debe impedir indexar el resto del
                # corpus; el error se reporta de forma aislada.
                reports.append(
                    FileReport(
                        source=file_path,
                        documents=0,
                        chunks=0,
                        error=str(exc),
                    )
                )
                continue

            chunks.extend(file_chunks)
            reports.append(
                FileReport(
                    source=file_path,
                    documents=len(documents),
                    chunks=len(file_chunks),
                )
            )

        if chunks:
            self._index_chunks(chunks, recreate=recreate)

        elapsed = time.perf_counter() - started_at
        return IngestionReport(
            files=tuple(reports),
            total_chunks=len(chunks),
            elapsed_seconds=elapsed,
            collection_name=self._settings.collection_name,
        )

    def _index_chunks(self, chunks: Sequence[Document], *, recreate: bool) -> None:
        """Crea la colección si hace falta y hace *upsert* de los fragmentos.

        Args:
            chunks: Fragmentos enriquecidos con metadatos.
            recreate: Si es ``True``, recrea la colección vacía.

        Raises:
            IngestionError: Si el *upsert* falla, por ejemplo por un cambio de
                dimensión del embedding.
        """
        client: QdrantClient = self._factory.create_client()
        try:
            self._factory.ensure_collection(
                client,
                self._embeddings,
                recreate=recreate,
            )
            # ensure_collection() ya validó la dimensión (o acaba de crear la
            # colección); se evita un embedding extra en build_store.
            store = self._factory.build_store(
                client,
                self._embeddings,
                validate_dimension=False,
            )
            point_ids = [self._point_id(chunk) for chunk in chunks]
            try:
                store.add_documents(documents=list(chunks), ids=point_ids)
            except Exception as exc:
                raise IngestionError(
                    "No se pudieron indexar los fragmentos. Si cambiaste el "
                    "modelo de embeddings, vuelve a ejecutar con --recreate. "
                    f"Detalle: {exc}"
                ) from exc
        finally:
            client.close()

    def _discover_files(self, path: Path) -> list[Path]:
        """Descubre archivos soportados en una ruta.

        Args:
            path: Archivo o directorio de entrada.

        Returns:
            Lista ordenada de rutas absolutas.

        Raises:
            IngestionError: Si la ruta no existe, es un archivo no soportado o
                el directorio no contiene documentos válidos.
        """
        candidate = path.expanduser()
        if not candidate.exists():
            raise IngestionError(f"La ruta no existe: {candidate}")

        if candidate.is_file():
            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise IngestionError(
                    f"Extensión no soportada: {candidate.suffix!r}. "
                    f"Extensiones válidas: {supported}."
                )
            return [candidate.resolve()]

        files: list[Path] = []
        try:
            for candidate_file in candidate.rglob("*"):
                if not candidate_file.is_file():
                    continue
                if candidate_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                relative = candidate_file.relative_to(candidate)
                if self._is_ignored(relative):
                    continue
                files.append(candidate_file.resolve())
        except OSError as exc:
            raise IngestionError(
                f"No se pudo recorrer el directorio {candidate}: {exc}"
            ) from exc

        if not files:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise IngestionError(
                f"No se encontraron documentos soportados en {candidate}. "
                f"Extensiones válidas: {supported}."
            )
        return sorted(files)

    @staticmethod
    def _is_ignored(relative: Path) -> bool:
        """Indica si una ruta relativa debe excluirse del recorrido.

        Args:
            relative: Ruta relativa al directorio raíz de ingestión.

        Returns:
            ``True`` para archivos ocultos o dentro de directorios excluidos.
        """
        if any(part.startswith(".") for part in relative.parts):
            return True
        return any(part in EXCLUDED_DIRS for part in relative.parts[:-1])

    def _load_file(self, path: Path) -> list[Document]:
        """Extrae documentos de un archivo según su extensión.

        Args:
            path: Archivo a procesar.

        Returns:
            Lista de documentos, uno por página en PDF o uno para texto.

        Raises:
            IngestionError: Si la extensión no está soportada o el archivo no
                contiene texto extraíble.
        """
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix in {".txt", ".md", ".markdown"}:
            return self._load_text(path)
        raise IngestionError(f"Extensión no soportada: {suffix!r}.")

    def _load_text(self, path: Path) -> list[Document]:
        """Carga un archivo de texto plano o Markdown.

        Args:
            path: Archivo de entrada.

        Returns:
            Lista con un único ``Document`` y metadatos de origen.

        Raises:
            IngestionError: Si el archivo está vacío.
        """
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Algunos TXT antiguos usan Latin-1/Win-1252. Se reemplazan los
            # caracteres inválidos en lugar de abortar la ingestión completa.
            content = path.read_text(encoding="utf-8", errors="replace")

        if not content.strip():
            raise IngestionError(f"El archivo está vacío: {path.name}")

        return [
            Document(
                page_content=content,
                metadata={
                    "source": str(path),
                    "page": 1,
                    "file_type": "markdown"
                    if path.suffix.lower() in {".md", ".markdown"}
                    else "text",
                },
            )
        ]

    def _load_pdf(self, path: Path) -> list[Document]:
        """Extrae texto y tablas estructuradas de un PDF página a página.

        Las tablas detectadas por PyMuPDF se añaden a la página en formato
        Markdown. Esto conserva la estructura de filas y columnas, que un
        extractor de texto plano perdería.

        Args:
            path: Ruta del PDF.

        Returns:
            Documentos con ``source``, ``page`` y ``file_type``.

        Raises:
            IngestionError: Si el PDF está protegido, no puede abrirse o no
                contiene texto extraíble (por ejemplo, un escaneo sin OCR).
        """
        try:
            pdf = pymupdf.open(str(path))
        except Exception as exc:
            raise IngestionError(
                f"No se pudo abrir el PDF '{path.name}': {exc}"
            ) from exc

        try:
            if pdf.needs_pass:
                raise IngestionError(
                    f"El PDF '{path.name}' está protegido con contraseña."
                )

            documents: list[Document] = []
            for page_index in range(pdf.page_count):
                page_number = page_index + 1
                page = pdf[page_index]
                text = page.get_text("text", sort=True).strip()
                tables = self._extract_tables(page)
                parts: list[str] = []
                if text:
                    parts.append(text)
                if tables:
                    parts.append(
                        f"## Tablas extraídas (página {page_number})\n\n"
                        + "\n\n".join(tables)
                    )

                page_content = "\n\n".join(parts).strip()
                if not page_content:
                    continue

                documents.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "source": str(path),
                            "page": page_number,
                            "file_type": "pdf",
                        },
                    )
                )
        finally:
            pdf.close()

        if not documents:
            raise IngestionError(
                f"El PDF '{path.name}' no contiene texto extraíble. "
                "Si es un documento escaneado, aplica OCR antes de indexarlo."
            )
        return documents

    def _extract_tables(self, page: pymupdf.Page) -> list[str]:
        """Extrae las tablas de una página como bloques Markdown.

        Args:
            page: Página de PyMuPDF.

        Returns:
            Lista de tablas serializadas; vacía si no hay tablas o la detección
            falla. Un fallo de tabla no invalida el texto de la página.
        """
        tables: list[str] = []
        try:
            finder = page.find_tables()
        except Exception:  # noqa: BLE001 - la tabla es opcional
            return tables

        for table in finder.tables:
            try:
                rows = table.extract()
            except Exception:  # noqa: BLE001, S112 - tabla ilegible
                continue
            markdown = self._rows_to_markdown(rows)
            if markdown:
                tables.append(markdown)
        return tables

    def _rows_to_markdown(
        self,
        rows: Sequence[Sequence[str | None]],
    ) -> str:
        """Convierte filas de una tabla a una tabla Markdown.

        Args:
            rows: Filas devueltas por PyMuPDF, posiblemente con celdas nulas.

        Returns:
            Tabla Markdown o cadena vacía si no hay contenido.
        """
        if not rows:
            return ""

        normalized: list[list[str]] = [
            [self._clean_cell(cell) for cell in row] for row in rows
        ]
        # PyMuPDF suele devolver una fila final vacía cuando detecta bordes;
        # se descarta para no generar una fila espuria en el Markdown.
        normalized = [row for row in normalized if any(row)]
        if not normalized:
            return ""

        width = max(len(row) for row in normalized)
        if width == 0:
            return ""

        padded = [row + [""] * (width - len(row)) for row in normalized]
        if not any(cell for row in padded for cell in row):
            return ""

        if any(padded[0]):
            header = padded[0]
            body = padded[1:]
        else:
            header = [f"Columna {index}" for index in range(1, width + 1)]
            body = padded

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(lines)

    @staticmethod
    def _clean_cell(cell: str | None) -> str:
        """Normaliza el contenido de una celda para Markdown.

        Args:
            cell: Valor original de la celda.

        Returns:
            Texto sin saltos de línea y con las barras verticales escapadas.
        """
        if cell is None:
            return ""
        return " ".join(str(cell).split()).replace("|", "\\|")

    def _chunk_documents(self, documents: Sequence[Document]) -> list[Document]:
        """Fragmenta documentos y enriquece sus metadatos.

        Args:
            documents: Documentos extraídos de un archivo.

        Returns:
            Fragmentos con ``source``, ``page``, ``chunk_id``, ``chunk_index``
            y ``start_index``.
        """
        raw_chunks = self._splitter.split_documents(list(documents))
        chunks: list[Document] = []

        for index, chunk in enumerate(raw_chunks):
            metadata = dict(chunk.metadata)
            source = str(metadata.get("source", "desconocido"))
            page_value = metadata.get("page", 1)
            page = page_value if isinstance(page_value, int) else 1
            start_value = metadata.get("start_index", 0)
            start_index = start_value if isinstance(start_value, int) else 0

            fingerprint = (
                f"{source}|{page}|{start_index}|{chunk.page_content}"
            ).encode()
            digest = hashlib.sha1(fingerprint).hexdigest()[:12]
            stem = Path(source).stem.replace(" ", "_") or "documento"
            chunk_id = f"{stem}-p{page}-{digest}"

            chunk.metadata = {
                **metadata,
                "chunk_id": chunk_id,
                "chunk_index": index,
            }
            chunks.append(chunk)

        return chunks

    @staticmethod
    def _point_id(chunk: Document) -> str:
        """Genera un UUID determinista para el punto de Qdrant.

        Qdrant acepta UUIDs o enteros como ID. Se deriva un UUID5 del
        ``chunk_id`` para que dos ingestas del mismo contenido hagan *upsert*
        sobre el mismo punto en lugar de duplicarlo.

        Args:
            chunk: Fragmento con ``chunk_id`` en metadatos.

        Returns:
            UUID en formato canónico.
        """
        chunk_id = str(chunk.metadata.get("chunk_id", ""))
        return str(uuid5(NAMESPACE_URL, chunk_id))
