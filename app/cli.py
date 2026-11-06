"""CLI del sistema RAG con Typer y salida enriquecida con Rich.

Comandos:
    ingest  Indexa documentos en Qdrant.
    ask     Responde una pregunta puntual.
    chat    Inicia una sesión interactiva.
    info    Muestra configuración y estado de la colección.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from app.config import Settings, get_settings
from app.core.factory import ModelFactory
from app.core.vector_store import CollectionStats, VectorStoreFactory
from app.exceptions import RagError
from app.services.ingestion import IngestionReport, IngestionService
from app.services.rag import RagResponse, RagService, SourceReference

app = typer.Typer(
    name="rag",
    help="Sistema RAG local/híbrido con LangChain, Qdrant y Ollama.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
console = Console()


def _configure_logging(settings: Settings) -> None:
    """Configura el logging global del proceso.

    Args:
        settings: Configuración que contiene ``log_level``.

    Returns:
        ``None``.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Las librerías HTTP registran cada request en INFO y rompen la estética
    # de la CLI; sus errores reales se siguen mostrando como WARNING/ERROR.
    for noisy_logger in ("httpx", "httpcore", "urllib3", "qdrant_client"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _load_settings() -> Settings:
    """Carga la configuración y presenta errores de validación de forma amable.

    Returns:
        Configuración validada.

    Raises:
        typer.Exit: Con código 2 si la configuración es inválida.
    """
    try:
        settings = get_settings()
    except ValidationError as exc:
        _abort(str(exc), "Configuración inválida", code=2)

    _configure_logging(settings)
    return settings


def _abort(message: str, title: str, *, code: int = 1) -> NoReturn:
    """Muestra un panel de error y termina la CLI.

    Args:
        message: Detalle del error.
        title: Título del panel.
        code: Código de salida del proceso.

    Raises:
        typer.Exit: Siempre.
    """
    console.print(Panel(message, title=title, border_style="red"))
    raise typer.Exit(code=code)


def _build_ingestion(settings: Settings) -> IngestionService:
    """Construye el servicio de ingestión con sus dependencias.

    Args:
        settings: Configuración validada.

    Returns:
        Servicio listo para indexar.

    Raises:
        ConfigurationError: Si faltan dependencias, credenciales o el modelo
            de embeddings no está disponible.
    """
    embeddings = ModelFactory(settings).create_embeddings()
    return IngestionService(
        settings=settings,
        embeddings=embeddings,
        vector_store_factory=VectorStoreFactory(settings),
    )


def _build_rag(settings: Settings, *, top_k: int | None = None) -> RagService:
    """Construye el servicio RAG y abre la colección vectorial.

    Args:
        settings: Configuración validada.
        top_k: Número de fragmentos a recuperar; ``None`` usa el valor de
            configuración.

    Returns:
        Servicio RAG que posee el cliente Qdrant abierto.

    Raises:
        RagError: Si falla la conexión, la colección no existe o el modelo no
            puede inicializarse.
    """
    factory = ModelFactory(settings)
    embeddings = factory.create_embeddings()
    vector_factory = VectorStoreFactory(settings)
    client = vector_factory.create_client()

    try:
        store = vector_factory.build_store(client, embeddings)
        llm = factory.create_llm()
        return RagService(
            settings=settings,
            vector_store=store,
            llm=llm,
            client=client,
            top_k=top_k,
        )
    except Exception:
        client.close()
        raise


def _settings_table(settings: Settings) -> Table:
    """Construye una tabla con la configuración efectiva.

    Args:
        settings: Configuración validada.

    Returns:
        Tabla de Rich lista para imprimir.
    """
    location = settings.qdrant_location
    table = Table(
        title="Configuración activa",
        show_header=False,
        box=None,
        padding=(0, 1),
    )
    table.add_column("Clave", style="bold cyan", no_wrap=True)
    table.add_column("Valor", overflow="fold")
    table.add_row("Entorno", settings.app_env)
    table.add_row("LLM", f"{settings.llm_provider} · {settings.llm_model}")
    table.add_row(
        "Embeddings",
        f"{settings.resolved_embedding_provider} · {settings.embedding_model}",
    )
    table.add_row("Qdrant", f"{settings.qdrant_mode} · {location}")
    table.add_row("Colección", settings.collection_name)
    table.add_row("Top-k", str(settings.top_k))
    return table


def _render_ingestion_report(report: IngestionReport) -> None:
    """Muestra el resultado de la ingestión.

    Args:
        report: Informe devuelto por ``IngestionService``.

    Returns:
        ``None``.
    """
    table = Table(title="Resultado de ingestión", header_style="bold cyan")
    table.add_column("Archivo", style="bold")
    table.add_column("Unidades", justify="right")
    table.add_column("Fragmentos", justify="right")
    table.add_column("Estado", justify="center")

    for file_report in report.files:
        state = (
            Text("OK", style="green")
            if file_report.ok
            else Text("ERROR", style="red")
        )
        table.add_row(
            file_report.source.name,
            str(file_report.documents),
            str(file_report.chunks),
            state,
        )
    console.print(table)

    for file_report in report.files:
        if file_report.error:
            console.print(
                Panel(
                    file_report.error,
                    title=f"Error en {file_report.source.name}",
                    border_style="red",
                )
            )

    console.print(
        Panel.fit(
            f"Archivos: {len(report.files)} "
            f"(OK: {report.successful_files}, errores: {report.failed_files})\n"
            f"Fragmentos indexados: {report.total_chunks}\n"
            f"Colección: {report.collection_name}\n"
            f"Tiempo: {report.elapsed_seconds:.2f}s",
            title="Resumen",
            border_style="cyan",
        )
    )


def _sources_table(sources: tuple[SourceReference, ...]) -> Table:
    """Construye la tabla de fuentes recuperadas.

    Args:
        sources: Referencias devueltas por el servicio RAG.

    Returns:
        Tabla de Rich lista para imprimir.
    """
    table = Table(
        title="Fuentes recuperadas",
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Documento", style="bold")
    table.add_column("Pág.", justify="center", no_wrap=True)
    table.add_column("Fragmento", style="dim", no_wrap=True)
    table.add_column("Extracto", overflow="fold")

    for index, source in enumerate(sources, start=1):
        page = str(source.page) if source.page is not None else "—"
        chunk_id = source.chunk_id or "—"
        table.add_row(
            str(index),
            source.display_name,
            page,
            chunk_id,
            source.excerpt,
        )
    return table


def _render_response(response: RagResponse, *, show_sources: bool) -> None:
    """Imprime una respuesta RAG con su panel y fuentes.

    Args:
        response: Resultado de la consulta.
        show_sources: Si es ``True``, muestra la tabla de fragmentos.

    Returns:
        ``None``.
    """
    if response.is_no_context:
        console.print(
            Panel(
                Markdown(response.answer),
                title="Sin contexto suficiente",
                border_style="yellow",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            Panel(
                Markdown(response.answer),
                title="Respuesta",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    if show_sources:
        if response.sources:
            console.print(_sources_table(response.sources))
        else:
            console.print(
                "[yellow]No se recuperaron fragmentos para esta consulta.[/]"
            )

    console.print(
        f"[dim]{len(response.sources)} fuente(s) · "
        f"{response.elapsed_seconds:.2f}s[/]"
    )


def _render_collection_stats(stats: CollectionStats, collection_name: str) -> None:
    """Muestra el estado de la colección vectorial.

    Args:
        stats: Estadísticas obtenidas de Qdrant.
        collection_name: Nombre de la colección.

    Returns:
        ``None``.
    """
    if not stats.exists:
        console.print(
            Panel(
                f"La colección '{collection_name}' todavía no existe.\n"
                "Ejecuta: python -m app.cli ingest --path ./data/documents/",
                title="Colección",
                border_style="yellow",
            )
        )
        return

    points = "—" if stats.points_count is None else f"{stats.points_count:,}"
    table = Table(title=f"Colección {collection_name}", show_header=False)
    table.add_column("Métrica", style="bold cyan")
    table.add_column("Valor")
    table.add_row("Puntos", points)
    table.add_row("Estado", (stats.status or "desconocido").upper())
    table.add_row(
        "Dimensión",
        str(stats.vector_size) if stats.vector_size is not None else "—",
    )
    console.print(table)


@app.command(help="Indexa documentos PDF, TXT y Markdown en Qdrant.")
def ingest(
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            "-p",
            help="Archivo o directorio de documentos (.pdf, .txt, .md).",
            resolve_path=True,
        ),
    ] = Path("./data/documents"),
    recreate: Annotated[
        bool,
        typer.Option(
            "--recreate",
            help="Elimina y recrea la colección antes de indexar.",
        ),
    ] = False,
) -> None:
    """Indexa documentos PDF, TXT y Markdown en Qdrant.

    Extrae texto y tablas, fragmenta con solape y conserva metadatos de
    origen, página y chunk para las citas.

    Args:
        path: Archivo o directorio de documentos.
        recreate: Si es ``True``, elimina la colección antes de indexar.

    Returns:
        ``None``.

    Raises:
        typer.Exit: Si la ruta no existe, la ingestión falla o no se indexa
            ningún fragmento.
    """
    settings = _load_settings()
    console.print(_settings_table(settings))

    if not path.exists():
        _abort(f"La ruta no existe: {path}", "Ruta inválida")

    try:
        service = _build_ingestion(settings)
        with console.status(
            "[bold cyan]Extrayendo, fragmentando e indexando documentos...[/]"
        ):
            report = service.ingest(path, recreate=recreate)
    except RagError as exc:
        _abort(str(exc), "Error de ingestión")

    _render_ingestion_report(report)

    if report.total_chunks == 0 or report.failed_files > 0:
        raise typer.Exit(code=1)


@app.command(help="Responde una pregunta puntual usando el corpus indexado.")
def ask(
    question: Annotated[
        str,
        typer.Argument(help="Pregunta técnica a responder con el corpus."),
    ],
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            "-k",
            min=1,
            max=50,
            help="Fragmentos a recuperar; por defecto usa TOP_K del .env.",
        ),
    ] = None,
    show_sources: Annotated[
        bool,
        typer.Option(
            "--sources/--no-sources",
            help="Muestra u oculta la tabla de fuentes recuperadas.",
        ),
    ] = True,
) -> None:
    """Responde una pregunta puntual usando el corpus indexado.

    Args:
        question: Pregunta técnica del usuario.
        top_k: Fragmentos a recuperar; ``None`` usa ``TOP_K`` del entorno.
        show_sources: Si es ``True``, imprime la tabla de fuentes.

    Returns:
        ``None``.

    Raises:
        typer.Exit: Si la colección no existe, el modelo falla o la
            configuración es inválida.
    """
    settings = _load_settings()

    try:
        service = _build_rag(settings, top_k=top_k)
    except RagError as exc:
        _abort(str(exc), "Error de consulta")

    try:
        with console.status("[bold cyan]Recuperando contexto y generando...[/]"):
            response = service.ask(question)
    except (RagError, ValueError) as exc:
        _abort(str(exc), "Error de consulta")
    finally:
        service.close()

    _render_response(response, show_sources=show_sources)


@app.command(help="Inicia una sesión interactiva de preguntas y respuestas.")
def chat(
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            "-k",
            min=1,
            max=50,
            help="Fragmentos a recuperar por turno.",
        ),
    ] = None,
    show_sources: Annotated[
        bool,
        typer.Option(
            "--sources/--no-sources",
            help="Muestra u oculta la tabla de fuentes en cada turno.",
        ),
    ] = True,
) -> None:
    """Inicia una sesión interactiva de preguntas y respuestas.

    El historial se usa para resolver referencias conversacionales, pero el
    prompt obliga a fundamentar los hechos únicamente en el contexto recuperado.

    Args:
        top_k: Fragmentos a recuperar en cada turno.
        show_sources: Si es ``True``, imprime las fuentes en cada respuesta.

    Returns:
        ``None``.

    Raises:
        typer.Exit: Si no se puede abrir la colección o inicializar el modelo.
    """
    settings = _load_settings()

    try:
        service = _build_rag(settings, top_k=top_k)
    except RagError as exc:
        _abort(str(exc), "Error de consulta")

    history: list[BaseMessage] = []
    max_messages = settings.max_history_turns * 2

    console.print(
        Panel(
            "Escribe una pregunta o usa:\n"
            "[bold]/help[/]   Ayuda\n"
            "[bold]/clear[/]  Borra el historial\n"
            "[bold]/exit[/]   Sale de la sesión",
            title="Chat RAG",
            border_style="cyan",
        )
    )

    try:
        while True:
            try:
                raw_question = Prompt.ask("\n[bold cyan]Tú[/]", console=console)
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Sesión finalizada.[/]")
                break

            question = raw_question.strip()
            if not question:
                continue

            command = question.lower()
            if command in {"/exit", "/quit", "/salir"}:
                console.print("[dim]Sesión finalizada.[/]")
                break
            if command in {"/clear", "/limpiar"}:
                history.clear()
                console.print("[green]Historial borrado.[/]")
                continue
            if command in {"/help", "/ayuda"}:
                console.print(
                    Panel(
                        "/help   Muestra esta ayuda\n"
                        "/clear  Borra el historial de la conversación\n"
                        "/exit   Termina la sesión",
                        title="Ayuda",
                        border_style="cyan",
                    )
                )
                continue

            try:
                with console.status(
                    "[bold cyan]Recuperando contexto y generando...[/]"
                ):
                    response = service.ask(question, history=history)
            except (RagError, ValueError) as exc:
                console.print(
                    Panel(str(exc), title="Error", border_style="red")
                )
                continue

            _render_response(response, show_sources=show_sources)

            # Solo se conserva el turno si hubo contexto: las respuestas de
            # "no encontrado" no aportan información factual al historial.
            if response.has_context:
                history.extend(
                    [
                        HumanMessage(content=question),
                        AIMessage(content=response.answer),
                    ]
                )
                if max_messages == 0:
                    history.clear()
                elif len(history) > max_messages:
                    del history[: len(history) - max_messages]
    finally:
        service.close()


@app.command(help="Muestra la configuración activa y el estado de la colección.")
def info() -> None:
    """Muestra la configuración activa y el estado de la colección.

    Returns:
        ``None``.

    Raises:
        typer.Exit: Si la configuración es inválida o Qdrant no responde.
    """
    settings = _load_settings()
    console.print(_settings_table(settings))

    vector_factory = VectorStoreFactory(settings)
    try:
        client = vector_factory.create_client()
    except RagError as exc:
        _abort(str(exc), "Error de conexión con Qdrant")

    try:
        stats = vector_factory.collection_stats(client)
    except RagError as exc:
        _abort(str(exc), "Error de conexión con Qdrant")
    finally:
        client.close()

    _render_collection_stats(stats, settings.collection_name)


if __name__ == "__main__":
    app()
