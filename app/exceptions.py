"""Excepciones específicas del dominio RAG.

Se separan de las excepciones genéricas para que la CLI pueda presentar
mensajes accionables sin exponer trazas internas al usuario final.
"""

from __future__ import annotations


class RagError(Exception):
    """Excepción base de todos los errores controlados del sistema RAG."""


class ConfigurationError(RagError):
    """La configuración o una dependencia externa no es válida."""


class IngestionError(RagError):
    """Falló la lectura, el chunking o la indexación de documentos."""


class CollectionNotFoundError(RagError):
    """La colección vectorial no existe o aún no fue creada."""


class GenerationError(RagError):
    """Falló la recuperación o la generación de la respuesta."""
