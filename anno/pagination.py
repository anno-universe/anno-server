from typing import Generic, TypeVar

from django.db.models import QuerySet
from ninja import Schema

T = TypeVar("T")

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


class PaginatedResponse(Schema, Generic[T]):
    count: int
    limit: int
    offset: int
    items: list[T]


def paginate_queryset(
    qs: QuerySet,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> tuple[int, int, int, list]:
    """Clamp limit/offset, count total rows, and slice the queryset.

    Returns ``(count, limit, offset, rows)`` with *limit* and *offset*
    clamped to sane bounds.  *rows* is an eagerly-evaluated list so that
    ``count`` and the page data reflect the same database snapshot.
    """
    limit = max(1, min(limit, MAX_LIMIT))
    offset = max(0, offset)
    count = qs.count()
    rows = list(qs[offset : offset + limit])
    return count, limit, offset, rows
