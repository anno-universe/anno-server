import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.utils.functional import Promise

from .api import api


class LazyEncoder(json.JSONEncoder):
    """Resolve Django lazy translation objects before encoding."""

    def default(self, obj):
        if isinstance(obj, Promise):
            return str(obj)
        return super().default(obj)


class Command(BaseCommand):
    help = "Export the OpenAPI JSON schema to a file."

    def add_arguments(self, parser):
        parser.add_argument(
            "-o",
            "--output",
            default="openapi.json",
            help="Output file path (default: openapi.json in the project root).",
        )

    def handle(self, *args, **options):
        output = options["output"]
        schema = api.get_openapi_schema()
        path = Path(output)
        path.write_text(
            json.dumps(schema, indent=2, ensure_ascii=False, cls=LazyEncoder),
            encoding="utf-8",
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"OpenAPI schema written to {path.absolute()} "
                f"({len(path.read_text())} bytes)"
            )
        )
