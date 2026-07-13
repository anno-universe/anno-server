#!/bin/sh
set -e

# Migrations and static collection must run in exactly one place. The worker
# uses the same image, so only RUN_MODE=server runs `migrate`/`collectstatic`.
# Otherwise the worker races the server on AddField -> DuplicateColumn crash loop.
if [ "${RUN_MODE:-server}" = "server" ]; then
    python manage.py migrate --noinput
    python manage.py collectstatic --noinput
fi

exec "$@"
