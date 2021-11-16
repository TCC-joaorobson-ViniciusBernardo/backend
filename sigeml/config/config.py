from typing import TypedDict

from dynaconf import Dynaconf
from dynaconf.constants import DEFAULT_SETTINGS_FILES

settings = Dynaconf(
    environments=True,
    lowercase_read=False,
    load_dotenv=True,
    default_settings_paths=DEFAULT_SETTINGS_FILES,
)


class RedisCredentials(TypedDict):
    host: str
    port: int


def get_postgres_uri() -> str:
    db_name = settings.DB_NAME
    user = settings.DB_USER
    password = settings.DB_PASSWORD
    host = settings.DB_HOST
    port = settings.DB_PORT
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def get_sige_api_url() -> str:
    return "http://sige.unb.br:443"


def get_redis_credentials() -> RedisCredentials:
    return {"host": settings.REDIS_HOST, "port": settings.REDIS_PORT}
