from dynaconf import Dynaconf
from dynaconf.constants import DEFAULT_SETTINGS_FILES

settings = Dynaconf(
    environments=True,
    lowercase_read=False,
    load_dotenv=True,
    default_settings_paths=DEFAULT_SETTINGS_FILES,
)


def get_postgres_uri() -> str:
    db_name = settings.DB_NAME
    user = settings.DB_USER
    password = settings.DB_PASSWORD
    host = settings.DB_HOST
    port = settings.DB_PORT
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


def get_sige_api_url() -> str:
    return "http://sige.unb.br:443/graph/quarterly-total-consumption/"
