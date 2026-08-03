from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from compliance_copilot.api.routers import analyse, clauses, documents, health
from compliance_copilot.config import get_settings
from compliance_copilot.db.session import init_models, make_engine, make_session_factory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    await init_models(engine)
    app.state.session_factory = make_session_factory(engine)
    yield
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="Compliance Copilot", lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(clauses.router)
    app.include_router(analyse.router)
    return app


app = create_app()
