from fastapi import FastAPI

from compliance_copilot.api.routers import health


def create_app() -> FastAPI:
    app = FastAPI(title="Compliance Copilot")
    app.include_router(health.router)
    return app


app = create_app()
