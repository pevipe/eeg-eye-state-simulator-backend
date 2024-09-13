# Copyright 2024 Pelayo Vieites Pérez
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

from os import environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.adapters.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        version="0.1.0",
        title="Classifiers API",
        description="Classifiers API for EEG data",
        docs_url="/docs",
        redoc_url="/redocs")

    app.include_router(prefix="/classifiers", router=router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[environ.get("FRONTEND_URL")],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    return app
