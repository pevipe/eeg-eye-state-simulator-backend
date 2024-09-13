# Copyright 2024 Pelayo Vieites Pérez
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

FROM python:3.12.3-alpine3.20

WORKDIR /app
ENV SIMULATOR_PERSISTENT_PATH=/app/eeg-persistence
ENV FRONTEND_URL=http://localhost

COPY Pipfile ./
RUN apk update && apk add git && apk add gcc && apk add --update alpine-sdk
RUN pip install --no-cache-dir pipenv

RUN pipenv install

COPY . .
RUN unzip /app/eeg-persistence.zip -d /app

CMD ["pipenv", "run", "uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000