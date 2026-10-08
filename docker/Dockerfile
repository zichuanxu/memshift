# CPU image for MemShift experiment drivers. Dependencies only: the repository is bind-mounted
# read-only at /work and run with PYTHONPATH=/work/src, so the image does not change per commit.
FROM python:3.12-slim-bookworm
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock && rm /tmp/requirements.lock
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /work
