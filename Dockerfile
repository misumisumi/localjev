# LocalJev: Jev-compatible System One API that reads first-token logprobs from an
# existing llama-server. Runtime has no third-party JS dependencies; Bun executes
# the TypeScript sources directly. curl is installed only so the Kubernetes
# httpGet probes work under `podman kube play`, which emulates them with curl.
#
#   podman build -t localhost/localjev:latest .
#   podman run --rm --network llm -p 8081:8081 \
#     -e LOCALJEV_UPSTREAM=http://qwen38-flash-next-mtp:9010 localhost/localjev:latest

FROM docker.io/oven/bun:1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV NODE_ENV=production

# No runtime dependencies exist today, but keep the layer so adding one works.
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile --production

COPY src ./src

ENV LOCALJEV_HOST=0.0.0.0 \
    LOCALJEV_PORT=8081

EXPOSE 8081

USER bun

ENTRYPOINT ["bun", "run", "/app/src/index.ts"]
