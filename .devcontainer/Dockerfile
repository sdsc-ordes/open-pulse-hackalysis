FROM ghcr.io/astral-sh/uv:python3.13-bookworm

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install system dependencies + just
RUN apt-get update && apt-get install -y \
    sudo \
    curl \
    jq \
    ca-certificates \
    gnupg \
    && curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for VS Code
RUN useradd -ms /bin/bash -u 1000 vscode \
    && echo "vscode ALL=(ALL) ALL" >> /etc/sudoers

# Gemini CLI
# Login outside the container and copy your credentials to ~/.gemini if needed.
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @google/gemini-cli \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/data \
    && chown -R 1000:1000 /app/data \
    && chmod 775 /app/data

USER vscode
WORKDIR /workspaces
