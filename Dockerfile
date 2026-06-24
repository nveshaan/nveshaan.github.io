# Use a stable Debian image
FROM debian:bookworm-slim

ENV HUGO_VERSION=0.128.0

# 1. Install system prerequisites along with Node.js and npm
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    ca-certificates \
    nodejs \
    npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Download and install Hugo Extended
RUN wget -O /tmp/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
    && dpkg -i /tmp/hugo.deb \
    && rm /tmp/hugo.deb

# 3. Install Dart Sass via npm globally (foolproof cross-platform method)
RUN npm install -g sass

WORKDIR /src

EXPOSE 1313

CMD ["hugo", "server", "--bind", "0.0.0.0", "--buildDrafts", "--buildFuture"]