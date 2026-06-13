ARG BASE_IMAGE=debian:bookworm-slim

# -----------------------------------------------------------------------------
# Base image: Zig toolchain + scientific dependencies for zsl
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS zsl-base

# disable tzdata questions
ENV DEBIAN_FRONTEND=noninteractive

# use bash for the multi-line RUN instructions
SHELL ["/bin/bash", "-c"]

# Install base tools and optional native libraries that zsl may link against.
# trivy:ignore:AVD-DS-0017 apt-get update and install run in the same heredoc.
RUN <<EOF
set -e
apt-get update -y
apt-get install -y --no-install-recommends \
  apt-utils 2> >( grep -v 'debconf: delaying package configuration, since apt-utils is not installed' >&2 ) \
  ca-certificates \
  curl \
  git \
  make \
  python3 \
  python3-pip \
  python3-venv \
  gcc \
  g++ \
  gfortran \
  libopenblas-dev \
  liblapacke-dev \
  libhdf5-dev \
  libfftw3-dev
apt-get autoremove -y
apt-get clean -y
rm -rf /var/lib/apt/lists/* /tmp/library-scripts
EOF

# Install the exact Zig version used by the project.
ARG ZIG_VERSION=0.17.0-dev.813+2153f8143
ENV ZIG_VERSION=${ZIG_VERSION}
RUN <<EOF
set -e
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)  ZIG_ARCH="x86_64-linux" ;;
    aarch64) ZIG_ARCH="aarch64-linux" ;;
    armv7l)  ZIG_ARCH="arm-linux" ;;
    riscv64) ZIG_ARCH="riscv64-linux" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac
ZIG_TARBALL="zig-${ZIG_ARCH}-${ZIG_VERSION}.tar.xz"
ZIG_URL="https://ziglang.org/builds/${ZIG_TARBALL}"
echo "Downloading Zig from ${ZIG_URL}"
curl -fsSL "${ZIG_URL}" -o "/tmp/${ZIG_TARBALL}"
mkdir -p /opt/zig
tar -xf "/tmp/${ZIG_TARBALL}" -C /opt/zig --strip-components=1
rm -f "/tmp/${ZIG_TARBALL}"
/opt/zig/zig version
EOF

ENV PATH="/opt/zig:${PATH}"

# Copy the project and set the default command to run the test suite.
WORKDIR /workspace
COPY . /workspace

CMD ["zig", "build", "test", "--summary", "all"]

# -----------------------------------------------------------------------------
# Dev image: adds a non-root user and shell conveniences
# -----------------------------------------------------------------------------
FROM zsl-base AS zsl-dev

ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="false"
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV EDITOR=code

COPY docker/common-debian.sh /tmp/library-scripts/
# trivy:ignore:AVD-DS-0017 common-debian.sh performs package installation after update.
RUN <<EOF
set -e
apt-get update -y
/bin/bash /tmp/library-scripts/common-debian.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}"
apt-get autoremove -y
apt-get clean -y
rm -rf /var/lib/apt/lists/* /tmp/library-scripts
EOF

USER ${USERNAME}
HEALTHCHECK CMD true
