#!/bin/bash
# Shared Debian/Ubuntu setup steps used by the zsl Dockerfile.
# Based on the standard devcontainers/common-debian script, trimmed to the
# minimum required for a reproducible development container.

set -e

INSTALL_ZSH=${1:-"true"}
USERNAME=${2:-"vscode"}
USER_UID=${3:-"1000"}
USER_GID=${4:-"$USER_UID"}
UPGRADE_PACKAGES=${5:-"false"}

if [ "$(id -u)" -ne 0 ]; then
    echo -e 'Script must be run as root. Use sudo, su, or add "USER root" to your Dockerfile before running this script.'
    exit 1
fi

# Treat unset variables as errors for the remainder of this script.
set -u

# Ensure apt is in non-interactive mode.
export DEBIAN_FRONTEND=noninteractive

# Optionally upgrade all installed packages.
if [ "${UPGRADE_PACKAGES}" = "true" ]; then
    apt-get upgrade -y
fi

# Install zsh and related convenience packages if requested.
if [ "${INSTALL_ZSH}" = "true" ]; then
    apt-get install -y --no-install-recommends zsh
fi

# Create or update the non-root user and group.
if ! getent group "${USERNAME}" >/dev/null 2>&1; then
    groupadd --gid "${USER_GID}" "${USERNAME}"
fi

if id -u "${USERNAME}" >/dev/null 2>&1; then
    usermod --uid "${USER_UID}" --gid "${USER_GID}" "${USERNAME}"
else
    useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}"
fi

# Add sudo support for the non-root user.
apt-get install -y --no-install-recommends sudo
mkdir -p /etc/sudoers.d
echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"
chmod 0440 "/etc/sudoers.d/${USERNAME}"

# Ensure the user owns their home directory.
chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}"

echo "common-debian.sh: user '${USERNAME}' created with UID ${USER_UID}, GID ${USER_GID}"
