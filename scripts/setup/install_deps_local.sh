# Check if ISAACLAB_PATH is set
if [ -z "${ISAACLAB_PATH}" ]; then
    echo "Error: ISAACLAB_PATH environment variable is not set."
    echo "Please set it to the path of your Isaac Lab installation."
    echo "Example: export ISAACLAB_PATH=/path/to/isaac_lab"
    exit 1
fi

echo "=== INSTALLING ARNOLD AND DEPENDENCIES ==="
# First install dependencies from pyproject.toml (excluding torch/numpy that Isaac provides)
echo "Installing dependencies from pyproject.toml..."
${ISAACLAB_PATH:?}/isaaclab.sh -p scripts/setup/install_deps_from_pyproject.py

echo "=== INSTALLING ARNOLD PACKAGE ==="
# Install ARNOLD package itself with --no-deps
${ISAACLAB_PATH:?}/isaaclab.sh -p -m pip install --no-deps -e .

# Check that IsaacLab is using the correct version.
if [ -d ${ISAACLAB_PATH}/.git ]; then
  expected_isaac_lab_tag="v2.3.2"
  if ! git -C ${ISAACLAB_PATH} tag -l "${expected_isaac_lab_tag}" | grep -q "${expected_isaac_lab_tag}"; then
      echo "Error: IsaacLab does not have the expected tag."
      echo "Expected tag: ${expected_isaac_lab_tag}"
      echo "Please checkout the correct version: git -C ${ISAACLAB_PATH} checkout ${expected_isaac_lab_tag}"
      exit 1
  fi
fi

echo "=== LOCAL DEVELOPMENT INSTALLATION COMPLETE ==="
echo "🎉 All dependencies and packages installed successfully!"