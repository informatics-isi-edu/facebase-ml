"""
CIFAR-10 CNN model configuration registrations for Hydra/Hydra-Zen.

This module defines configurations for the CIFAR-10 2-layer CNN model and
registers them into Hydra's store under the "model_config" group.

All model parameters are configurable via Hydra:
- Architecture: conv1_channels, conv2_channels, hidden_size, dropout_rate
- Training: learning_rate, epochs, batch_size, weight_decay
- Data: label_column

Example usage:
    # Run with default config
    uv run src/deriva_run.py +model_config=cifar10_default

    # Run with higher learning rate
    uv run src/deriva_run.py +model_config=cifar10_fast_lr

    # Override specific parameters
    uv run src/deriva_run.py +model_config=cifar10_default +model_config.epochs=50
"""
from __future__ import annotations

from hydra_zen import builds, store

from models.cifar10_cnn import cifar10_cnn

# Build the base CIFAR-10 CNN configuration.
# All parameters have sensible defaults for a simple training run.
Cifar10CNNConfig = builds(
    cifar10_cnn,
    # Architecture parameters
    conv1_channels=32,
    conv2_channels=64,
    hidden_size=128,
    dropout_rate=0.0,
    # Training parameters
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    weight_decay=0.0,
    # Data parameters
    label_column="Label",
    # Hydra-zen settings
    populate_full_signature=True,
    zen_partial=True,  # Execution context added later
)

# Register configurations to the model_config group
model_store = store(group="model_config")

# Default configuration - good starting point
model_store(Cifar10CNNConfig, name="default_model")

# Quick training - fewer epochs for testing
model_store(
    Cifar10CNNConfig,
    name="cifar10_quick",
    epochs=3,
    batch_size=128,
)

# Larger model - more capacity
model_store(
    Cifar10CNNConfig,
    name="cifar10_large",
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=256,
    epochs=20,
)

# With dropout for regularization
model_store(
    Cifar10CNNConfig,
    name="cifar10_regularized",
    dropout_rate=0.25,
    weight_decay=1e-4,
    epochs=20,
)

# Fast learning rate - may converge faster but less stable
model_store(
    Cifar10CNNConfig,
    name="cifar10_fast_lr",
    learning_rate=1e-2,
    epochs=15,
)

# Slow learning rate - more stable, may need more epochs
model_store(
    Cifar10CNNConfig,
    name="cifar10_slow_lr",
    learning_rate=1e-4,
    epochs=30,
)

# Extended training - for best accuracy
model_store(
    Cifar10CNNConfig,
    name="cifar10_extended",
    conv1_channels=64,
    conv2_channels=128,
    hidden_size=256,
    dropout_rate=0.25,
    weight_decay=1e-4,
    learning_rate=1e-3,
    epochs=50,
)
