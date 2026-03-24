import sdot

def test_driver():
    # will force jax
    import jax.numpy as jnp

    assert sdot.driver.normalized_framework == "jax"
    assert sdot.driver.normalized_dtype == "FP64"
    assert sdot.driver.user_dtype is None
    assert sdot.driver.dtype == jnp.float64

    sdot.driver.dtype = "FP32"
    assert sdot.driver.normalized_dtype == "FP32"
    assert sdot.driver.user_dtype == "FP32"
    assert sdot.driver._instance is None
    assert sdot.driver.dtype == jnp.float32

    sdot.driver.framework = "torch"
    assert sdot.driver.user_dtype == "FP32"
    assert sdot.driver._instance is None

    import torch
    assert sdot.driver.dtype == torch.float32
