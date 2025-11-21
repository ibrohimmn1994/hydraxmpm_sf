def _run():
    import jax
    import jax.numpy as jnp
    import jaxlib
    from jax import jit
    from jax.lib import xla_bridge

    print("jax:", jax.__version__)
    print("jaxlib:", jaxlib.__version__)
    try:
        be = xla_bridge.get_backend()
        print("backend:", be.platform)
        # shows CUDA & driver when on GPU
        print("platform_version:", getattr(be, "platform_version", "<unknown>"))
    except Exception as e:
        # If backend selection failed, provide a clear hint
        print("backend: <unavailable>")
        print("backend_init_error:", repr(e))
        raise

    @jit
    def f(A):
        return jnp.linalg.inv(A)

    A = jnp.eye(8)
    print(f(A).block_until_ready())


_run()


















