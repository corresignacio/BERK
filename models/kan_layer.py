import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Layer
from typing import Union, Tuple, List, Any, Callable

def _build_uniform_grid_from_data(inputs: tf.Tensor,
                                  grid_size: int,
                                  spline_order: int,
                                  prev_grid: tf.Tensor,
                                  margin: float = 0.05,
                                  grid_eps: float = 0.02,
                                  dtype=tf.float32) -> tf.Tensor:
    x = tf.reshape(inputs, (-1, tf.shape(inputs)[-1]))
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    span  = tf.maximum(x_max - x_min, tf.cast(1e-6, x.dtype))
    lo    = x_min - margin * span
    hi    = x_max + margin * span

    h = (hi - lo) / tf.cast(grid_size, x.dtype)
    M = grid_size + 2 * spline_order + 1
    adapt = tf.linspace(lo - spline_order * h, hi + spline_order * h, M)
    adapt = tf.cast(adapt, dtype)

    prev = tf.reshape(tf.cast(prev_grid, dtype), (-1,))
    new_vec = grid_eps * prev + (1.0 - grid_eps) * adapt
    return tf.reshape(new_vec, (1, M))

def _fit_spline_kernel_ls(basis_new: tf.Tensor,
                          spline_old: tf.Tensor,
                          l2_reg: float = 1e-6,
                          fast: bool = True) -> tf.Tensor:
    # basis_new: (..., in_size, M), spline_old: (..., in_size, U)
    in_size = tf.shape(basis_new)[-2]
    Bn = tf.reshape(basis_new, (-1, in_size, tf.shape(basis_new)[-1]))  # (N, in_size, M)
    Sn = tf.reshape(spline_old,  (-1, in_size, tf.shape(spline_old)[-1]))  # (N, in_size, U)
    Bn = tf.transpose(Bn, [1, 0, 2])  # (in_size, N, M)
    Sn = tf.transpose(Sn, [1, 0, 2])  # (in_size, N, U)
    W  = tf.linalg.lstsq(Bn, Sn, l2_regularizer=l2_reg, fast=fast)  # (in_size, M, U)
    return W



def calc_spline_values(x: tf.Tensor, grid: tf.Tensor, spline_order: int) -> tf.Tensor:
    """
    Calcula los valores de la base B-spline usando broadcasting.
    Es robusto para tensores de entrada de cualquier rango (ej. 2D o 3D).
    """
    x = tf.expand_dims(x, axis=-1)
    bases = tf.logical_and(tf.greater_equal(x, grid[:, :-1]), tf.less(x, grid[:, 1:]))
    bases = tf.cast(bases, x.dtype)

    for k in range(1, spline_order + 1):
        denominator1 = grid[:, k:-1] - grid[:, : -(k + 1)]
        term1 = (x - grid[:, : -(k + 1)]) / (denominator1 + 1e-8) * bases[..., :-1]
        
        denominator2 = grid[:, k + 1 :] - grid[:, 1:(-k)]
        term2 = (grid[:, k + 1 :] - x) / (denominator2 + 1e-8) * bases[..., 1:]
        
        bases = term1 + term2
        
    return bases

class DenseKAN(Layer):
    def __init__(self,
        units: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Union[Tuple[float], List[float]] = [-4.0, 4.0],
        basis_activation: Union[str, Callable] = 'silu',
        spline_initialize_stddev: float = 0.1,
        use_bias: bool = True,
        track_inputs: bool = False,           
        track_buffer_size: int = 4096,        
        track_sample_size: int = 256,         
        track_random: bool = True,            
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.use_bias = use_bias
        self.spline_initialize_stddev = spline_initialize_stddev
        self.track_inputs = track_inputs
        self.track_buffer_size = int(track_buffer_size)
        self.track_sample_size = int(track_sample_size)
        self.track_random = bool(track_random)

        if isinstance(basis_activation, str):
            self.basis_activation = tf.keras.activations.get(basis_activation)
        else:
            self.basis_activation = basis_activation

    def build(self, input_shape: Any):
        in_size = input_shape[-1]
        self.in_size = in_size
        self.spline_basis_size = self.grid_size + self.spline_order

        h = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        grid_points = tf.linspace(self.grid_range[0] - self.spline_order * h,
                                  self.grid_range[1] + self.spline_order * h,
                                  self.grid_size + 2 * self.spline_order + 1)

        self.grid = self.add_weight(
            name='grid',
            shape=(1, self.grid_size + 2 * self.spline_order + 1),
            initializer=tf.keras.initializers.Constant(grid_points[None, :]),
            trainable=False
        )

        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.in_size, self.spline_basis_size, self.units),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.spline_initialize_stddev),
            trainable=True,
        )

        self.scale_factor = self.add_weight(
            name="scale_factor",
            shape=(self.in_size, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias", shape=(self.units,),
                initializer=tf.keras.initializers.Zeros(), trainable=True,
            )

        # === NUEVO: buffers de tracking (no entrenables) ===
        if self.track_inputs:
            self._buf = self.add_weight(
                name="__track_buffer__", shape=(self.track_buffer_size, self.in_size),
                initializer="zeros", trainable=False
            )
            self._buf_ptr = self.add_weight(
                name="__track_ptr__", shape=(), initializer="zeros",
                trainable=False, dtype=tf.int64
            )
            self._buf_filled = self.add_weight(
                name="__track_filled__", shape=(), initializer="zeros",
                trainable=False, dtype=tf.int64
            )
            self._run_min = self.add_weight(
                name="__track_run_min__", shape=(self.in_size,),
                initializer=tf.keras.initializers.Constant(1e9),
                trainable=False
            )
            self._run_max = self.add_weight(
                name="__track_run_max__", shape=(self.in_size,),
                initializer=tf.keras.initializers.Constant(-1e9),
                trainable=False
            )

        super().build(input_shape)

    def _track(self, inputs):
        x = tf.stop_gradient(tf.reshape(inputs, (-1, self.in_size)))
        n = tf.shape(x)[0]
        m = tf.minimum(tf.cast(self.track_sample_size, tf.int32), n)
        idx = tf.range(n) if not self.track_random else tf.random.shuffle(tf.range(n))
        idx = idx[:m]
        samp = tf.gather(x, idx)  # (m, in_size)

        buf_size = tf.shape(self._buf)[0]
        ptr = self._buf_ptr
        pos = (tf.range(tf.cast(m, tf.int64), dtype=tf.int64) + ptr) % tf.cast(buf_size, tf.int64)

        updated = tf.tensor_scatter_nd_update(self._buf, tf.expand_dims(pos, 1), samp)
        self._buf.assign(updated)
        self._buf_ptr.assign((ptr + tf.cast(m, tf.int64)) % tf.cast(buf_size, tf.int64))
        self._buf_filled.assign(tf.minimum(self._buf_filled + tf.cast(m, tf.int64), tf.cast(buf_size, tf.int64)))

        bmin = tf.reduce_min(samp, axis=0)
        bmax = tf.reduce_max(samp, axis=0)
        self._run_min.assign(tf.minimum(self._run_min, bmin))
        self._run_max.assign(tf.maximum(self._run_max, bmax))

    def call(self, inputs):
        if getattr(self, "track_inputs", False):
            self._track(inputs)

        # 1) spline(x)
        spline_basis = calc_spline_values(inputs, self.grid, self.spline_order)
        spline_out = tf.einsum("...ik,iko->...io", spline_basis, self.spline_kernel)

        # 2) base b(x)
        basis_out = self.basis_activation(inputs)
        basis_out = tf.expand_dims(basis_out, axis=-1)

        # 3) combinar y escalar
        combined_out = basis_out + spline_out
        scaled_out = combined_out * self.scale_factor

        # 4) sumar sobre dim de entrada
        output = tf.reduce_sum(scaled_out, axis=-2)
        if self.use_bias:
            output += self.bias
        return output

    def get_tracked_samples(self) -> Union[tf.Tensor, None]:
        if not getattr(self, "track_inputs", False):
            return None
        valid = tf.cast(self._buf_filled, tf.int32)
        return None if tf.equal(valid, 0) else self._buf[:valid]

    def update_grid_from_samples(self,
                                 inputs: tf.Tensor,
                                 margin: float = 0.05,
                                 grid_eps: float = 0.02,
                                 l2_reg: float = 1e-6,
                                 fast: bool = True):
        # 1) target spline con grid/kernel actuales
        spline_basis_old = calc_spline_values(inputs, self.grid, self.spline_order)
        spline_out_old   = tf.einsum("...ik,iko->...io", spline_basis_old, self.spline_kernel)
        # 2) grid nuevo
        new_grid = _build_uniform_grid_from_data(inputs, self.grid_size, self.spline_order,
                                                 self.grid, margin, grid_eps, self.grid.dtype)
        # 3) base con grid nuevo y resolver LS
        spline_basis_new = calc_spline_values(inputs, new_grid, self.spline_order)
        W_new = _fit_spline_kernel_ls(spline_basis_new, spline_out_old, l2_reg=l2_reg, fast=fast)
        # 4) asignar
        self.grid.assign(new_grid)
        self.spline_kernel.assign(tf.cast(W_new, self.spline_kernel.dtype))

    def update_grid_from_tracked(self,
                                 min_required: int = 128,
                                 margin: float = 0.05,
                                 grid_eps: float = 0.02,
                                 l2_reg: float = 1e-6,
                                 fast: bool = True):
        """
        Usa las muestras trackeadas (agnóstico del modelo) para recalibrar grid+kernel.
        """
        samples = self.get_tracked_samples()
        if samples is None:
            return
        n = tf.shape(samples)[0]
        if tf.less(n, min_required):
            return
        self.update_grid_from_samples(samples, margin=margin, grid_eps=grid_eps,
                                      l2_reg=l2_reg, fast=fast)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
            "spline_initialize_stddev": self.spline_initialize_stddev,
            "use_bias": self.use_bias,
            "basis_activation": tf.keras.activations.serialize(self.basis_activation),
            # NUEVO:
            "track_inputs": self.track_inputs,
            "track_buffer_size": self.track_buffer_size,
            "track_sample_size": self.track_sample_size,
            "track_random": self.track_random,
        })
        return config




class DenseKANAutoGridUpdate(tf.keras.callbacks.Callback):
    def __init__(self, model_with_kan,
                 on_epoch_end: bool = True,
                 every_n_batches: int = 0,
                 min_required: int = 128,
                 margin: float = 0.05,
                 grid_eps: float = 0.02,
                 l2_reg: float = 1e-6,
                 fast: bool = True):
        super().__init__()
        self.root = model_with_kan
        self.on_epoch_end_flag = on_epoch_end
        self.every_n_batches = every_n_batches
        self.min_required = min_required
        self.kw = dict(margin=margin, grid_eps=grid_eps, l2_reg=l2_reg, fast=fast)
        self._bc = 0
        self._layers = None  # cache

    def _collect(self):
        if self._layers is not None:
            return self._layers
        found = []
        def walk(layer):
            if isinstance(layer, DenseKAN):
                found.append(layer)
            if hasattr(layer, "layers"):
                for sub in layer.layers:
                    walk(sub)
        walk(self.root)
        self._layers = found
        return found

    def _do_update(self):
        for layer in self._collect():
            layer.update_grid_from_tracked(min_required=self.min_required, **self.kw)

    def on_train_batch_end(self, batch, logs=None):
        if not self.on_epoch_end_flag and self.every_n_batches > 0:
            self._bc += 1
            if self._bc % self.every_n_batches == 0:
                print("Updating_grid")
                self._do_update()

    def on_epoch_end(self, epoch, logs=None):
        if self.on_epoch_end_flag:
            print("Updating_grid")
            self._do_update()
