import typing

import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from tqdm import tqdm
from im2col_cython import im2col_cython, col2im_cython

# TODO: work on the NN plotter
# TODO: tidy layer.shape /.shape_out mess
# TODO: Handle multidimensional inputs (examples, *, n)
# TODO: swap (n_features, m_examples) to (m_examples, n_features)


class Layer:
    def __init__(self, shape):
        self.shape = shape
        # self.shape_out = self.shape
        # self.shape_in = None
        self.gradients = {}
        self.parameters = {}
        self._parameters_init_spec = {}
        self.cache = {}
        self.trainable = False
        self.draw_string = 'Layer'
        self.type = 'Layer'
        self.prevLayer = None
        self.name = ''

    def __str__(self):
        s = (f'Layer: {self.type}'
             f', IN: {self.shape_in}'
             f', OUT: {self.shape_out}')
        if self.trainable:
            s += (f', PARAM: {[(param, value.shape) for param, value in self.parameters.items()]}')
        return s

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape if hasattr(shape, "__len__") else (shape,)

    def add_parameter(self, name, initializer, rule):
        initializer = self._parse_initializer(initializer)
        self._parameters_init_spec[name] = (initializer, rule)

    def init_shapes(self, shape_in):
        # shape_in = self.prevLayer.shape_out
        self.shape_in = shape_in
        # self.shape_out = self.shape_in would be a better default ?
        self.shape_out = self.shape

    def init_parameters(self, shape_in):
        self.init_shapes(shape_in)
        FAN_IN = np.prod(self.shape_in)
        FAN_OUT = np.prod(self.shape_out)
        for param, (initializer, rule) in self._parameters_init_spec.items():
            shape = self._parse_shape_rule(shape_in, rule)
            self.parameters[param] = initializer(shape, FAN_IN, FAN_OUT)

    def _parse_shape_rule(self, shape_in, rule):
        shape = []
        for spec in rule:
            if isinstance(spec, int):
                shape.append(spec)
            elif isinstance(spec, str):
                if spec[0] == "i":
                    source = shape_in
                elif spec[0] == "o":
                    source = self.shape
                else:
                    raise TypeError(f"Invalid source shape specification {spec[0]}")
                source_axis = int(spec[1:])
                shape.append(source[source_axis])
        return tuple(shape)

    def _parse_initializer(self, initializer):
        """
        """
        if initializer == "normal":
            return RandomInitializer()
        elif initializer == "he":
            return RandomInitializer(
                scaling=lambda FAN_IN, FAN_OUT: np.sqrt(2 / FAN_IN)
            )
        elif initializer == "he_uniform":
            return RandomInitializer(uniform=True,
                scaling=lambda FAN_IN, FAN_OUT: np.sqrt(6 / FAN_IN)
            )
        elif initializer == "xavier":
            return RandomInitializer(
                scaling=lambda FAN_IN, FAN_OUT: np.sqrt(2 / (FAN_IN + FAN_OUT))
            )
        elif initializer == "xavier_uniform":
            return RandomInitializer(uniform=True,
                scaling=lambda FAN_IN, FAN_OUT: np.sqrt(6 / (FAN_IN + FAN_OUT)))
        elif initializer == "ones":
            return OnesInitializer()
        elif initializer == "zeros":
            return ZerosInitializer()
        else:
            return initializer

    def forward(self, *args, **kwargs):
        pass

    def backward(self, *args, **kwargs):
        pass

    def norm(self):
        try:
            return np.sum(np.square(self.parameters["W"]))
        except KeyError:
            return 0

    def draw(self, ax, x_inp, x0, y0, max_height, r, v_spacing, h_spacing, update):
        if update:
            return self.update_drawing(self, ax)

        x = self.draw_nodes(ax, x0, y0, r, h_spacing, v_spacing, max_height)
        x = self.draw_edges(ax, x_inp, x0, y0, r, h_spacing, v_spacing, max_height) or x

        return x

    def update_drawing(self, *args):
        return

    def draw_nodes(self, ax, x0, y0, r, h_spacing, v_spacing, max_height):
        x = x0
        n = self.shape[0]
        height = v_spacing * (n - 1)
        y = y0 + 1/2 * (max_height - height)
        for i in range(n):
            c = plt.Circle((x, y), r, color="w", ec="k", zorder=3)
            ax.add_artist(c)
            ax.annotate(self.draw_string, xy=(x, y), ha="center", va="center", fontsize=8, zorder=4)
            y += v_spacing
        return x, x

    def draw_edges(self, ax, x_inp, x0, y0, r, h_spacing, v_spacing, max_height):
        n = self.shape[0]
        n_inputs = n #self.shape_in[0]
        height = v_spacing * (n - 1)
        height_inputs = v_spacing * (n_inputs - 1)
        y = y0 + 1/2 * (max_height - height)
        y_inp = y0 + 1/2 * (max_height - height_inputs)
        x = x0
        # Edges
        for i in range(n):
            X = (x, x_inp)
            Y = (y, y_inp)
            l = plt.Line2D(X, Y, c="k", linewidth=1)
            ax.add_artist(l)
            y += v_spacing
            y_inp += v_spacing
        return x


class LinearLayer(Layer):
    def __init__(self, shape, initializer="normal"):
        super().__init__(shape)
        self.add_parameter("W", initializer, ("o0", "i0"))
        self.add_parameter("b", "zeros", ("o0", 1))
        self.trainable = True
        self.draw_string = r'$Wx+b$'
        self.type = 'Linear'

    def forward(self, X, update=True, **kwargs):
        W, b = self.parameters["W"], self.parameters["b"]
        Z = np.dot(W, X) + b
        if update:
            self.cache["X"] = X
        return Z

    def backward(self, dZ, lambd=0, **kwargs):
        W = self.parameters["W"]
        X = self.cache["X"]
        m = dZ.shape[1]

        dA = np.dot(W.T, dZ)
        dW = 1 / m * (np.dot(dZ, X.T) + lambd * W)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        self.gradients["W"] = dW
        self.gradients["b"] = db
        return dA

    def init_shapes(self, shape_in):
        self.shape_in = shape_in
        self.shape_out = self.shape

    def update_drawing(self, *args):
        for w, artist in zip(self.parameters['W'].flatten(), self.updatable_artists):
            col = "r" if w >= 0 else "b"
            artist.set_color(col)
            artist.set_linewidth(w)
        return

    def draw_edges(self, ax, x_inp, x0, y0, r, h_spacing, v_spacing, max_height):
        n = self.shape[0]
        n_inputs = self.shape_in[0]
        height = v_spacing * (n - 1)
        height_inputs = v_spacing * (n_inputs - 1)
        y = y0 + 1/2 * (max_height - height)
        x = x0
        w_artists = []
        for i in range(n):
            y_inp = y0 + 1/2 * (max_height - height_inputs)
            for j in range(n_inputs):
                X = (x, x_inp)
                Y = (y, y_inp)
                w = self.parameters["W"][i, j]
                col = "r" if w >= 0 else "b"
                l = plt.Line2D(X, Y, c=col, linewidth=w)
                w_artists.append(l)
                ax.add_artist(l)
                y_inp += v_spacing
            y += v_spacing

        self.updatable_artists = w_artists
        return x


class ActivationLayer(Layer):
    def __init__(self, shape, activation="relu"):
        super().__init__(shape)
        self.activation = activation
        self.draw_string = activation
        self.type = 'Activation'

    # def _wormhole(self):
    #     skip = self.skip
    #     if not skip:
    #         return
    #     prev = self
    #     for i in range(skip):
    #         prev = prev.prevLayer
    #         while not isinstance(prev, ActivationLayer):
    #             if prev is None:
    #                 return prev
    #             prev = prev.prevLayer
    #     return prev

    def __str__(self):
        s = super().__str__()
        s += (f', ACTIVATION: {self.activation}')
        return s

    def init_shapes(self, shape_in):
        self.shape_in = shape_in
        self.shape_out = shape_in
        # self.skipLayer = self._wormhole()

    def forward(self, Z, update=True, **kwargs):
        # if self.skipLayer:
        #     Z += self.skipLayer.cache['A']
        g = self._activation_function_forward
        A = g(Z)
        if update:
            self.cache["Z"] = Z
            self.cache["A"] = A
        return A

    def backward(self, dA, **kwargs):
        g_prime = self._activation_function_backward
        Z = self.cache["Z"]
        dAdZ = g_prime(Z)
        if dAdZ.shape != dA.shape:  # expect dAdZ to be of shape (m, Jacobian(n, n))
            dZ = np.einsum("ijk,ki->ji", dAdZ, dA)
        else:
            dZ = dA * dAdZ
        return dZ

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, activation):
        """
        Read self.settings and set the corresponding activation function.
        if self.settings['activation'] is not a valid name for a predefined function,
        it is assumed to be a list of functions itself of the form:
        [foo(z) -> a, dfoo(z) -> da/dz]
        """
        if activation == "relu":
            self._activation = activation
            self._activation_function_forward = self._relu_f
            self._activation_function_backward = self._relu_b
        elif activation == "sigmoid":
            self._activation = activation
            self._activation_function_forward = self._sigmoid_f
            self._activation_function_backward = self._sigmoid_b
        elif activation == "tanh":
            self._activation = activation
            self._activation_function_forward = self._tanh_f
            self._activation_function_backward = self._tanh_b
        elif activation == "linear":
            self._activation = activation
            self._activation_function_forward = self._lin_f
            self._activation_function_backward = self._lin_b
        elif activation == "softmax":
            self._activation = activation
            self._activation_function_forward = self._softmax_f
            self._activation_function_backward = self._softmax_b
        else:  # allow custom function
            self._activation = activation
            self._activation_function_forward = self.activation[0]
            self._activation_function_backward = self.activation[1]


    def draw_nodes(self, ax, x0, y0, r, h_spacing, v_spacing, max_height):
        x = x0
        n = self.shape[0]
        height = v_spacing * (n - 1)
        y = y0 + 1/2 * (max_height - height)

        if self.activation == "softmax":
            c = plt.Rectangle((x-r, y-r), width=2*r, height=height+2*r, color="w", ec="k", zorder=3)
            ax.add_artist(c)
            ax.annotate(f"${self.activation}$", xy=(x, y), ha="center", va="center", fontsize=8, zorder=4, weight="bold")
        else:
            super().draw_nodes(ax, x0, y0, r, h_spacing, v_spacing, max_height)

        return x

    def _sigmoid_f(self, z):
        return np.exp(-np.logaddexp(0, -z))
        # return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def _sigmoid_b(self, z):
        return self.cache["A"] * (1 - self.cache["A"])

    def _tanh_f(self, z):
        return np.tanh(z)

    def _tanh_b(self, z):
        return 1 - np.square(self.cache["A"])

    def _relu_f(self, z):
        return np.maximum(0, z)

    def _relu_b(self, z):
        return 1.0 * (z > 0)

    def _lin_f(self, z):
        return z

    def _lin_b(self, z):
        return 1

    def _softmax_f(self, z):
        z -= z.max()
        t = np.exp(z)
        a = t / t.sum(axis=0)
        return a

    def _softmax_b(self, z):
        """
        this function returns 1, dL/dz is computed directly as a - y by
        _softmax_loss_b and fed as if it was dA to the backprop
        """
        n, m = z.shape
        a = self.cache["A"]  # (n, m)
        dadz = np.einsum("ij,ik->jik", a, np.eye(n, n)) - np.einsum("ij,kj->jik", a, a)
        # dz = np.einsum('ijk,ki->ji', dadz, da)  # (m, n)
        # for each j: (each a in da/dz)
        #     for each i: (each m example)
        #         sum = 0
        #         for each k: (each z in da/dz)
        #             sum += dadz[i][j][k] * da[k][i]
        #         R[j][i] = sum
        return dadz


class DropoutLayer(Layer):
    def __init__(self, shape, keep_prob=1):
        super().__init__(shape)
        self.keep_prob = keep_prob
        self.draw_string = 'dropout'
        self.type = 'DropOut'

    def __str__(self):
        s = super().__str__()
        s += f', P: {self.keep_prob}'
        return s

    def forward(self, A, update=True, **kwargs):
        if update:
            keep_prob = self.keep_prob
            dropout_mask = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A = A * dropout_mask / keep_prob
            self.cache["dropout_mask"] = dropout_mask
            return A
        return A

    def backward(self, dA, **kwargs):
        keep_prob = self.keep_prob
        dropout_mask = self.cache["dropout_mask"]
        return dA * dropout_mask / keep_prob


class BatchNormLayer(Layer):
    def __init__(self, shape):
        super().__init__(shape)
        self.add_parameter("gamma", "ones", ("o0", 1))
        self.add_parameter("beta", "ones", ("o0", 1))
        self.cache["Z_mu_avg"] = ZerosInitializer()((self.shape[0], 1))
        self.cache["Z_ivar_avg"] = ZerosInitializer()((self.shape[0], 1))
        self.trainable = True
        self.draw_string = 'BatchNorm'
        self.type = 'BatchNorm'

    def forward(self, Z, update=True, eps=1e-15, rav=0.9, **kwargs):
        gamma, beta = self.parameters["gamma"], self.parameters["beta"]
        Z_mu_avg, Z_ivar_avg = self.cache["Z_mu_avg"], self.cache["Z_ivar_avg"]
        m = Z.shape[1]

        if update:
            Z_mu = np.mean(Z, axis=1, keepdims=True)
            Z_zero = Z - Z_mu
            Z_var = np.sum(np.square(Z_zero), axis=1, keepdims=True) / m
            Z_ivar = 1 / np.sqrt(Z_var + eps)
            Z_norm = Z_zero * Z_ivar
            Z_t = gamma * Z_norm + beta

            self.cache["Z_ivar"] = Z_ivar
            self.cache["Z_norm"] = Z_norm
            self.cache["Z_mu_avg"] = rav * Z_mu_avg + (1 - rav) * Z_mu
            self.cache["Z_ivar_avg"] = rav * Z_ivar_avg + (1 - rav) * Z_ivar

        else:
            Z_mu = Z_mu_avg
            Z_ivar = Z_ivar_avg
            Z_norm = (Z - Z_mu) * Z_ivar
            Z_t = gamma * Z_norm + beta

        return Z_t

    def backward(self, dZ_t, **kwargs):
        Z_norm = self.cache["Z_norm"]
        Z_ivar = self.cache["Z_ivar"]
        gamma = self.parameters["gamma"]
        m = dZ_t.shape[1]

        dgamma = np.sum(dZ_t * Z_norm, axis=1, keepdims=True)  # / m
        dbeta = np.sum(Z_norm, axis=1, keepdims=True)  # / m
        dZ_norm = dZ_t * gamma
        # fmt: off
        dZ = 1 / m * Z_ivar * (
                m * dZ_norm
                - np.sum(dZ_norm, axis=1, keepdims=True)
                - Z_norm * np.sum(dZ_norm * Z_norm, axis=1, keepdims=True)
        )
        # fmt: on
        self.gradients["gamma"] = dgamma
        self.gradients["beta"] = dbeta
        return dZ


class CompositeLayer(Layer):
    def __init__(
        self,
        shape,
        initializer="normal",
        activation="relu",
        keep_prob=1,
        batch_norm=False,
    ):
        super().__init__(shape)
        self.component_layers: typing.List[Layer] = []
        self.component_layers.append(
            LinearLayer(self.shape, initializer=initializer)
        )
        if batch_norm:
            self.component_layers.append(
                BatchNormLayer(self.shape)
            )
        self.component_layers.append(ActivationLayer(self.shape, activation=activation))
        if keep_prob < 1:
            self.component_layers.append(DropoutLayer(self.shape, keep_prob=keep_prob))

    def __iter__(self):
        for layer in self.component_layers:
            yield layer

    # def init_parameters(self, shape_in):
    #     for layer in self.component_layers:
    #         layer.init_parameters(shape_in)

    # def forward(self, X, **kwargs):
    #     """
    #     :param X: layer inputs np.ndarray of shape (n_features, m_examples)
    #     :param kwargs: dicionary of kwargs accepted by individual
    #     forward functions

    #     :return: the otput (processed inputs) of the layer; np.ndarray of
    #     shape(n)
    #     """
    #     for layer in self.component_layers:
    #         X = layer.forward(X, **kwargs)
    #     return X

    # def backward(self, dA, **kwargs):
    #     """
    #     :param dA: derivatives of the loss function w.r.t. the layer output
    #     :return: derivatives of the loss function w.r.t. the layer input
    #     """
    #     for layer in reversed(self.component_layers):
    #         dA = layer.backward(dA, **kwargs)
    #     return dA

    def norm(self):
        return self.component_layers[0].norm()

    def draw(self, *args, **kwargs):
        pass


class ConvolutionLayer(Layer):
    def __init__(self, shape, mode=None, padding=[0, 0], stride=1, initializer='normal'):
        super().__init__(shape)
        # shape (nf,fh,fw)
        self.mode = mode
        self.padding = padding
        self.stride = stride
        self.initializer = initializer
        d_out, fh, fw = shape
        self.add_parameter('W', initializer, (d_out, 'i0', fh, fw))
        self.add_parameter('b', initializer, (d_out, 1))
        self.trainable = True
        self.type = 'Convolution'

    def __str__(self):
        s = super().__str__()
        s += (f', FILTERS: {self.shape[0]} x {self.shape[1]} x {self.shape[2]}'
              f', PADDING: {self.padding}'
              f', STRIDE: {self.stride}')
        return s

    def forward(self, X, update=False, **kwargs):
        # X (m, d_in, h, w)
        # W (d_out, d_in, fh, fw) -> (d_out, d_in * fh * fw)
        # b (d_out, 1)
        # col (d_in * fh * fw, m * h_out * w_out)
        # Z_col (d_out, m * h_out * w_out)
        # Z (m, d_out, h_out, w_out)
        # h_out = (h + 2p - fh)/s + 1
        # w_out = (w + 2p - fw)/s + 1)
        m, d_in, h, w = X.shape
        d_out, fh, fw = self.shape
        s = self.stride
        p = self.padding
        W = self.parameters['W']
        b = self.parameters['b']
        W_col = W.reshape(W.shape[0], -1)
        _, h_out, w_out = self.shape_out

        # X_pad = np.pad(X, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), constant_values=(0, 0))

        # X_col = im2col_indices(X, fh, fw, padding=p, stride=s)
        X_col = im2col_cython(X, fh, fw, p[0], s) # only square padding for now!
        Z_col = np.dot(W_col, X_col) + b
        Z = Z_col.reshape(d_out, h_out, w_out, m).transpose(3, 0, 1, 2)

        if update:
            self.cache['X_col'] = X_col
            self.cache['X'] = X
        return Z

    def backward(self, dZ, update=True, **kwargs):
        X = self.cache['X']
        X_col = self.cache['X_col']
        W = self.parameters['W']
        s = self.stride
        p = self.padding
        d_out, fh, fw = self.shape
        m = dZ.shape[0]

        db = np.sum(dZ, axis=(0, 2, 3)) / m
        db = db.reshape(d_out, -1)

        dZ_col = dZ.transpose(1, 2, 3, 0).reshape(d_out, -1)
        dW_col = dZ_col @ X_col.T
        dW = dW_col.reshape(W.shape)

        W_col = W.reshape(d_out, -1)
        dX_col = W_col.T @ dZ_col
        # dX = col2im_indices(dX_col, X.shape, fh, fw, padding=p, stride=s)
        dX = col2im_cython(dX_col, *X.shape, fh, fw, p[0], s) # only square padding for now!

        if update:
            self.gradients['W'] = dW
            self.gradients['b'] = db

        return dX

    def init_shapes(self, shape_in):
        self.shape_in = shape_in
        d_in, h_in, w_in = shape_in
        d_out, fh, fw = self.shape
        s = self.stride
        if self.mode == 'same':
            self.padding[0] = int(((s - 1) * h_in - s + fh) / 2)
            self.padding[1] = int(((s - 1) * w_in - s + fw) / 2)
        p = self.padding
        h_out = int((h_in + 2*p[0] - fh)/s + 1)
        w_out = int((w_in + 2*p[1] - fw)/s + 1)
        self.shape_out = (d_out, h_out, w_out)

    def update_drawing(self, *args):
        W = self.parameters['W'].sum(axis=1)
        for i, ins in enumerate(self.filter_axs):
            ins.images.clear()
            ins.imshow(W[i], cmap='gray')
        return

    def draw_nodes(self, ax, x0, y0, r, h_spacing, v_spacing, max_height):
        n = self.shape_out[0]
        d_out, h_out, w_out = self.shape_out
        W = self.parameters['W'].sum(axis=1)
        height = v_spacing * (n - 1)

        x = x0
        y = y0 + 1/2 * max_height
        ax.annotate(r'$\ast$', (x, y), fontsize=8, ha='center', va='center')

        x += h_spacing
        y = y0 + 1/2 * (max_height - height)

        filter_axs = []
        for i in range(n):
            ins = ax.inset_axes((x-r, y-r, 2*r, 2*r), transform=ax.transData)
            ins.imshow(W[i], cmap='gray')
            ins.axis('off')
            filter_axs.append(ins)
            y += v_spacing
        self.filter_axs = filter_axs

        x += h_spacing
        y = y0 + 1/2 * max_height

        draw_volume(ax, (x - r, y - r), d_out, h_out, w_out, 2*r)
        return x

    def draw_edges(*args):
        pass


def get_im2col_indices(x_shape, field_height, field_width, padding=(0, 0), stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    # assert (H + 2 * padding[0] - field_height) % stride == 0
    # assert (W + 2 * padding[1] - field_height) % stride == 0
    out_height = int((H + 2 * padding[0] - field_height) / stride + 1)
    out_width = int((W + 2 * padding[1] - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=(0, 0), stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    ph, pw = int(padding[0]), int(padding[1])
    x_padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height, field_width, padding=(0, 0), stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding[0], W + 2 * padding[1]
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    ph_s = padding[0] if padding[0] != 0 else None
    ph_e = -padding[0] if padding[0] != 0 else None
    pw_s = padding[1] if padding[1] != 0 else None
    pw_e = -padding[1] if padding[1] != 0 else None
    return x_padded[:, :, ph_s:ph_e, pw_s:pw_e]
    return x_padded

# def img2col(img, fshape, stride=1):
    # thanks to https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
    # m, d, h, w = img.shape
    # fh, fw = fshape
    # # col_extent = int(np.floor((h - fw)/stride + 1))
    # # row_extent = int(np.floor((w - fh)/stride + 1))
    # col_extent = h - fw + 1
    # row_extent = w - fh + 1

    # m_idx = np.arange(m)[:, None, None] * d * h * w  # (m, 1, 1)

    # d_idx = h * w * np.arange(d)[None, :, None]  # (1, d, 1)

    # start_idx = np.arange(fh)[None, :, None] * w + np.arange(
    #     fw
    # )  # (1, f0, 1) + (f1,) -> (1, f0, f1)
    # start_idx = (d_idx + start_idx.ravel()).reshape(
    #     (-1, fh, fw)
    # )  # (1, d, 1) + (f0 * f1,) -> (1, d, f0*f1)

    # # (1, (w - f0 + 1)/s, 1) + ((h - f1 + 1)/s,) -> (1, (w - f0 + 1)/s, (h - f1 + 1)/s)
    # offset_idx = np.arange(row_extent, step=stride)[None, :, None] * w + np.arange(
    #     col_extent, step=stride
    # )
    # # (m, 1, 1) + (1, d * f0 * f1, 1) -> (m, d * f0 * f1, 1) + ((w - f0 + 1)/s * (h - f1 + 1)/s,) -> (m, d * f0 * f1, (w - f0 + 1)/s * (h - f1 + 1)/s)
    # act_idx = m_idx + start_idx.ravel()[None, :, None] + offset_idx.ravel()
    # out = np.take(img, act_idx)  # (m, d * f0 * f1, (w - f0 + 1)/s * (h - f1 + 1)/s)
    # return out


class PoolingLayer(Layer):
    def __init__(self, shape, mode=None, operation='max', padding=[0, 0], stride=1):
        super().__init__(shape)
        self.stride = stride
        self.padding = padding
        self.mode = mode
        self.operation = operation
        self.type = 'Pooling'

    def __str__(self):
        s = super().__str__()
        s += (f', PADDING: {self.padding}'
              f',  STRIDE: {self.stride}'
              f',  OPERATION: {self.operation}')
        return s

    def forward(self, X, update=False, **kwargs):
        # X       (              m, d_in, h, w)
        # col     ( d_in * fh * fw, m * h_out * w_out)
        # Z_col   (          d_out, m * h_out * w_out)
        # h_out = ( h + 2p - fh)/s + 1
        # w_out = ( w + 2p - fw)/s + 1)
        m, d_in, h, w = X.shape
        d_out, fh, fw = self.shape
        d_out = d_in
        s = self.stride
        p = self.padding
        _, h_out, w_out = self.shape_out

        X_reshaped = X.reshape(m * d_in, 1, h, w)
        # X_col = im2col_indices(X_reshaped, fh, fw, padding=p, stride=s) # -> (fh * fw, d_in * m * h_out * w_out)
        X_col = im2col_cython(X_reshaped, fh, fw, p[0], s) # -> (fh * fw, d_in * m * h_out * w_out)
        # X_col.reshape(d_in, fh * fw, m * h_out * w_out)
        if self.operation == 'max':
            idx = np.argmax(X_col, axis=0)
            out = X_col[idx, range(idx.size)] # -> (d_in * m * h_out * w_out,)
        if self.operation == 'avg':
            out = np.mean(X_col, axis=0)
        out = out.reshape(h_out, w_out, m, d_out).transpose(2, 3, 0, 1)

        if update:
            self.cache['X_col'] = X_col
            self.cache['X'] = X
            try:
                self.cache['idx'] = idx
            except NameError:
                pass
        return out

    def backward(self, dX, update=True, **kwargs):
        X = self.cache['X']
        X_col = self.cache['X_col']

        m, d_in, h, w = X.shape
        d_out, fh, fw = self.shape
        s = self.stride
        p = self.padding

        if self.operation == 'max':
            idx = self.cache['idx']
            dX_col = np.zeros_like(X_col)  # -> (fh * fw, d_in * m * h_out * w_out)
            dX_flat = dX.transpose(2, 3, 0, 1).reshape(-1)  # (fh * fw * m * d,)
            dX_col[idx, range(idx.size)] = dX_flat

        if self.operation == 'avg':
            dX_col = X_col / (fh * fw)

        # dX = col2im_indices(dX_col, (m * d_in, 1, h, w), fh, fw, padding=p, stride=s)
        dX = col2im_cython(dX_col, m * d_in, 1, h, w, fh, fw, p[0], s)
        dX = dX.reshape(X.shape)

        return dX

    def init_shapes(self, shape_in):
        self.shape_in = shape_in
        d_in, h_in, w_in = shape_in
        d_out, fh, fw = self.shape
        s = self.stride
        if self.mode == 'same':
            self.padding[0] = int(((s - 1) * h_in - s + fh) / 2)
            self.padding[1] = int(((s - 1) * w_in - s + fw) / 2)
        p = self.padding
        h_out = int((h_in + 2*p[0] - fh)/s + 1)
        w_out = int((w_in + 2*p[1] - fw)/s + 1)
        self.shape_out = (d_in, h_out, w_out)

    def draw_nodes(self, ax, x0, y0, r, h_spacing, v_spacing, max_height):
        n = self.shape_out[0]
        d_out, h_out, w_out = self.shape_out
        height = v_spacing * (n - 1)

        x = x0
        y = y0 + 1/2 * max_height
        ax.annotate(rf'$\vec{ {self.operation} }$', (x, y), fontsize=8, ha='center', va='center')

        x += h_spacing
        draw_volume(ax, (x -r , y -r), d_out, h_out, w_out, 2*r)
        return x

    def draw_edges(*args):
        pass


class AdapterLayer(Layer):
    def __init__(self, shape):
        super().__init__(shape)
        # shape (nf,fh,fw)
        self.type = 'Adapter2D1D'

    def forward(self, X, update=False, **kwargs):
        orig_shape = X.shape
        m, d, h, w = orig_shape
        if update:
            self.cache['orig_shape'] = orig_shape
        return np.einsum('mdhw->dhwm', X).reshape(-1, m)

    def backward(self, dX, **kwargs):
        orig_shape = self.cache['orig_shape']
        return dX.T.reshape(*orig_shape)

    def init_shapes(self, shape_in):
        d_in, h_in, w_in = shape_in
        self.shape_out = (d_in * h_in * w_in, 1)
        self.shape_in = shape_in

    def draw_nodes(self, ax, x0, y0, r, h_spacing, v_spacing, max_height):
        x = x0
        n = self.shape_out[0]
        height = v_spacing * (n - 1)
        y = y0 + 1/2 * (max_height - height)
        R = plt.Rectangle((x-.5*r, y-r), r, height + 2*r, ec='k', fc='w')
        ax.add_artist(R)

        y = y0 + 1/2 * max_height
        ax.annotate(fr'${n}\times{1}$', (x, y), fontsize=8, ha='center', va='center')
        return x

    def draw_edges(*args):
        pass


class IdentityLayer(Layer):
    def __init__(self, shape):
        super().__init__(shape)

    def forward(self, X, **kwargs):
        return X

    def backward(self, dX, **kwargs):
        return dX


class BranchLayer(Layer):
    def __init__(self, shape, name, branch_layers=[]):
        super().__init__(shape)
        self.name = name
        self.branch_layers = branch_layers
        self.draw_string = 'Branch'
        self.type = 'Branch'

    def __str__(self):
        s = super().__str__()
        s += (f'  NAME: {self.name}'
              f'  BRANCH_LAYERS: {self.branch_layers}')
        return s

    def init_parameters(self, shape_in):
        super().init_parameters(shape_in)
        shape_in = self.shape_in
        for layer in self.branch_layers:
            layer.init_parameters(shape_in)
            shape_in = layer.shape_out

    def init_shapes(self, shape_in):
        self.shape_in = shape_in
        self.shape_out = self.shape_in

    def forward(self, X_main_path, **kwargs):
        X = X_main_path.copy()
        for layer in self.branch_layers:
            X = layer.forward(X, **kwargs)
        # this will be accessed by AddLayers that want to link to this branch
        self.cache['X'] = X
        return X_main_path

    def backward(self, dX_main_path, **kwargs):
        # this is set by the AddLayer downstream
        dX_branch = self.gradients['X'].copy()
        for layer in reversed(self.branch_layers):
            dX_branch = layer.backward(dX_branch, **kwargs)
        return dX_main_path + dX_branch


class JoinLayer(Layer):
    def __init__(self, shape, branch_name):
        super().__init__(shape)
        self.branch_name = branch_name
        self.branchLayer = []

    def __str__(self):
        s = super().__str__()
        s += (f'  BRANCH_NAME: {self.branch_name}')
        return s

    def get_branchLayer(self):
        prev = self
        while True:
            if prev.name == self.branch_name:
                self.branchLayer = prev
                break
            prev = prev.prevLayer

    def init_shapes(self, shape_in):
        self.shape_in = shape_in
        self.shape_out = self.shape_in
        self.get_branchLayer()

    def forward(self, X_main_path, **kwargs):
        X_main_path += self.branchLayer.cache['X']
        return X_main_path

    def backward(self, dX_main_path, **kwargs):
        self.branchLayer.gradients['X'] = dX_main_path
        return dX_main_path


class Initializer:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, shape, *args):
        pass


class OnesInitializer(Initializer):
    def __call__(self, shape, *args, **kwargs):
        return np.ones(shape)


class ZerosInitializer(Initializer):
    def __call__(self, shape, *args, **kwargs):
        return np.ones(shape)


class RandomInitializer(Initializer):
    def __init__(self, uniform=False, scaling=lambda FAN_IN, FAN_OUT: 1):
        self.scaling = scaling
        self.uniform = uniform

    def __call__(self, shape, FAN_IN, FAN_OUT, *args, **wjargs):
        if self.uniform:
            r = np.random.rand(*shape) * 2 - 1
        else:
            r = np.random.randn(*shape)
        return r * self.scaling(FAN_IN, FAN_OUT)


class Optimizer:
    def __init__(self, layer: Layer):
        self.layer = layer

    def initialize(self, *args, **kwargs):
        pass

    def update_parameters(self, *args, **kwargs):
        pass


class AdamOptimizer(Optimizer):
    def __init__(self, layer: Layer):
        super().__init__(layer)
        self.V: typing.Dict[Layer, typing.Dict] = {}
        self.S: typing.Dict[Layer, typing.Dict] = {}

    def initialize(self, layer):
        self.V.setdefault(layer, {})
        self.S.setdefault(layer, {})
        for theta in layer.parameters.keys():
            self.V[layer][theta] = np.zeros_like(layer.parameters[theta])
            self.S[layer][theta] = np.zeros_like(layer.parameters[theta])

    def update_parameters(
        self,
        layer,
        learning_rate,
        t,
        *args,
        beta1=0.9,
        beta2=0.999,
        eps=1e-15,
        **kwargs,
    ):
        for theta in layer.parameters.keys():
            V_dtheta = self.V[layer][theta]
            S_dtheta = self.S[layer][theta]
            dtheta = layer.gradients[theta]

            V_dtheta = beta1 * V_dtheta + (1 - beta1) * dtheta
            S_dtheta = beta2 * S_dtheta + (1 - beta2) * np.square(dtheta)

            V_corr = V_dtheta / (1 - beta1 ** (t + 1))  # zero_indexed epoch
            S_corr = S_dtheta / (1 - beta2 ** (t + 1))

            self.V[layer][theta] = V_dtheta
            self.S[layer][theta] = S_dtheta
            layer.parameters[theta] -= learning_rate * V_corr / (np.sqrt(S_corr) + eps)


class GradientDescentOptimizer(Optimizer):
    def __init__(self, layer: Layer):
        super().__init__(layer)

    def update_parameters(self, layer, learning_rate, *args, **kwargs):
        for theta in layer.parameter_keys():
            layer.parameters[theta] -= learning_rate * layer.gradients[theta]


class NN:
    def __init__(
        self, *layers,
    ):
        """
        """
        self.layers: typing.List[Layer] = list(layers)
        self.trainable_layers = None
        self.unpacked_layers = None

    def compile(
        self,
        prediction_function="binary",
        loss_function="logloss",
        accuracy_metric="mad",
        optimizer="sgd",
    ):
        """
        Set the followings:
            - prediction_function
            - loss_function
            - accuracy_metric
            - optimizer
        Get:
            - trainable_layers
            - unpacked_layers
        Calls chain_layers
        """
        self.prediction_function = prediction_function
        self.loss_function = loss_function
        self.accuracy_metric = accuracy_metric
        self.optimizer = optimizer
        self.trainable_layers = [layer for layer in self.unpack_layers() if layer.trainable]
        self.unpacked_layers = self.unpack_layers()
        self.chain_layers()

    def chain_layers(self):
        for layer, prev in zip(self.unpacked_layers[1:], self.unpacked_layers[:-1]):
            layer.prevLayer = prev

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        if optimizer == "adam":
            self._optimizer = AdamOptimizer(self)
        elif optimizer == "sgd":
            self._optimizer = GradientDescentOptimizer(self)
        else:
            self._optimizer = optimizer

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function):
        self._loss_function = loss_function
        if loss_function == "logloss":
            self._loss_function_forward = self._logloss_f
            self._loss_function_backward = self._logloss_b
        elif loss_function == "cross_entropy":
            self._loss_function_forward = self._cross_entropy_f
            self._loss_function_backward = self._cross_entropy_b
        elif loss_function == "mae":
            self._loss_function_forward = self._mean_abs_error_f
            self._loss_function_backward = self._mean_abs_error_b
        elif loss_function == "mse":
            self._loss_function_forward = self._mean_sqr_error_f
            self._loss_function_backward = self._mean_sqr_error_b
        else:
            self._loss_function_forward = self.loss_function[0]
            self._loss_function_backward = self.loss_function[1]

    @property
    def prediction_function(self):
        return self._prediction_function

    @prediction_function.setter
    def prediction_function(self, prediction_function):
        if prediction_function == "binary":
            self._prediction_function = self._binary_prediction

    @property
    def accuracy_metric(self):
        return self._accuracy_metric

    @accuracy_metric.setter
    def accuracy_metric(self, accuracy_metric):
        if accuracy_metric == "mad":
            self._accuracy_metric = self._mean_abs_distance

    def unpack_layers(self):
        layers = []
        for layer in self.layers:
            if hasattr(layer, "__iter__"):
                layers.extend(list(layer))
            else:
                layers.append(layer)
        return layers

    def add_layer(self, layer: Layer):
        """
        Add a Layer to NN.

        :param Layer: Layer object
        """
        self.layers.append(layer)

    def pop_layer(self):
        """
        Remove (pop) a Layer from NN.

        """
        self.layers.pop()

    def _split_set(self, X, Y, dev_split):
        """
        Randomly split arrays X and Y of shapes (n, m) into four arrays of shape
        X_, Y_ (n, m * (1 - dev_split))
        X__, Y__ (n, m * dev_split)
        Rounding to int applyies.

        :param X: np.ndarray of training examples
        :param Y: np.ndarray of labels
        :param dev_split: float within [0, 1]
        :return: X_, Y_, X__, Y__ np.ndarrays
        """
        if len(X.shape) > 2:
            return self._split_set_T(X, Y, dev_split)
        dev_indices = np.random.choice(
            X.shape[1], size=int(X.shape[1] * dev_split), replace=False
        )
        mask = np.ones(X.shape[1])
        mask[dev_indices] = 0
        X_train = X[:, mask == 1]
        Y_train = Y[:, mask == 1]
        X_dev = X[:, mask == 0]
        Y_dev = Y[:, mask == 0]
        return X_train, Y_train, X_dev, Y_dev

    def _split_set_T(self, X, Y, dev_split):
        """
        Randomly split arrays X and Y of shapes (n, m) into four arrays of shape
        X_, Y_ (n, m * (1 - dev_split))
        X__, Y__ (n, m * dev_split)
        Rounding to int applyies.

        :param X: np.ndarray of training examples
        :param Y: np.ndarray of labels
        :param dev_split: float within [0, 1]
        :return: X_, Y_, X__, Y__ np.ndarrays
        """
        dev_indices = np.random.choice(
            X.shape[0], size=int(X.shape[0] * dev_split), replace=False
        )
        mask = np.ones(X.shape[0])
        mask[dev_indices] = 0
        X_train = X[mask == 1, :]
        Y_train = Y[:, mask == 1]
        X_dev = X[mask == 0, :]
        Y_dev = Y[:, mask == 0]
        return X_train, Y_train, X_dev, Y_dev

    def _batch_generator(self, batch_size, X, Y, shuffle=False):
        """
        Splits arrays X and Y of shape (n, m) into m / batch_size
        subarrays of shape (n, batch_size)

        :param batch_size: integer, number of examples per batch
        :param X: np.ndarray of training examples
        :param Y: np.ndarray of labels
        :param shuffle: bool, wether X and Y are randomly shuffled before
        partitioning
        :return: X_batch, Y_batch np.ndarray of shape(n, batch_size)
        """
        if len(X.shape) > 2:
            return self._batch_generator_T(batch_size, X, Y, shuffle)

        if shuffle:
            indices = np.arange(X.shape[1])
            np.random.shuffle(indices)
            X = X[:, indices]
            Y = Y[:, indices]
        batch_size = batch_size if batch_size != 0 else X.shape[1]
        X_batch = np.array_split(X, X.shape[1] / batch_size, axis=1)
        Y_batch = np.array_split(Y, Y.shape[1] / batch_size, axis=1)
        return X_batch, Y_batch

    def _batch_generator_T(self, batch_size, X, Y, shuffle=False):
        """
        Splits arrays X and Y of shape (n, m) into m / batch_size
        subarrays of shape (n, batch_size)

        :param batch_size: integer, number of examples per batch
        :param X: np.ndarray of training examples
        :param Y: np.ndarray of labels
        :param shuffle: bool, wether X and Y are randomly shuffled before
        partitioning
        :return: X_batch, Y_batch np.ndarray of shape(n, batch_size)
        """
        if shuffle:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices, :]
            Y = Y[:, indices]
        batch_size = batch_size if batch_size != 0 else X.shape[0]
        X_batch = np.array_split(X, X.shape[0] / batch_size, axis=0)
        Y_batch = np.array_split(Y, Y.shape[1] / batch_size, axis=1)
        return X_batch, Y_batch

    def _logloss_f(self, a, y, eps=1e-15):
        a = np.clip(a, eps, 1 - eps)
        return -y * np.log(a) - (1 - y) * np.log(1 - a)

    def _logloss_b(self, a, y, eps=1e-15):
        a = np.clip(a, eps, 1 - eps)
        return -y / (a) + (1 - y) / (1 - a)

    def _cross_entropy_f(self, a, y, eps=1e-15):
        a = np.clip(a, eps, 1 - eps)
        return -np.log(a[y.astype(bool)]).reshape(1, -1)

    def _cross_entropy_b(self, a, y, eps=1e-15):
        a = np.clip(a, eps, 1 - eps)
        return -y / a

    def _mean_abs_error_f(self, a, y):
        return np.absolute(a - y)

    def _mean_abs_error_b(self, a, y):
        return np.where(a > y, 1, -1)

    def _mean_sqr_error_f(self, a, y):
        return np.square(a - y)

    def _mean_sqr_error_b(self, a, y):
        return 2 * (a - y)

    def _binary_prediction(self, A_out):
        return (A_out > 0.5).astype(int)

    def _mean_abs_distance(self, pred, Y):
        return np.mean(abs(pred - Y))

    def init_parameters(self, shape_in):
        """
        :n_features: number of input features
        """
        for layer in self.unpacked_layers:
            layer.init_parameters(shape_in)
            shape_in = layer.shape_out

    def init_optimizer(self):
        for layer in self.trainable_layers:
            self.optimizer.initialize(layer)

    def _forward_propagation(self, X, update=True, **kwargs):
        """
        Compute forward propagation steps on each layer [l]
        """
        A_input = X
        for layer in self.unpacked_layers:
            A = layer.forward(A_input, update=update, **kwargs)
            A_input = A
        A_out = A
        return A_out

    def _backward_propagation(self, dA_out, **kwargs):
        """
        """
        dA = dA_out
        for layer in reversed(self.unpacked_layers):
            dA_prev = layer.backward(dA, **kwargs)
            dA = dA_prev
        dA_input = dA
        return dA_input

    def predict(self, X):
        a = self._forward_propagation(X, update=False)
        return self.prediction_function(a), a

    def compute_accuracy(self, predictions, Y):
        accuracy = 1 - self.accuracy_metric(predictions, Y)
        return accuracy

    def compute_cost(self, A_out, Y, lambd=0):
        m = A_out.shape[1]
        L2 = (
            lambd / (2 * m) * sum(layer.norm() for layer in self.layers)
            if lambd != 0
            else 0
        )
        L = self._loss_function_forward(A_out, Y) + L2
        J = 1 / m * np.sum(L, axis=1)
        return np.squeeze(J)

    def _update_parameters(self, epoch, learning_rate=0.01, beta1=0.9, beta2=0.999):
        for layer in self.trainable_layers:
            self.optimizer.update_parameters(
                layer, learning_rate, epoch, beta1=beta1, beta2=beta2
            )

    def stats(self):
        for layer in self.trainable_layers:
            print(str(layer))
            for (k, v), (d, g) in zip(layer.parameters.items(), layer.gradients.items()):
                print('  ', k, 'mean:', v.mean(), '± std:', v.std(), 'norm:', np.linalg.norm(v))
                print('  ', f'd{d}', 'mean:', g.mean(), '± std:', g.std(), 'norm:', np.linalg.norm(g))
            print('---')

    def info(self, v=False):
        print(f'# LAYERS: {len(self.unpacked_layers)}')
        layers = self.trainable_layers if not v else self.unpacked_layers
        nparam = sum(param.size for layer in layers for param in layer.parameters.values())
        print('  ', end='')
        print('\n  '.join(str(layer) for layer in layers))
        print('# TOT PARAMETERS:', nparam)
        print(f'LOSS FUNCTION: {self.loss_function}')
        print(f'PREDICTION FUNCTION: {self.prediction_function}')
        print(f'ACCURACY METRICS: {self.accuracy_metric}')



    def train(
        self,
        X,
        Y,
        dev_split=0,
        batch_size=0,
        epochs=10000,
        learning_rate=0.01,
        decay=0,
        lambd=0,
        eps=1e-15,
        rav=0.9,
        beta1=0.9,
        beta2=0.999,
        draw=False,
        ax=None,
        every=1000,
    ):
        """
        """
        X_train, Y_train, X_dev, Y_dev = self._split_set(X, Y, dev_split)
        cycle = 0
        if len(X.shape) > 2:
            batch_size = X_train.shape[0] if batch_size == 0 else batch_size
        else:
            batch_size = X_train.shape[1] if batch_size == 0 else batch_size
        maxiter = epochs * (
            X_train.shape[1] // batch_size + X_train.shape[1] % batch_size
        )
        if draw:
            if not ax:
                fig, ax = plt.subplots()
            self.draw(1, ax=ax, update=False)
        for i in tqdm(range(epochs)):
            lr = learning_rate / (1 + decay * i)
            X_train_batches, Y_train_batches = self._batch_generator(
                batch_size, X_train, Y_train, shuffle=True
            )
            for b, (X_train_batch, Y_train_batch) in enumerate(tqdm(zip(X_train_batches, Y_train_batches), leave=False)):
                A_out = self._forward_propagation(
                    X_train_batch, update=True, rav=rav, eps=eps
                )
                dA_out = self._loss_function_backward(A_out, Y_train_batch)
                dX = self._backward_propagation(dA_out, lambd=lambd)
                self._update_parameters(i, learning_rate=lr, beta1=beta1, beta2=beta2)
                train_cost = self.compute_cost(A_out, Y_train_batch, lambd=lambd)
                train_pred = self.prediction_function(A_out)
                train_accuracy = self.compute_accuracy(train_pred, Y_train_batch)

                if dev_split > 0:
                    dev_pred, dev_A_out = self.predict(X_dev)
                    dev_cost = self.compute_cost(dev_A_out, Y_dev, lambd=lambd)
                    dev_accuracy = self.compute_accuracy(dev_pred, Y_dev)
                    self._training_log(
                        i,
                        cycle,
                        maxiter,
                        train_cost,
                        train_accuracy,
                        dev_cost,
                        dev_accuracy,
                        every=every,
                    )

                else:
                    self._training_log(i, cycle, maxiter, train_cost, train_accuracy,every=every)

                if draw:
                    if cycle == 0 or (cycle + 1) % 10 == 0 or cycle == maxiter - 1:
                        self.draw(1, ax=ax, update=True)
                        plt.pause(1e-10)

                cycle += 1
                # if gen:
                #     yield i, m  # , cost, pred, accuracy

        return 0

    def _training_log(
        self,
        i,
        cycle,
        maxiter,
        train_cost,
        train_accuracy,
        dev_cost=None,
        dev_accuracy=None,
        every=1000,
        ax=None,
    ):
        if cycle == 0 or (cycle + 1) % every == 0 or cycle == maxiter - 1:
            print(f"Epoch {i + 1} (iteration {cycle + 1})")
            print(f"    Train cost: {train_cost}")
            print(f"    Train accuracy: {train_accuracy:.2%}")
            print(f"    Dev cosst:  {dev_cost}")
            print(f"    Dev accuracy:   {dev_accuracy:.2%}")
        if (cycle + 1) % int(maxiter/4) == 0 or cycle == maxiter - 1:
            print('Time for some stats!')
            self.stats()

    def draw(self, n_features, ax=None, left=0, bottom=0, h_spacing=1, v_spacing=1, update=False):
        layers = self.unpacked_layers
        if update:
            for i, layer in enumerate(layers):
                layer.draw(ax, None, None, None, None, None, None, None, update)
            return ax

        plt.rcParams["mathtext.fontset"] = "stix"

        if not ax:
            fig, ax = plt.subplots()

        v_items = [n_features] + [l.shape_out[0] for l in layers]
        max_v_items = max(v_items) if sorted(v_items)[-1] - sorted(v_items)[-2] <= 50 else sorted(v_items)[-2]

        max_height = bottom + v_spacing * (max_v_items - 1)
        r = v_spacing / 3

        x = left
        y = bottom
        x = draw_features(ax, n_features, x, y, max_height, r, bottom, v_spacing, update)

        x_inp = x
        for i, layer in enumerate(layers):
            x = x_inp + h_spacing
            x_inp = layer.draw(ax, x_inp, x, y, max_height, r, v_spacing, h_spacing, update)


        # Y_j = np.linspace(bottom, top, 3)[1:-1][0]

        # J = self.loss_function
        # if self.loss_function == 'cross_entropy':
        #     J = r'$- \sum_{i}^{C} y_i \log(a_i)$'

        # ax.text(X[-1], Y_j, J, ha="center", va="center", fontsize=r * 200)
        ax.set_xlim(left-h_spacing, x_inp + h_spacing)
        ax.set_ylim(bottom - v_spacing, max_height+v_spacing)
        xa, xb = ax.get_xlim()
        ya, yb = ax.get_ylim()
        aspect_ratio = (xb - xa) / (yb - ya)
        ax.figure.set_size_inches(5*aspect_ratio, 5)
        ax.figure.tight_layout()
        ax.axis("off")
        # ax.grid()

        return ax


def draw_features(ax, n, x0, y0, max_height, r, bottom=0, v_spacing=1, update=False):
    if update:
        return
    height = (n - 1) * v_spacing
    x = x0
    y = bottom + y0 + 1/2 * (max_height - height)
    for i in range(n):
        circle = plt.Circle((x, y), r, color="w", ec="w", zorder=3)
        ax.text( x, y, f"$x_{i+1}$", fontsize=8, zorder=4, ha="center", va="center")
        ax.add_artist(circle)
        y += v_spacing
    return x0

def draw_volume(ax, xy, d, h, w, r, deg=45):
    D, H, W = d, h, w
    shift_x = d * np.cos(np.deg2rad(deg))
    shift_y = d * np.sin(np.deg2rad(deg))
    maxdim = max(h+shift_y, w+shift_x)
    scaling = r/maxdim
    d, h, w, shift_x, shift_y = d*scaling, h*scaling, w*scaling, shift_x*scaling, shift_y*scaling
    x, y = xy

    front = plt.Rectangle(xy, w, h, ec='k', fc='w', zorder=3)
    back  = plt.Rectangle((x+shift_x, y+shift_y), w, h, ec='k', fc='w')

    front_edges = np.array(((x, y), (x+w, y), (x+w, y+w), (x, y+w)))
    back_edges = front_edges + np.array([shift_x, shift_y])

    ax.add_artist(back)
    ax.annotate(fr'${D}\times{H}\times{W}$', (x + .5*w, y + .5*h), fontsize=8, ha='center', va='center', zorder=4)
    for f, b in zip(front_edges[1:], back_edges[1:]):
        l = plt.Line2D((f[0], b[0]), (f[1], b[1]), c='k', linewidth=1)
        ax.add_artist(l)
    ax.add_artist(front)

