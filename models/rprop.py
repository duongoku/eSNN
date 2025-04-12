import tensorflow as tf

__name__ = "rprop"

# Original implemetation: https://stackoverflow.com/questions/43768411/implementing-the-rprop-algorithm-in-keras/45849212#45849212

class RProp(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        init_alpha=1e-3,
        scale_up=1.2,
        scale_down=0.5,
        min_alpha=1e-6,
        max_alpha=50.0,
        name="RProp",
        **kwargs
    ):
        super().__init__(name=name, learning_rate=learning_rate, **kwargs)
        self.init_alpha = init_alpha
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
    def build(self, variables):
        super().build(variables)
        self.alphas = []
        self.old_grads = []
        self.prev_weight_deltas = []
        for variable in variables:
            self.alphas.append(
                self.add_variable_from_reference(
                    reference_variable=variable,
                    name="alpha",
                    initializer=tf.keras.initializers.Constant(self.init_alpha),
                )
            )
            self.old_grads.append(
                self.add_variable_from_reference(
                    reference_variable=variable,
                    name="old_grad",
                    initializer=tf.keras.initializers.Zeros(),
                )
            )
            self.prev_weight_deltas.append(
                self.add_variable_from_reference(
                    reference_variable=variable,
                    name="prev_weight_delta",
                    initializer=tf.keras.initializers.Zeros(),
                )
            )
            
    def update_step(self, gradient, variable, learning_rate):
        index = self._get_variable_index(variable)
        alpha = self.alphas[index]
        old_grad = self.old_grads[index]
        prev_weight_delta = self.prev_weight_deltas[index]

        grad_old_grad = gradient * old_grad

        new_alpha = tf.where(
            grad_old_grad > 0,
            tf.minimum(alpha * self.scale_up, self.max_alpha),
            tf.where(
                grad_old_grad < 0,
                tf.maximum(alpha * self.scale_down, self.min_alpha),
                alpha,
            ),
        )

        new_delta = tf.where(
            gradient > 0,
            -new_alpha,
            tf.where(gradient < 0, new_alpha, tf.zeros_like(gradient)),
        )

        weight_delta = tf.where(
            grad_old_grad < 0,
            -prev_weight_delta,
            new_delta,
        )

        variable.assign_add(weight_delta)

        new_old_grad = tf.where(grad_old_grad < 0, tf.zeros_like(gradient), gradient)
        old_grad.assign(new_old_grad)

        alpha.assign(new_alpha)

        prev_weight_delta.assign(weight_delta)

    def get_config(self):
        config = {
            "init_alpha": self.init_alpha,
            "scale_up": self.scale_up,
            "scale_down": self.scale_down,
            "min_alpha": self.min_alpha,
            "max_alpha": self.max_alpha,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class iRprop_(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
        init_alpha=0.01,
        scale_up=1.2,
        scale_down=0.5,
        min_alpha=0.00001,
        max_alpha=50.0,
        name="iRprop_",
        **kwargs
    ):
        super().__init__(name=name, learning_rate=learning_rate, **kwargs)
        self.init_alpha = init_alpha
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
    def build(self, variables):
        super().build(variables)
        self.alphas = []
        self.old_grads = []
        for variable in variables:
            self.alphas.append(
                self.add_variable_from_reference(
                    reference_variable=variable,
                    name="alpha",
                    initializer=tf.keras.initializers.Constant(self.init_alpha),
                )
            )
            self.old_grads.append(
                self.add_variable_from_reference(
                    reference_variable=variable,
                    name="old_grad",
                    initializer=tf.keras.initializers.Zeros(),
                )
            )
            
    def update_step(self, gradient, variable, learning_rate):
        index = self._get_variable_index(variable)
        alpha = self.alphas[index]
        old_grad = self.old_grads[index]

        grad_sign = tf.sign(gradient)

        grad_old_grad = grad_sign * old_grad
        new_alpha = tf.where(
            grad_old_grad > 0,
            tf.minimum(alpha * self.scale_up, self.max_alpha),
            tf.where(
                grad_old_grad < 0,
                tf.maximum(alpha * self.scale_down, self.min_alpha),
                alpha,
            ),
        )

        new_grad_sign = tf.where(grad_old_grad < 0, 0.0, grad_sign)

        weight_delta = - new_grad_sign * new_alpha

        variable.assign_add(weight_delta)

        alpha.assign(new_alpha)

        old_grad.assign(grad_sign)

    def get_config(self):
        config = {
            "init_alpha": self.init_alpha,
            "scale_up": self.scale_up,
            "scale_down": self.scale_down,
            "min_alpha": self.min_alpha,
            "max_alpha": self.max_alpha,
        }
        base_config = super().get_config()
        return {**base_config, **config}
