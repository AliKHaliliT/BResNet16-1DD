import unittest
from BResNet161DD.assets.blocks.bottleneck_residual1d_d import BottleneckResidual1DD
import tensorflow as tf
import inspect


class TestBottleneckResidual1DD(unittest.TestCase):

    def test_creation_init_layer(self):

        # Arrange and Act
        layer = BottleneckResidual1DD(filters=1, strides=(1, ))

        # Assert
        self.assertTrue(isinstance(layer, tf.keras.layers.Layer))


    def test_build_input__shape_layer(self):

        # Arrange
        layer = BottleneckResidual1DD(filters=1, strides=(1, ))
        input_shape = (None, 1, 3)

        # Act
        layer.build(input_shape=input_shape)

        # Assert
        self.assertTrue(layer.built)


    def test_output_tensor_intended__shape(self):

        # Arrange
        filters = 1
        strides = (1, )
        input_tensor = tf.random.normal((1, 1, 3))

        # Act
        output = BottleneckResidual1DD(filters=filters, strides=strides)(input_tensor)

        # Assert
        self.assertEqual(output.shape, (*input_tensor.shape[:-1], filters * 4))


    def test_compute__output__shape_shape_intended__shape(self):

        # Arrange
        layer = BottleneckResidual1DD(filters=1, strides=(1, ))
        input_tensor = tf.random.normal((1, 1, 3))

        # Act
        output = layer(input_tensor)
        output_shape = layer.compute_output_shape(input_tensor.shape)

        # Assert
        self.assertEqual(output.shape, output_shape)


    def test_get__config_init_matching__dict(self):

        # Arrange
        layer = BottleneckResidual1DD(filters=1, strides=(1, ))

        # Act
        init_params = [
            param.name
            for param in inspect.signature(BottleneckResidual1DD.__init__).parameters.values()
            if param.name != "self" and param.name != "kwargs" 
        ]

        # Assert
        self.assertTrue(all(param in layer.get_config() for param in init_params), "Missing parameters in get_config.")


if __name__ == "__main__":
    unittest.main()