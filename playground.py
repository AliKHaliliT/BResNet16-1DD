import os
# Reduce TensorFlow log level for minimal logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'


from BResNet161DD import BResNet161DD


model = BResNet161DD()
model.build((None, 128, 32))
model.summary()