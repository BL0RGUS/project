import pyhelayers
import utils as utils
import os
import h5py
import numpy as np


utils.verify_memory()

print('Misc. initializations')

batch_size = 127

## encode and encrypt the mnist network, also run the optimiser for a batch size of 16
he_run_req = pyhelayers.HeRunRequirements()
he_run_req.set_he_context_options([pyhelayers.DefaultContext()])
he_run_req.optimize_for_batch_size(batch_size)
he_run_req.set_model_encrypted(False)
print(he_run_req)

nn = pyhelayers.NeuralNet()
nn.encode(["model.json", "model.h5"], he_run_req)

# get keys used for network
context = nn.get_created_he_context()
print(context.get_security_level())

# load mnist samples to classify
with h5py.File("x_test.h5") as f:
    x_test = np.array(f["x_test"])
with h5py.File("y_test.h5") as f:
    y_test = np.array(f["y_test"])
    

plain_samples, labels = utils.extract_batch(x_test, y_test, batch_size, 0)

print('Batch of size',batch_size,'loaded')

# encode and encrypt samples using same keys for network
model_io_encoder = pyhelayers.ModelIoEncoder(nn)
samples = pyhelayers.EncryptedData(context)
model_io_encoder.encode_encrypt(samples, [plain_samples])
print('Test data encrypted')

## perform  the inference and time it
utils.start_timer()
predictions = pyhelayers.EncryptedData(context)
nn.predict(predictions, samples)


duration=utils.end_timer('predict')
utils.report_duration('predict per sample',duration/batch_size)

## decode predicitons
plain_predictions = model_io_encoder.decrypt_decode_output(predictions)
print('predictions',plain_predictions)
utils.assess_results(labels, plain_predictions)