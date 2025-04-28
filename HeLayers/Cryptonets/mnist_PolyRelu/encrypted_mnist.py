import pyhelayers
import utils as utils
import os
import numpy as np
import torch
import json
from datetime import datetime

utils.verify_memory()

print('Misc. initializations')

num_samples = 127
plain_samples = torch.load('outputs/mltoolbox/plain_samples.pt')
plain_labels = torch.load('outputs/mltoolbox/labels.pt')
plain_samples = plain_samples[:num_samples]
batch_size = len(plain_samples)



checkpoint = torch.load(os.path.join('outputs/mltoolbox/polynomial/Cryptonets_last_checkpoint.pth.tar'), map_location=torch.device('cpu'))
model = checkpoint['model']
plain_model_predictions = model(plain_samples).detach().numpy()
plain_predicted_labels = np.argmax(plain_model_predictions, 1)


model_path = 'outputs/mltoolbox/polynomial/Cryptonets.onnx'
nnp = pyhelayers.NeuralNetPlain()
hyper_params = pyhelayers.PlainModelHyperParams()
nnp.init_from_files(hyper_params,[model_path])


## encode and encrypt the network, also run the optimiser for a batch size of 16
he_run_req = pyhelayers.HeRunRequirements()
he_run_req.set_he_context_options([pyhelayers.SealCkksContext()])
# The encryption is at least as strong as 128-bit encryption.
he_run_req.set_security_level(128)
# Our numbers are theoretically stored with a precision of about 2^-40.
he_run_req.set_fractional_part_precision(30)
# The model weights are kept in the plain
he_run_req.set_model_encrypted(False)


he_run_req.optimize_for_batch_size(batch_size)
profile = pyhelayers.HeModel.compile(nnp, he_run_req)

profile_as_json = profile.to_string()
# Profile supports I/O operations and can be stored on file.
print(json.dumps(json.loads(profile_as_json), indent=4))

context=pyhelayers.HeModel.create_context(profile)
nn = pyhelayers.NeuralNet(context)
nn.encode(nnp, profile)


model_io_encoder = pyhelayers.ModelIoEncoder(nn)
encrypted_samples = pyhelayers.EncryptedData(context)
model_io_encoder.encode_encrypt(encrypted_samples, [plain_samples])
print('Test data encrypted')


encrypted_predictions = pyhelayers.EncryptedData(context)
predict_start = datetime.now()
nn.predict(encrypted_predictions, encrypted_samples)
predict_end = datetime.now()
print('prediction time = %d seconds', predict_end - predict_start)


fhe_model_predictions = model_io_encoder.decrypt_decode_output(encrypted_predictions)
fhe_predicted_labels = np.argmax(fhe_model_predictions, 1)

print('labels predicted by the FHE model: ', fhe_predicted_labels)
print('labels predicted by the plain model: ', plain_predicted_labels)
np.testing.assert_array_equal(fhe_predicted_labels, plain_predicted_labels)

plain_labels = plain_labels.tolist()
fhe_predicted_labels = fhe_predicted_labels.tolist()
correct = 0 
for i in range(len(fhe_predicted_labels)):
    if plain_labels[i] == fhe_predicted_labels[i]:
        correct += 1
    
print(correct/num_samples)

#TODO tonight
# Get results on newly trained mnist w/ deg = 10
# Fix Concrete to not be so shit, get around 50% acc, why is it so slow ?