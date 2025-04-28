//
// Created by Lorenzo on 24/10/23.
//

#ifndef PAPERRESNET_FHECONTROLLER_H
#define PAPERRESNET_FHECONTROLLER_H

#include "openfhe.h"
#include "ciphertext-ser.h"
#include "scheme/ckksrns/ckksrns-ser.h"
#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include <thread>

#include "Utils.h"

using namespace lbcrypto;
using namespace std;
using namespace std::chrono;

using namespace utils;

using Ptxt = Plaintext;
using Ctxt = Ciphertext<DCRTPoly>;

class FHEController {
    CryptoContext<DCRTPoly> context;

public:
    int circuit_depth;
    int num_slots;

    FHEController() {}

    /*
     * Context generating/loading stuff
     */
    void generate_context(bool serialize = false);
    void generate_context(int log_ring, int log_scale, int log_primes, int digits_hks, int cts_levels, int stc_levels, int relu_deg, bool serialize = false);
    void load_context(bool verbose = true);
    void test_context();

    /*
     * Generating bootstrapping and rotation keys stuff
     */
    void generate_bootstrapping_keys(int bootstrap_slots);
    void generate_rotation_keys(vector<int> rotations, bool serialize = false, string filename = "");
    void generate_bootstrapping_and_rotation_keys(vector<int> rotations,
                                                  int bootstrap_slots,
                                                  bool serialize,
                                                  const string& filename);


    void load_bootstrapping_and_rotation_keys(const string& filename, int bootstrap_slots, bool verbose);
    void load_rotation_keys(const string& filename, bool verbose);
    void clear_bootstrapping_and_rotation_keys(int bootstrap_num_slots);
    void clear_rotation_keys();
    void clear_context(int bootstrapping_key_slots);


    /*
     * CKKS Encoding/Decoding/Encryption/Decryption
     */
    Ptxt encode(const vector<double>& vec, int level, int plaintext_num_slots = 0);
    Ptxt encode(double val, int level, int plaintext_num_slots = 0);
    Ctxt encrypt(const vector<double>& vec, int level = 0, int plaintext_num_slots = 0);
    Ctxt encrypt_ptxt(const Ptxt& p);
    Ptxt decrypt(const Ctxt& c);
    vector<double> decrypt_tovector(const Ctxt& c, int slots);


    /*
     * Homomorphic operations
     */
    Ctxt add(const Ctxt& c1, const Ctxt& c2);
    Ctxt mult(const Ctxt& c, double d);
    Ctxt mult(const Ctxt& c, const Ptxt& p);
    Ctxt bootstrap(const Ctxt& c, bool timing = false);
    Ctxt bootstrap(const Ctxt& c, int precision, bool timing = false);
    Ctxt relu(const Ctxt& c, double scale, bool timing = false);
    Ctxt relu_wide(const Ctxt& c, double a, double b, int degree, double scale, bool timing = false);
    Ctxt square(const Ctxt &c);

    /*
     * I/O
     */
    Ctxt read_input(const string& filename, double scale = 1);
    void print(const Ctxt& c, int slots = 0, int modulus=28, string prefix = "");
    void print_padded(const Ctxt& c, int slots = 0, int padding = 1, string prefix = "");
    void print_min_max(const Ctxt& c);

    /*
     * Convolutional Neural Network functions
     */
    Ctxt convbn_initial(const Ctxt &in, double scale = 0.5, bool timing = false,  int img_width=28, int padding=1, int channels=1, int kernel_size=3);
    Ctxt convbn(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false, int img_width=28, int padding=1, int channels=5, int kernel_size=3);
    vector<Ctxt> convbnsx(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false, int img_width=28, int padding=1, int channels=5, int kernel_size=3);
    vector<Ctxt> convbndx(const Ctxt &in, int layer, int n, double scale = 0.5, bool timing = false, int img_width=28, int padding=1, int channels=5, int kernel_size=3);


    Ctxt downsample(const Ctxt &c1In, const Ctxt &c2In, int img_width_in, int channels_in, int img_width_out, int channels_out);
    Ctxt fully_connected(const Ctxt &c1, int layer, int features_in, int features_out, double scale);
    Ctxt rotsum(const Ctxt &in, int slots);
    Ctxt rotsum_padded(const Ctxt &in, int slots);

    Ctxt repeat(const Ctxt &in, int slots);



    /*
     * Masking things
     */
    Ptxt gen_mask(int n, int img_width, int level);
    Ptxt mask_first_n(int n, int level, int repetitionDelta);
    Ptxt mask_second_n(int n, int level, int repetitionDelta);
    Ptxt mask_first_n_mod(int n, int padding, int pos, int level, int channels_out);
    Ptxt mask_channel(int n, int level, int img_width_in, int img_width_out, int channels_out);
    Ptxt mask_from_to(int from, int to, int level);
    Ptxt mask_from_to_from(int img_size, int ctSize, int level);

    Ptxt mask_mod(int n, int level, double custom_val);

    void bootstrap_precision(const Ctxt& c);

    int relu_degree = 119;
    string parameters_folder = "NO_FOLDER";

private:
    KeyPair<DCRTPoly> key_pair;
    vector<uint32_t> level_budget = {4, 4};


};


#endif //PAPERRESNET_FHECONTROLLER_H
