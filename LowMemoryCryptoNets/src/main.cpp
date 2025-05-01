#include <iostream>
#include <sys/stat.h>

#include "FHEController.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GREEN_TEXT "\033[1;32m"
#define RED_TEXT "\033[1;31m"
#define RESET_COLOR "\033[0m"


void check_arguments(int argc, char *argv[]);
vector<double> read_image(const char *filename, int w, int h, int c);

void executeResNet20();

Ctxt initial_layer(const Ctxt& in);
Ctxt layer2(const Ctxt& in);
Ctxt final_layer(const Ctxt& in);

FHEController controller;

int generate_context;
string input_filename;
int verbose;
bool test;
bool plain;

/*
 * TODO:
 * 1) Migliorare convbn sfruttando tutti gli slot del ciphertext
 */

int main(int argc, char *argv[]) {
    //TODO: possibile che il bootstrap a 8192 ci metta lo stesso tempo? indaga
    int num_slots = 16384;
    check_arguments(argc, argv);
    if (test) {
        controller.test_context();
        exit(0);
    }

    if (generate_context == -1) {
        cerr << "You either have to use the argument \"generate_keys\" or \"load_keys\"!\nIf it is your first time, you could try "
                "with \"./LowMemoryFHEResNet20 generate_keys 1\"\nCheck the README.md.\nAborting. :-(" << endl;
        exit(1);
    }


    if (generate_context > 0) {
        switch (generate_context) {
            case 1:
                controller.generate_context(16, 52, 48, 2, 3, 3, 27, true);
                break;
            case 2:
                controller.generate_context(16, 50, 46, 3, 4, 4, 200, true);
                break;
            case 3:
                controller.generate_context(16, 50, 46, 3, 5, 4, 59, true);
                break;
            case 4:
                controller.generate_context(16, 48, 44, 2, 4, 4, 59, true);
                break;
            default:
                controller.generate_context(true);
                break;
        }

        
        int img_width = 32;
        int img_width_out = img_width/2;
        int channels_in = 16;
        int channels_out = 32;
        int img_size = img_width*img_width;
        int img_size_out = img_size/4; 
        controller.generate_bootstrapping_and_rotation_keys({1, -1, img_width, -1*img_width,img_size, -1*img_size, 2, 4, 8, img_width_out*3, -(img_size - img_size_out), (img_size - img_size_out) * channels_out, -num_slots/2, (channels_in*img_size)-num_slots, num_slots-(channels_in*img_size)},
                                                            num_slots,
                                                            true,
                                                            "rotations-layer1.bin");
        //After each serialization I release and re-load the context, otherwise OpenFHE gives a weird error (something
        //like "4kb missing"), but I have no time to investigate :D
    

        controller.clear_context(num_slots);
        controller.load_context(false);

        num_slots = 8192;
        img_width = 16;
        img_width_out = img_width/2;
        channels_in = 32;
        channels_out = 64;
        img_size = img_width*img_width;
        img_size_out = img_size/4;
        controller.generate_bootstrapping_and_rotation_keys({1, -1, img_width, -1*img_width,img_size, -1*img_size, 2, 4, img_width_out*3, -(img_size - img_size_out), (img_size - img_size_out) * channels_out, -num_slots/2, (channels_in*img_size)-num_slots, num_slots-(channels_in*img_size)},
                                                            num_slots,
                                                            true,
                                                            "rotations-layer2.bin");
        //After each serialization I release and re-load the context, otherwise OpenFHE gives a weird error (something
        //like "4kb missing"), but I have no time to investigate :D
    

        controller.clear_context(num_slots);
        controller.load_context(false);

        num_slots = 4096;
        img_width = 8;
        img_width_out = img_width/2;
        channels_in = 64;
        channels_out = 128;
        img_size = img_width*img_width;
        img_size_out = img_size/4;
        controller.generate_bootstrapping_and_rotation_keys({1, -1, img_width, -1*img_width,img_size, -1*img_size, 2, 4, img_width_out*3, -(img_size - img_size_out), (img_size - img_size_out) * channels_out, -num_slots/2, (channels_in*img_size)-num_slots, num_slots-(channels_in*img_size)},
                                                            num_slots,
                                                            true,
                                                            "rotations-layer3.bin");
        //After each serialization I release and re-load the context, otherwise OpenFHE gives a weird error (something
        //like "4kb missing"), but I have no time to investigate :D
    

        controller.clear_context(num_slots);
        controller.load_context(false);
        
        num_slots = 2048;
        controller.generate_bootstrapping_and_rotation_keys({1, num_slots-400, num_slots-10},
                                                            num_slots,
                                                            true,
                                                            "rotations-layer4.bin");
        //After each serialization I release and re-load the context, otherwise OpenFHE gives a weird error (something
        //like "4kb missing"), but I have no time to investigate :D
        controller.clear_context(num_slots);
        controller.load_context(true);
        cout << "Context created correctly." << endl;
        exit(0);

    } else {
        controller.load_context(true);
    }

 // try read, encrypt an mnist image into 4096 batch
        // it probably maybe must be fine cause it normally reads 3x32x32 in which is not power of two
    // next step is to work out how initial layer does it
    int correct = 0;
    cout << correct << endl;
    for(int i = 1; i < 2; i++){
        controller.num_slots = 16384;
        num_slots = 16384; 
        //input_filename = "../mnist_sample/padded_img_" + to_string(i) + ".jpg";
        input_filename = "../CIFAR-10-images/" + to_string(i) + "/0001.jpg";
        //input_filename = "../CIFAR-10-images/0/0003.jpg";
        
        cout << input_filename << endl;
        auto encrpytTime = start_time();
        vector<double> input_image = read_image(input_filename.c_str(), 32, 32, 3);

        Ctxt in = controller.encrypt(input_image, 14);
        print_duration(encrpytTime, "Encrpytime: ");
        // CONVBN1
        //controller.print(in, num59_slots, 32, "in: ");

        auto start = start_time();
        controller.load_bootstrapping_and_rotation_keys("rotations-layer1.bin", num_slots, false);
        Ctxt res = controller.convbn_initial(in,  0.1061, false, 32, 1, 16, 3);
        

        //controller.print(res, num_slots, 32, "before fc: ");

        // RELU 1
        res = controller.relu(res, 0.1061, false);
        
        // BOOTSTRAP level 18
        res = controller.bootstrap(res, false);
        // CONV2BN2 (Downsampling)

        vector<Ctxt> res1sx = controller.convbnsx(res, 2, 1, 0.0825, false, 32, 1, 16, 3);

      
        
        Ctxt fullpackSx = controller.downsample(res1sx[0], res1sx[1], 32, 16, 16, 32);
        res1sx.clear();
        controller.clear_bootstrapping_and_rotation_keys(num_slots);

        controller.num_slots = 8192;
        num_slots = 8192; 
        controller.load_bootstrapping_and_rotation_keys("rotations-layer2.bin", num_slots, false);
        
        fullpackSx = controller.bootstrap(fullpackSx, false);

        res = controller.relu(fullpackSx, 0.0825, false);
        res = controller.bootstrap(res, false);

        res1sx = controller.convbnsx(res, 3, 1, 0.0647, false, 16, 1, 32, 3);
       
        fullpackSx = controller.downsample(res1sx[0], res1sx[1], 16, 32, 8, 64);
        res1sx.clear();
        controller.clear_bootstrapping_and_rotation_keys(num_slots);
        controller.num_slots = 4096;
        num_slots = 4096; 
        controller.load_bootstrapping_and_rotation_keys("rotations-layer3.bin", num_slots, false);
        fullpackSx = controller.bootstrap(fullpackSx, false);

        res = controller.relu(fullpackSx, 0.0647, false);

        res = controller.convbn(res, 4, 1, 0.0827, false, 8, 1, 64, 3);

        res = controller.bootstrap(res, false);

        res = controller.relu(res, 0.0827, false);
        res = controller.bootstrap(res, false);
        res1sx = controller.convbnsx(res, 5, 1, 0.0815, false, 8, 1, 64, 3);
      
        fullpackSx = controller.downsample(res1sx[0], res1sx[1], 8, 64, 4, 128);
        res1sx.clear();
        controller.clear_bootstrapping_and_rotation_keys(num_slots);
        
        controller.num_slots = 2048;
        num_slots = 2048; 
        controller.load_bootstrapping_and_rotation_keys("rotations-layer4.bin", num_slots, false);

        res = controller.bootstrap(fullpackSx, false);

        res = controller.relu(res, 0.0815, false);
        // FC ONE
        print_duration(start, "Inference before fc layers: ");
        res = controller.fully_connected(res, 1, 128*4*4, 400, 0.013);
        res = controller.bootstrap(res, false);

        //controller.print(res, 10, 10, "AFTER relu3");
        res = controller.relu(res, 0.013, false);
        // FC TWO
        res = controller.fully_connected(res, 2, 400, 10, 1.0);
        //res = controller.bootstrap(res, false);
        //res = controller.relu(res, 0.00415, false);
       // res = controller.fully_connected(res, 3, 100, 10, 1.0);
        print_duration(start, "Inference: ");
        controller.clear_bootstrapping_and_rotation_keys(num_slots);

        auto decryptTime = start_time();
        vector<double> clear_result = controller.decrypt_tovector(res, 10);
        print_duration(decryptTime, "DECRYPT: ");
        //Index of the max element
        auto max_element_iterator = std::max_element(clear_result.begin(), clear_result.end());
        int index_max = distance(clear_result.begin(), max_element_iterator);
        if (verbose >= 0) {
            cout << "The prediction is: " << get_class(index_max) << endl;
        }  
        if(index_max == i){
            correct += 1;
        }
        cout << correct << endl;
    }    
    cout << correct << endl;
}

void executeCryptonet() {
    if (verbose >= 0) cout << "Encrypted ResNet20 classification started." << endl;

    Ctxt firstLayer, resLayer1, resLayer2, resLayer3, finalRes;

    bool print_intermediate_values = false;
    bool print_bootstrap_precision = false;

    if (verbose > 1) {
        print_intermediate_values = true;
        print_bootstrap_precision = true;
    }

    if (input_filename.empty()) {
        input_filename = "../inputs/luis.png";
        if (verbose >= 0) cout << "You did not set any input, I use " << GREEN_TEXT << "../inputs/luis.png" << RESET_COLOR << "." << endl;
    } else {
        if (verbose >= 0) cout << "I am going to encrypt and classify " << GREEN_TEXT<< input_filename << RESET_COLOR << "." << endl;
    }

    vector<double> input_image = read_image(input_filename.c_str(), 28, 28, 1);

    Ctxt in = controller.encrypt(input_image, controller.circuit_depth - 4 - get_relu_depth(controller.relu_degree));
    //bookmark
    controller.load_bootstrapping_and_rotation_keys("rotations-layer1.bin", 16384, verbose > 1);

    if (print_bootstrap_precision){
        controller.bootstrap_precision(controller.encrypt(input_image, controller.circuit_depth - 2));
    }

    auto start = start_time();

    //conv1bn1
    firstLayer = initial_layer(in);
    if (print_intermediate_values) controller.print(firstLayer, 16384, 32, "Initial layer: ");
  
    /*
     * Layer 1: 16 channels of 32x32
     */
    auto startLayer = start_time();
    resLayer1 = layer2(firstLayer);
    Serial::SerializeToFile("../checkpoints/layer1.bin", resLayer1, SerType::BINARY);
    if (print_intermediate_values) controller.print(resLayer1, 16384, 32, "Layer 1: ");
    if (verbose > 0) print_duration(startLayer, "Layer 1 took:");


    Serial::DeserializeFromFile("../checkpoints/layer1.bin", resLayer1, SerType::BINARY);
    finalRes = final_layer(resLayer1);
    Serial::SerializeToFile("../checkpoints/finalres.bin", finalRes, SerType::BINARY);

    if (verbose >= 0) print_duration_yellow(start, "The evaluation of the whole circuit took: ");
}

Ctxt initial_layer(const Ctxt& in) {
    double scale = 0.90;

    Ctxt res = controller.convbn_initial(in, scale, verbose > 1, 28, 1, 5, 3);
    res = controller.relu(res, scale, verbose > 1);
    return res;
}

Ctxt final_layer(const Ctxt& in) {
    controller.clear_bootstrapping_and_rotation_keys(4096);
    controller.load_rotation_keys("rotations-finallayer.bin", false);

    controller.num_slots = 4096;

    Ptxt weight = controller.encode(read_fc_weight("../weights/fc.bin", 69, 1.0), in->GetLevel(), controller.num_slots);

    Ctxt res = controller.rotsum(in, 64);
    res = controller.mult(res, controller.mask_mod(64, res->GetLevel(), 1.0 / 64.0));

    //From here, I need 10 repetitons, but I use 16 since *repeat* goes exponentially
    res = controller.repeat(res, 16);
    res = controller.mult(res, weight);
    res = controller.rotsum_padded(res, 64);

    if (verbose >= 0) {
        cout << "Decrypting the output..." << endl;
        controller.print(res, 10, 10, "Output: ");
    }

    vector<double> clear_result = controller.decrypt_tovector(res, 10);

    //Index of the max element
    auto max_element_iterator = std::max_element(clear_result.begin(), clear_result.end());
    int index_max = distance(clear_result.begin(), max_element_iterator);

    if (verbose >= 0) {
        cout << "The input image is classified as " << YELLOW_TEXT << utils::get_class(index_max) << RESET_COLOR << "" << endl;
        cout << "The index of max element is " << YELLOW_TEXT << index_max << RESET_COLOR << "" << endl;
        if (plain) {
            string command = "python3 ../src/plain/script.py \"" + input_filename + "\"";
            int return_sys = system(command.c_str());
            if (return_sys == 1) {
                cout << "There was an error launching src/plain/script.py. Run it from Python in order to debug it." << endl;
            }
        }
    }


    return res;
}


Ctxt layer2(const Ctxt& in) {

    double scaleSx = 0.57;
    double scaleDx = 0.40;


    bool timing = verbose > 1;

    if (verbose > 1) cout << "---Start: Layer2 - Block 1---" << endl;
    auto start = start_time();
    Ctxt boot_in = controller.bootstrap(in, timing);

    vector<Ctxt> res1sx = controller.convbnsx(boot_in, 4, 1, scaleSx, timing); //Questo è lento

    vector<Ctxt> res1dx = controller.convbndx(boot_in, 4, 1, scaleDx, timing); //Questo è lento


    controller.clear_bootstrapping_and_rotation_keys(16384);
    controller.load_rotation_keys("rotations-layer2-downsample.bin", timing);

    Ctxt fullpackSx = controller.downsample(res1sx[0], res1sx[1], 28, 5, 14, 10);
    Ctxt fullpackDx = controller.downsample(res1dx[0], res1dx[1], 28, 5, 14, 10);


    res1sx.clear();
    res1dx.clear();

    controller.clear_rotation_keys();
    controller.load_bootstrapping_and_rotation_keys("rotations-layer2.bin", 8192, verbose > 1);

    controller.num_slots = 8192;
    fullpackSx = controller.bootstrap(fullpackSx, timing);

    fullpackSx = controller.relu(fullpackSx, scaleSx, timing);

    //I use the scale of the right branch since they will be added together
    fullpackSx = controller.convbn(fullpackSx, 4, 2, scaleDx, timing);
    Ctxt res1 = controller.add(fullpackSx, fullpackDx);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scaleDx, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer2 - Block 1---" << endl;

    double scale = 0.76;

    if (verbose > 1) cout << "---Start: Layer2 - Block 2---" << endl;
    start = start_time();
    Ctxt res2;
    res2 = controller.convbn(res1, 5, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.37;

    res2 = controller.convbn(res2, 5, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer2 - Block 2---" << endl;

    scale = 0.63;

    if (verbose > 1) cout << "---Start: Layer2 - Block 3---" << endl;
    start = start_time();
    Ctxt res3;
    res3 = controller.convbn(res2, 6, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);
  
    scale = 0.25;

    res3 = controller.convbn(res3, 6, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);
    if (verbose > 1) print_duration(start, "Total");
    if (verbose > 1) cout << "---End  : Layer2 - Block 3---" << endl;

    return res3;
}

void check_arguments(int argc, char *argv[]) {
    generate_context = -1;
    verbose = 0;

    for (int i = 1; i < argc; ++i) {
        //I first check the "verbose" command
        if (string(argv[i]) == "verbose") {
            if (i + 1 < argc) { // Verifica se c'è un argomento successivo a "input"
                verbose = atoi(argv[i + 1]);
            }
        }
    }


    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "load_keys") {
            if (i + 1 < argc) {
                controller.parameters_folder = "keys_exp" + string(argv[i + 1]);
                if (verbose > 1) cout << "Context folder set to: \"" << controller.parameters_folder << "\"." << endl;
                generate_context = 0;
            }
        }

        if (string(argv[i]) == "test") {
            test = true;
        }

        if (string(argv[i]) == "generate_keys") {
            if (i + 1 < argc) {
                string folder = "";
                if (string(argv[i+1]) == "1") {
                    folder = "keys_exp1";
                    generate_context = 1;
                } else if (string(argv[i+1]) == "2") {
                    folder = "keys_exp2";
                    generate_context = 2;
                } else if (string(argv[i+1]) == "3") {
                    folder = "keys_exp3";
                    generate_context = 3;
                } else if (string(argv[i+1]) == "4") {
                    folder = "keys_exp4";
                    generate_context = 4;
                } else {
                    cerr << "Set a proper value for 'generate_keys'. For instance, use '1'. Check the README.md" << endl;
                    exit(1);
                }

                struct stat sb;
                if (stat(("../" + folder).c_str(), &sb) == 0) {
                    cerr << "The keys folder \"" << folder << "\" already exists, I will abort.";
                    exit(1);
                }
                else {
                    mkdir(("../" + folder).c_str(), 0777);
                }

                controller.parameters_folder = folder;
                if (verbose > 1) cout << "Context folder set to: \"" << controller.parameters_folder << "\"." << endl;
            }
        }
        if (string(argv[i]) == "input") {
            if (i + 1 < argc) {
                input_filename = "../" + string(argv[i + 1]);
                if (verbose > 1) cout << "Input image set to: \"" << input_filename << "\"." << endl;
            }
        }

        if (string(argv[i]) == "plain") {
            plain = true;
        }

    }

}

vector<double> read_image(const char *filename, int w, int h, int c) {
    int width = w;
    int height = h;
    int channels = c;
    unsigned char* image_data = stbi_load(filename, &width, &height, &channels, 0);

    if (!image_data) {
        cerr << "Could not load the image in " << filename << endl;
        return vector<double>();
    }

    vector<double> imageVector;
    imageVector.reserve(width * height * channels);
    for (int j = 0; j < channels; j++){
        for (int i = 0; i < width * height; ++i) {
            imageVector.push_back(static_cast<double>(image_data[j + (channels * i)])/255.0f);
        }
    }

    stbi_image_free(image_data);

    return imageVector;
}
