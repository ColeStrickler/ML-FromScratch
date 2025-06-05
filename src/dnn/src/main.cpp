#include <H5Cpp.h>
#include <iostream>
#include <vector>
#include <string>
#include <neuralnetwork.h>
#include <memory>
#include <cstdint>  // for fixed size types
#include <fstream>
using namespace H5;
using namespace std;

// Helper to read big-endian 32-bit int from file
uint32_t readBigEndianInt(std::ifstream& file) {
    uint8_t bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) | (uint32_t(bytes[2]) << 8) | uint32_t(bytes[3]);
}

// Load one MNIST image, flatten it, normalize pixels to [0,1]
std::vector<float> loadMNISTImage(const std::string& filename, int image_index) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file " << filename << std::endl;
        return {};
    }

    uint32_t magic = readBigEndianInt(file);
    if (magic != 2051) {
        std::cerr << "Invalid magic number in MNIST image file: " << magic << std::endl;
        return {};
    }

    uint32_t num_images = readBigEndianInt(file);
    uint32_t num_rows = readBigEndianInt(file);
    uint32_t num_cols = readBigEndianInt(file);

    if (image_index >= num_images) {
        std::cerr << "Requested image index exceeds number of images in file." << std::endl;
        return {};
    }

    // Skip to the requested image (each image is 28*28 = 784 bytes)
    size_t image_size = num_rows * num_cols;
    file.seekg(image_index * image_size, std::ios::cur);

    std::vector<uint8_t> buffer(image_size);
    file.read(reinterpret_cast<char*>(buffer.data()), image_size);

    // Flatten and normalize to [0,1]
    std::vector<float> image_flattened(image_size);
    for (size_t i = 0; i < image_size; ++i) {
        image_flattened[i] = buffer[i] / 255.0f;
    }

    return image_flattened;
}


// accept other layers in the future
void readDataset(H5File& file, const std::string& full_path, DenseLayer& layer) {
    try {
        DataSet dataset = file.openDataSet(full_path);
        DataSpace dataspace = dataset.getSpace();

        int ndims = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(ndims);
        dataspace.getSimpleExtentDims(dims.data());

        cout << "  Dataset: " << full_path << " | Shape: (";
        for (int i = 0; i < ndims; ++i) {
            cout << dims[i];
            if (i < ndims - 1) cout << ", ";
        }
        cout << ")" << endl;

        size_t total_size = 1;
        for (auto d : dims) total_size *= d;
        std::vector<float> data(total_size);
        dataset.read(data.data(), PredType::NATIVE_FLOAT);

        cout << "    First weight: " << data[0] << endl;

    } catch (Exception& e) {
        cerr << "  Failed to read dataset: " << full_path << endl;
    }
}


std::unique_ptr<NetworkLayer> readDenseLayer(H5File& file, const std::string& full_path)
{
    try {
        
        DataSet weight_dataset = file.openDataSet(full_path + "/0");
        DataSpace weight_dataspace = weight_dataset.getSpace();
        
        DataSet bias_dataset = file.openDataSet(full_path + "/1");
        DataSpace bias_dataspace = bias_dataset.getSpace();


        int ndims = weight_dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(ndims);
        weight_dataspace.getSimpleExtentDims(dims.data());


        int bias_ndims = bias_dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> bias_dims(bias_ndims);
        bias_dataspace.getSimpleExtentDims(bias_dims.data());

        int input_shape = dims[0];
        int num_neurons = dims[1];
        int bias_shape = bias_dims[0];
        assert(num_neurons == bias_shape);
        
        std::unique_ptr<DenseLayer> dense = std::make_unique<DenseLayer>(num_neurons, input_shape);
        
        
        

        cout << "  Dataset: " << full_path << " | Shape: (";
        for (int i = 0; i < ndims; ++i) {
            cout << dims[i];
            if (i < ndims - 1) cout << ", ";
        }
        cout << ")" << endl;

        size_t total_size = 1;
        for (auto d : dims) total_size *= d;
        std::vector<float> data(total_size);
        weight_dataset.read(data.data(), PredType::NATIVE_FLOAT);
        
        

        size_t bias_total_size = 1;
        for (auto d : bias_dims) bias_total_size *= d;
        std::vector<float> bias_data(bias_total_size);
        bias_dataset.read(bias_data.data(), PredType::NATIVE_FLOAT);
        std::cout << "here\n";
        dense->SetWeights(data.data(), total_size);
        dense->SetBiases(bias_data.data(), bias_total_size);

        

        cout << "    First weight: " << data[0] << endl;
        return dense;

    } catch (Exception& e) {
        cerr << "  Failed to read dataset: " << full_path << endl;
        exit(-1);
    }
}


std::vector<std::unique_ptr<NetworkLayer>> processDenseLayers(H5File& file) {
    Group layersGroup = file.openGroup("/layers");
    hsize_t numObjs = layersGroup.getNumObjs();
    std::vector<std::unique_ptr<NetworkLayer>> layers;




    for (hsize_t i = 0; i < numObjs; ++i) {
        string name = layersGroup.getObjnameByIdx(i);
        H5G_obj_t type = layersGroup.getObjTypeByIdx(i);
        

        if (type == H5G_GROUP && name.find("dense") == 0) {  // Name starts with "dense"
            string varsPath = "/layers/" + name + "/vars";
            try {
                Group varsGroup = file.openGroup(varsPath);

                // Read datasets "0" and "1" inside vars
                auto layer = readDenseLayer(file, varsPath);
                layers.push_back(std::move(layer));
            } catch (Exception& e) {
                cerr << "Failed to open vars group for " << name << endl;
            }
        }
    }
    return layers;
}


uint8_t readLabel(const std::string& filename, int index) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open label file: " << filename << "\n";
        return 255; // invalid label
    }

    uint32_t magic = readBigEndianInt(file);
    uint32_t num_labels = readBigEndianInt(file);

    if (index < 0 || index >= static_cast<int>(num_labels)) {
        std::cerr << "Label index out of range\n";
        return 255;
    }

    file.seekg(index, std::ios::cur); // skip to the label at 'index'
    uint8_t label;
    file.read(reinterpret_cast<char*>(&label), 1);

    return label;
}

void printMNISTImage(const std::vector<float>& image) {
    // Assuming image is 28x28 flattened
    const int width = 28;
    const int height = 28;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float pixel = image[y * width + x];
            if (pixel > 0.75) std::cout << "@";
            else if (pixel > 0.5) std::cout << "#";
            else if (pixel > 0.25) std::cout << "*";
            else if (pixel > 0.1) std::cout << ".";
            else std::cout << " ";
        }
        std::cout << "\n";
    }
}

int getGuess(const LayerActivation& output_layer, int ncategories)
{
    int guess = 0;
    float highest = 0.00f;
    for (int i = 0; i < ncategories; i++)
    {
        auto activation = output_layer.activations[i];
        if (activation > highest)
        {
            highest = activation;
            guess = i;
        }
    }
    return guess;
}

int get_mnist_image_count(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return -1;
    }

    file.seekg(4);  // skip magic number

    int32_t num_images = 0;
    file.read(reinterpret_cast<char*>(&num_images), 4);

    // MNIST uses big-endian format; convert if on little-endian system
    num_images = __builtin_bswap32(num_images);

    return num_images;
}


std::vector<std::pair<int, std::vector<float>>> MakeDataSet(std::string filename, std::string labels)
{
    std::vector<std::pair<int, std::vector<float>>> ret;
    auto count = get_mnist_image_count(filename);

    for (int i = 0; i < count; i++)
    {
        std::vector<float> image = loadMNISTImage(filename, i);
        int label = readLabel(labels, i);
        ret.push_back(std::make_pair(label, image));
    }

    return ret;
}



int main() {

    std::string train = "train-images-idx3-ubyte/train-images-idx3-ubyte";
    std::string train_labels = "train-labels-idx1-ubyte/train-labels-idx1-ubyte";
    std::string test = "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    std::string test_labels = "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";

   auto train_set = MakeDataSet(train, train_labels);
   auto test_set = MakeDataSet(test, test_labels);

    assert(train_set.size());
    assert(test_set.size());
    


    const std::string weights = "./mnist_dnn.weights.h5";
    int correct = 0;

    int total = 1000;
    try {
        H5File file(weights, H5F_ACC_RDONLY);
        
        //auto layers = processDenseLayers(file);


        std::vector<std::unique_ptr<NetworkLayer>> layers;
        layers.push_back(std::make_unique<DenseLayer>(256, 784));
        layers.push_back(std::make_unique<DenseLayer>(128, 256));
        layers.push_back(std::make_unique<DenseLayer>(10, 128));

        NeuralNetwork network(layers);
        

        network.SGD(train_set, 25, 64, 0.1f, test_set);
        
        
        //for (int i = 0; i < total; i++)
        //{
        //    std::vector<float> image = loadMNISTImage(filename, i);
        //    int label = readLabel(labels, i);
        //    if (image.empty())
        //    {
        //        std::cout << "image load fail\n";
        //        return -1;
        //    }
        //   int guess = network.Inference(image);
        //    if (guess == label)
        //        correct++;
        //}
        //printf("Score %d/%d\n", correct, total);

    } catch (FileIException& e) {
        e.printErrorStack();
    } catch (GroupIException& e) {
        e.printErrorStack();
    } catch (DataSetIException& e) {
        e.printErrorStack();
    }

    return 0;
}
