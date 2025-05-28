#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<std::vector<double>> load_mnist_images(const std::string &filename,
                                                   int &num_images) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  // each file starts with a header = metadata
  // images: magic number, number of images, rows, columns
  // magic number distinguishes file type: image or label
  uint32_t magic, num, rows, cols;
  file.read(reinterpret_cast<char *>(&magic), 4);
  magic = __builtin_bswap32(magic); // big-endian --> little-endian
  file.read(reinterpret_cast<char *>(&num), 4);
  num = __builtin_bswap32(num);
  file.read(reinterpret_cast<char *>(&rows), 4);
  rows = __builtin_bswap32(rows);
  file.read(reinterpret_cast<char *>(&cols), 4);
  cols = __builtin_bswap32(cols);

  num_images = static_cast<int>(num);
  std::vector<std::vector<double>> images(num_images,
                                          std::vector<double>(rows * cols));

  for (int i = 0; i < num_images; ++i) {
    for (unsigned int j = 0; j < rows * cols; ++j) {
      unsigned char pixel = 0;
      file.read(
          reinterpret_cast<char *>(&pixel),
          1); // reinterpret_cast converts pointer type to what read() needs
      images[i][j] = pixel / 255.0; // normalize
    }
  }

  return images;
}

std::vector<std::vector<double>> load_mnist_labels(const std::string &filename,
                                                   int &num_labels) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  uint32_t magic, num;
  file.read(reinterpret_cast<char *>(&magic), 4);
  magic = __builtin_bswap32(magic);
  file.read(reinterpret_cast<char *>(&num), 4);
  num = __builtin_bswap32(num);

  num_labels = static_cast<int>(num);
  std::vector<std::vector<double>> labels(num_labels,
                                          std::vector<double>(10, 0.0));

  // one-hot encoded
  // so [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] = 3

  for (int i = 0; i < num_labels; ++i) {
    unsigned char label = 0;
    file.read(reinterpret_cast<char *>(&label), 1);
    labels[i][label] = 1.0;
  }

  return labels;
}