#ifndef LOADER_H
#define LOADER_H 1

#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

class loader
{
public:
  loader(const std::string& imagesFile, const std::string& labelsFile);

  bool getNextImage(std::vector<double>& out, unsigned* label = nullptr);
private:
  std::ifstream m_imagesFile;
  std::ifstream m_labelsFile;
  std::size_t   m_nItems;
  std::size_t   m_nRows;
  std::size_t   m_nCols;
  std::size_t   m_iNextImage;
};

loader::loader(const std::string& imagesFile, const std::string& labelsFile)
: m_imagesFile{imagesFile}
, m_labelsFile{labelsFile}
, m_iNextImage{0}
{
  int32_t magic1;
  int32_t magic2;

  int32_t nItems1;
  int32_t nItems2;

  m_imagesFile.read((char *) &magic1, sizeof(magic1));
  if (m_imagesFile.gcount() != sizeof(magic1))
    throw std::runtime_error("Invalid file read (magic) for images");

  m_labelsFile.read((char *) &magic2, sizeof(magic2));
  if (m_labelsFile.gcount() != sizeof(magic2))
    throw std::runtime_error("Invalid file read (magic) for labels (gcount=" + std::to_string(m_labelsFile.gcount()) + ")");

  if (__builtin_bswap32(magic1) != 2051)
    throw std::runtime_error("Invalid images file format");

  if (__builtin_bswap32(magic2) != 2049)
    throw std::runtime_error("Invalid labels file format");

  m_imagesFile.read((char *) &nItems1, sizeof(nItems1));
  if (m_imagesFile.gcount() != sizeof(nItems1))
    throw std::runtime_error("Invalid file read (number of items) for images");

  m_labelsFile.read((char *) &nItems2, sizeof(nItems2));
  if (m_labelsFile.gcount() != sizeof(nItems2))
    throw std::runtime_error("Invalid file read (number of items) for labels");

  if (nItems1 != nItems2)
    throw std::runtime_error("Number of items mismatch!");

  m_nItems = __builtin_bswap32(nItems1);

  m_imagesFile.read((char *) &m_nRows, sizeof(int32_t));
  if (m_imagesFile.gcount() != sizeof(int32_t))
    throw std::runtime_error("Invalid file read (rows) for images");

  m_imagesFile.read((char *) &m_nCols, sizeof(int32_t));
  if (m_imagesFile.gcount() != sizeof(int32_t))
    throw std::runtime_error("Invalid file read (colums) for images");

  m_nCols = __builtin_bswap32(m_nCols);
  m_nRows = __builtin_bswap32(m_nRows);

  if (m_nCols <= 0 || m_nRows <= 0)
    throw std::runtime_error("Invalid image data size");
}

bool
loader::getNextImage(std::vector<double>& out, unsigned* label)
{
  if (m_iNextImage >= m_nItems)
    return false;

  std::vector<uint8_t> data(m_nCols * m_nRows);
  m_imagesFile.read((char *) &data[0], sizeof(uint8_t) * m_nRows * m_nCols);
  if (m_imagesFile.gcount() != (unsigned) (sizeof(uint8_t) * m_nRows * m_nCols))
    throw std::runtime_error("Invalid file read (data) for images");

  std::vector<double> result;
  for (std::size_t i = 0; i < m_nRows; ++i)
  {
    for (std::size_t j = 0; j < m_nCols; ++j)
    {
      result.push_back((double) data[i * m_nCols + j] / 255.0);
    }
  }


  if (label)
  {
    uint8_t data;
    m_labelsFile.read((char *) &data, sizeof(uint8_t));
    if (m_labelsFile.gcount() != sizeof(uint8_t))
      throw std::runtime_error("Invalid file read (data) for labels");

    *label = data;
  }

  out.swap(result);
  return true;
}

#endif
