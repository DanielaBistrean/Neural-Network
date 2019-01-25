#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>

#include <ctime>
#include <cstdlib>

#include "loader.h"

const double alpha = 0.1;
const double beta = 0.1;

double sigmoid(double x)
{
  double e;
  double result;

  e = exp(-x);

  result = 1 / (1 + e);

  return result;
}

class Neuron
{
public:
  Neuron() = default;
  Neuron(std::size_t nWeights, bool bRandom = true);

  void   initialize(std::size_t nWeights, bool bRandom = true);
  double activation(const std::vector<double>& vInputs);
  double activation(double input);

  double error1    (double delta, std::size_t index)
  {
    return delta * m_vWeights[index];
  }

  void updateW(double delta, std::size_t index)
  {
    m_vDeltaW[index] = m_vDeltaW[index] * alpha + delta * beta;
    m_vWeights[index] -= m_vDeltaW[index];
  }
private:
  double              m_dBias;
  std::size_t         m_nWeights;
  std::vector<double> m_vWeights;
  std::vector<double> m_vDeltaW;
};

Neuron::Neuron(std::size_t nWeights, bool bRandom)
: m_nWeights{nWeights}
, m_vDeltaW(nWeights, 0.0)
{
  initialize(nWeights, bRandom);
}

void
Neuron::initialize(std::size_t nWeights, bool bRandom)
{
  m_nWeights = nWeights;
  m_vWeights.resize(nWeights);
  m_vDeltaW = std::vector<double>(nWeights, 0.0);

  if (bRandom)
  {
    for (std::size_t i = 0; i < nWeights; ++i)
    {
      m_vWeights[i] = 2 * (((double) rand() / RAND_MAX) - 0.5);
      // std::cout << "w[" << i << "]=" << m_vWeights[i] << "\n";
    }

    // m_dBias = 2 * (((double) rand() / RAND_MAX) - 0.5);    
    m_dBias = 0.0;
  }
}

double
Neuron::activation(const std::vector<double>& vInputs)
{
  if (vInputs.size() < m_nWeights)
    throw std::runtime_error("Wrong size");

  double output = m_dBias;
  for (std::size_t i = 0; i < m_nWeights; ++i)
  {
    // std::cout << "multiply " << m_vWeights[i] << " with " << vInputs[i] << "\n";
    output += m_vWeights[i] * vInputs[i];
  }

  // std::cout << "before sigmoid=" << output << "\n";

  return sigmoid(output);
}

double
Neuron::activation(double input)
{
  return input;
}

int main(int argc, char const *argv[])
{
  const std::size_t nInputSize = 28;

  srand(time(NULL));

  loader ldr{"data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte"};

  std::vector<Neuron> vHiddenLayer1;
  std::vector<Neuron> vHiddenLayer2;
  std::vector<Neuron> vOutputLayer;

  vHiddenLayer1.resize(16);
  vHiddenLayer2.resize(16);
  vOutputLayer.resize(10);

  for (Neuron& n : vHiddenLayer1)
    n.initialize(nInputSize * nInputSize);

  for (Neuron& n : vHiddenLayer2)
    n.initialize(16);

  for (Neuron& n : vOutputLayer)
    n.initialize(16);

  unsigned count = 0;

  std::vector<double> vData;
  unsigned label;
  while (ldr.getNextImage(vData, &label))
  {
    std::vector<double> vHiddenLayer1Activation;
    for (Neuron& n : vHiddenLayer1)
      vHiddenLayer1Activation.push_back(n.activation(vData));

    std::vector<double> vHiddenLayer2Activation;
    for (Neuron& n : vHiddenLayer2)
      vHiddenLayer2Activation.push_back(n.activation(vHiddenLayer1Activation));

    std::vector<double> vOutput;
    for (Neuron& n : vOutputLayer)
      vOutput.push_back(n.activation(vHiddenLayer2Activation));

    std::vector<double> vOutputDelta;
    std::vector<double> vHiddenLayer1Delta;
    std::vector<double> vHiddenLayer2Delta;

    if (count == 59901)
      std::cout << "Network results (expecting=" << label << "):\n";

    for (std::size_t digit = 0; digit < 10; digit++)
    {
      double err;
      
      if (digit == label)
        err = (vOutput[digit] - 1.0);
      else
        err = (vOutput[digit]);

      vOutputDelta.push_back(err * sigmoid(vOutput[digit]) * (1.0 - sigmoid(vOutput[digit])));

      if (count == 59901)
        std::cout << "    " << digit << ": " << vOutput[digit] << "(err=" << err * err << ")\n";
    }

    for (std::size_t i = 0; i < 16; ++i)
    {
      double err = 0.0;
      for (std::size_t j = 0; j < 10; ++j)
        err += vOutputLayer[j].error1(vOutputDelta[j], i);

      vHiddenLayer2Delta.push_back(err * (1.0 + vHiddenLayer2Activation[i]) * (1.0 - vHiddenLayer2Activation[i]));
    }


    for (std::size_t i = 0; i < 16; ++i)
    {
      double err = 0.0;
      for (std::size_t j = 0; j < 16; ++j)
        err += vHiddenLayer2[j].error1(vHiddenLayer2Delta[j], i);

      vHiddenLayer1Delta.push_back(err * (1.0 + vHiddenLayer1Activation[i]) * (1.0 - vHiddenLayer1Activation[i]));
    }

    // std::cout << "update\n";
    // update

    for (std::size_t i = 0; i < 10; ++i)
    {
      for (std::size_t j = 0; j < 16; ++j)
      {
        vOutputLayer[i].updateW(vOutputDelta[i] * vHiddenLayer2Activation[j], j);
      }
    }

    for (std::size_t i = 0; i < 16; ++i)
    {
      for (std::size_t j = 0; j < 16; ++j)
      {
        vHiddenLayer2[i].updateW(vHiddenLayer2Delta[i] * vHiddenLayer1Activation[j], j);
      }
    }

    for (std::size_t i = 0; i < 16; ++i)
    {
      for (std::size_t j = 0; j < (nInputSize * nInputSize); ++j)
      {
        vHiddenLayer1[i].updateW(vHiddenLayer1Delta[i] * vData[j], j);
      }
    }

    if (count == 59901)
      break;

    ++count;
    std::cout << "\rTraining progress: " << count << "/59901 (" << count / 599 << "%)";
  }

  std::cout << "\n";
  return 0;
}
