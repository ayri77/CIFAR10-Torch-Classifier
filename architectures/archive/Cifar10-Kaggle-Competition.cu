// A deep convolutional network trained using 
// SparseConvNet (see http://arxiv.org/abs/1409.6070) taking advantage of
// - spatial-sparsity in the 126x126 input layer,
// - batchwise dropout and a form of Nesterov's accelerated gradient descent,
// - (very) leaky rectified linear units,
// - spatial and color training data augmentation.
// Architecture: input=(3x126x126) -
//    320C2 -                320C2 - MP2 -
//    640C2 - 10% dropout -  640C2 - 10% dropout - MP2 -
//    960C2 - 20% dropout -  960C2 - 20% dropout - MP2 -
//   1280C2 - 30% dropout - 1280C2 - 30% dropout - MP2 -
//   1600C2 - 40% dropout - 1600C2 - 40% dropout - MP2 -
//   1920C2 - 50% dropout - 1920C1 - 50% dropout - 10C1 - Softmax output
// (inspired by
//  Multi-column deep neural networks for image classification; Ciresan, Meier and Schmidhuber;
//  Network In Network; Lin, Chen and Yan;
//  Very Deep Convolutional Networks for Large-Scale Image Recognition; Simonyan and Zisserman)

#include "sparse.h"
#include "OpenCVPicture.h"
#include "readCIFAR10Kaggle.h"

const int OpenCVPicture_maxTranslation=16;
#include "OpenCVPicture_AffineTransform.h"
//Run for 500 to 1000 iterations. Then replace the two lines about with ...
//    Picture* OpenCVPicture::distort(RNG& rng) { OpenCVPicture* pic=new OpenCVPicture(*this); pic->loadData(); return pic;}
// ... and run for 10 more iterations

class KaggleCifar10SparseConvNet : public SpatiallySparseCNN {
public:
  KaggleCifar10SparseConvNet(int nInputFeatures, int nClasses, int cudaDevice) : SpatiallySparseCNN(nInputFeatures, nClasses, cudaDevice) {
    int l=5;
    int k=320;
    float p=0.1f;
    cnn.push_back(new ColorShiftLayer(0.2,0.3,0.6,0.6));
    for (int i=0;i<l;i++) {
      addLeNetLayer((i+1)*k,2,1,1,1,VLEAKYRELU,p*i);
      addLeNetLayer((i+1)*k,2,1,2,2,VLEAKYRELU,p*i);
    }
    int i=l;
    addLeNetLayer((i+1)*k,2,1,1,1,VLEAKYRELU,p*i);
    addLeNetLayer((i+1)*k,1,1,1,1,VLEAKYRELU,p*i);
    addSoftmaxLayer();
  }
};

int main() {
  SpatialDataset trainSet=KaggleCifar10TrainSet();
  SpatialDataset testSet=KaggleCifar10TestSet();
  trainSet.summary();
  testSet.summary();

  int batchSize=50;   //Modify according to available GPU memory.
  int epoch=0;
  int cudaDevice=0;
  string baseName="network_parameters";

  KaggleCifar10SparseConvNet cnn(trainSet.nFeatures,trainSet.nClasses,cudaDevice);
  if (epoch>0)
    cnn.loadWeights(baseName,epoch);
  for (;;) {
    cout <<"epoch: " << ++epoch << " " << flush;
    trainSet.shuffle();
    iterate(cnn, trainSet, batchSize,0.003*batchSize*exp(-0.005*epoch));
    cnn.saveWeights(baseName,epoch);
    if (epoch%100==0) {
      cout << "Producing predictions.labels ...\n";
      iterate(cnn, testSet,  batchSize);
      cout << " ... now run labels2csv\n";
    }
  }
}
