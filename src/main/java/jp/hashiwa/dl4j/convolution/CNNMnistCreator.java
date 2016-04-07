package jp.hashiwa.dl4j.convolution;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.*;

/**
 * Reference:
 * Deeplearning4j sample:
 * https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java
 *
 * Caffe LeNet sample:
 * http://caffe.berkeleyvision.org/tutorial/layers.html
 * https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt
 */
public class CNNMnistCreator {

  private static final Logger log = LoggerFactory.getLogger(CNNMnistCreator.class);

  public static void main(String[] args) throws Exception {
    int numRows = 28;
    int numColumns = 28;
    int nChannels = 1;
    int outputNum = 10;
    int numSamples = 2000;
    int batchSize = 500;
    int iterations = 10;
    int splitTrainNum = (int) (batchSize*.8);
    int seed = 123;
    int listenerFreq = iterations/5;
    DataSet mnist;
    SplitTestAndTrain trainTest;
    DataSet trainInput;
    List<INDArray> testInput = new ArrayList<>();
    List<INDArray> testLabels = new ArrayList<>();
    String binFile = "logs/convolution.bin";
    String confFile = "logs/convolution.json";

    log.info("Load data....");
    DataSetIterator mnistIter = new MnistDataSetIterator(batchSize,numSamples, true);

    log.info("Build model....");

    MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .regularization(true).l2(0.0005)
            .learningRate(0.01)
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .list(6)
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                    .nIn(nChannels)
                    .stride(1, 1)
                    .nOut(20)
//                    .dropOut(0.5)
                    .activation("relu")
                    .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2,2)
                    .stride(2,2)
                    .build())
            .layer(2, new ConvolutionLayer.Builder(5, 5)
                    .nIn(nChannels)
                    .stride(1, 1)
                    .nOut(50)
//                    .dropOut(0.5)
                    .activation("relu")
                    .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2,2)
                    .stride(2,2)
                    .build())
            .layer(4, new DenseLayer.Builder().activation("relu")
                    .nOut(500)
                    .build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation("softmax")
                    .build())
            .backprop(true).pretrain(false);
    new ConvolutionLayerSetup(builder, numRows, numColumns, nChannels);

    MultiLayerConfiguration conf = builder.build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    log.info("Train model....");
    model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
    while(mnistIter.hasNext()) {
      mnist = mnistIter.next();
      trainTest = mnist.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
      trainInput = trainTest.getTrain(); // get feature matrix and labels for training
      testInput.add(trainTest.getTest().getFeatureMatrix());
      testLabels.add(trainTest.getTest().getLabels());
      model.fit(trainInput);
    }

    log.info("Evaluate weights....");

    log.info("Evaluate model....");
    Evaluation eval = new Evaluation(outputNum);
    for(int i = 0; i < testInput.size(); i++) {
      INDArray output = model.output(testInput.get(i));
      eval.eval(testLabels.get(i), output);
    }
    log.info(eval.stats());

    log.info("Save model....");
    try (OutputStream fos = new FileOutputStream(binFile);
         DataOutputStream dos = new DataOutputStream(fos)) {
      Nd4j.write(model.params(), dos);
    }
    FileUtils.writeStringToFile(new File(confFile), model.getLayerWiseConfigurations().toJson());

    log.info("****************Example finished********************");
  }
}
