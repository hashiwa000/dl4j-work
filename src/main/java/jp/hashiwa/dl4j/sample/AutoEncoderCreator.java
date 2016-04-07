package jp.hashiwa.dl4j.sample;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

/**
 * Created by Hashiwa on 2016/03/22.
 */
public class AutoEncoderCreator {
  public static void main(String[] args) throws Exception {
    Logger log = LoggerFactory.getLogger(AutoEncoderCreator.class);

    int numRows = 28;
    int numColumns = 28;
    int seed = 123;
    int numSamples = MnistDataFetcher.NUM_EXAMPLES;
    int batchSize = 1000;
    int iterations = 1;
    int listenerFreq = iterations/5;
    String binFile = "autoencoder.bin";
    String confFile = "autoencoder.json";

    log.info("Load data....");
    DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples, true);

    log.info("Build model....");

//    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//            .seed(seed)
//            .iterations(iterations)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .list(2)
//
//            //encoding stops
//            .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//
//            //decoding starts
//            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(100).nOut(numRows*numColumns).build())
//            .pretrain(true).backprop(true)
//            .build();

//    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
//            .seed(seed)
//            .iterations(iterations)
//            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//            .list(8)
//            .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//            .layer(1, new RBM.Builder().nIn(1000).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//            .layer(2, new RBM.Builder().nIn(250).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//            .layer(3, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//
//            //encoding stops
//            .layer(4, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//
//            //decoding starts
//            .layer(5, new RBM.Builder().nIn(100).nOut(250).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//            .layer(6, new RBM.Builder().nIn(250).nOut(1000).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
//            .layer(7, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(1000).nOut(numRows*numColumns).build())
//            .pretrain(true).backprop(true)
//            .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list(4)
            .layer(0, new RBM.Builder().nIn(numRows * numColumns).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
            .layer(1, new RBM.Builder().nIn(100).nOut(30).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())

            //encoding stops
            .layer(2, new RBM.Builder().nIn(30).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())

            //decoding starts
            .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(100).nOut(numRows*numColumns).build())
            .pretrain(true).backprop(true)
            .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

    model.setListeners(new ScoreIterationListener(listenerFreq));

    log.info("Train model....");
    while (iter.hasNext()) {
      DataSet next = iter.next();
      model.fit(new DataSet(next.getFeatureMatrix(), next.getFeatureMatrix()));
    }

    log.info("Save model....");
    try (OutputStream fos = new FileOutputStream(binFile);
         DataOutputStream dos = new DataOutputStream(fos)) {
      Nd4j.write(model.params(), dos);
    }
    FileUtils.writeStringToFile(new File(confFile), model.getLayerWiseConfigurations().toJson());

    log.info("Finish!");
  }
}
