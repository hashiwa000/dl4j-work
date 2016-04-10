package jp.hashiwa.dl4j.sample;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
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
 * Refer:
 * http://qiita.com/kenmatsu4/items/99d4a54d5a57405ecaf8
 *
 * Created by Hashiwa on 2016/03/22.
 */
public class AutoEncoderCreator {
  public static void main(String[] args) throws Exception {
    Logger log = LoggerFactory.getLogger(AutoEncoderCreator.class);

    final int numRows = 28;
    final int numColumns = 28;
    int numSamples = 200; //60000;
    int batchSize = 50; //100;
    int iterations = 50;
    int seed = 123;
    int listenerFreq = batchSize / 10;

    String binFile = "logs/autoencoder.bin";
    String confFile = "logs/autoencoder.json";

    log.info("Load data....");
    DataSetIterator iter = new MnistDataSetIterator(batchSize, numSamples, true);

    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .list(2)

            // encoding
            .layer(0, new OutputLayer.Builder()
                    .nIn(numRows * numColumns)
                    .nOut(400)
                    .activation("relu")
                    .dropOut(0.3)
                    .lossFunction(LossFunctions.LossFunction.MSE)
                    .build())

            // decoding
            .layer(1, new OutputLayer.Builder()
                    .nIn(400)
                    .nOut(numRows * numColumns)
                    .activation("relu")
                    .dropOut(0.3)
                    .lossFunction(LossFunctions.LossFunction.MSE)
                    .build())
            .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();

//    model.setListeners(new ScoreIterationListener(listenerFreq));
    model.setListeners(new HistogramIterationListener(listenerFreq));

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
