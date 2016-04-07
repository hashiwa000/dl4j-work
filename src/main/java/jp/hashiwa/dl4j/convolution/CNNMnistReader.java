package jp.hashiwa.dl4j.convolution;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;

/**
 * Created by Hashiwa on 2016/04/03.
 */
public class CNNMnistReader {
  public static void main(String[] args) throws Exception {
    String confFile = "logs/convolution.json";
    String binFile = "logs/convolution.bin";
    int outputNum = 10;

    Logger log = LoggerFactory.getLogger(CNNMnistReader.class);

    log.info("Load stored model ...");
    MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File(confFile)));
    DataInputStream dis = new DataInputStream(new FileInputStream(new File(binFile)));
    INDArray newParams = Nd4j.read(dis);
    dis.close();
    MultiLayerNetwork model = new MultiLayerNetwork(confFromJson);
    model.init();
    model.setParams(newParams);

    System.out.println(model);

    log.info("Evaluate weights....");

    log.info("Evaluate model....");
    MnistDataSetIterator testIter = new MnistDataSetIterator(100, 500);
    Evaluation eval = new Evaluation(outputNum);
    while (testIter.hasNext()) {
      DataSet dataSet = testIter.next();
      INDArray output = model.output(dataSet.getFeatureMatrix());
      eval.eval(dataSet.getLabels(), output);
    }
    log.info(eval.stats());

//    log.info("Test ...");
//    MnistDataSetIterator testIter = new MnistDataSetIterator(1, 10);
//    while (testIter.hasNext()) {
//      DataSet testMnist = testIter.next();
//      INDArray feature = testMnist.getFeatureMatrix();
//      INDArray labels = testMnist.getLabels();
//      INDArray predicted = model.output(feature, Layer.TrainingMode.TEST);
////      System.out.println(feature);
//      System.out.println(labels);
//      System.out.println(predicted);
//      System.out.println("--------------");
//    }
  }

}
