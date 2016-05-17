package jp.hashiwa.dl4j.sample;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Hashiwa on 2016/03/22.
 */
public class AutoEncoderReader {
  public static void main(String[] args) throws Exception {
    String confFile = "logs/autoencoder.json";
    String binFile = "logs/autoencoder.bin";

    Logger log = LoggerFactory.getLogger(AutoEncoderReader.class);

    log.info("Load stored model ...");
    MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File(confFile)));
    DataInputStream dis = new DataInputStream(new FileInputStream(new File(binFile)));
    INDArray newParams = Nd4j.read(dis);
    dis.close();
    MultiLayerNetwork model = new MultiLayerNetwork(confFromJson);
    model.init();
    model.setParams(newParams);

    System.out.println(model);

    log.info("Decode ...");
    List<INDArray> originalArrays = new ArrayList<>();
    List<INDArray> decodedArrays = new ArrayList<>();
    MnistDataSetIterator testIter = new MnistDataSetIterator(100, 500);
    while (testIter.hasNext()) {
      DataSet testMnist = testIter.next();
      INDArray feature = testMnist.getFeatureMatrix();
      INDArray decoded = model.output(feature, Layer.TrainingMode.TEST);
      for (int rowIndex=0 ; rowIndex<feature.rows() ; rowIndex++) {
        originalArrays.add(feature.getRow(rowIndex));
        decodedArrays.add(decoded.getRow(rowIndex));
      }
    }

    log.info("Visualize ...");
    MnistViewer.visualize("Original", originalArrays, 2.0);
    MnistImgWriter.write("logs/original.png", originalArrays, 2.0);
    MnistViewer.visualize("Decoded", decodedArrays, 2.0);
    MnistImgWriter.write("logs/decoded.png", decodedArrays, 2.0);

    try {
      INDArray paramW1 = model.getLayer(0).getParam("W"); // rows=784, columns=1000
      List<INDArray> params = new ArrayList<>();
      for (int columnIndex=0 ; columnIndex<paramW1.columns() ; columnIndex++) {
        params.add(paramW1.getColumn(columnIndex));
      }
      List<INDArray> normalized = normalizedParamW(params);
      MnistViewer.visualize("Params of layer 0" , normalized, 2.0);
      MnistImgWriter.write("logs/param0.png", normalized, 2.0);
    } catch (Exception e) {
      e.printStackTrace();
    }

    try {
      INDArray paramW2 = model.getLayer(1).getParam("W").transposei(); // rows=784, columns=1000
      List<INDArray> params = new ArrayList<>();
      for (int columnIndex=0 ; columnIndex<paramW2.columns() ; columnIndex++) {
        params.add(paramW2.getColumn(columnIndex));
      }
      List<INDArray> normalized = normalizedParamW(params);
      MnistViewer.visualize("Params of layer 1" , normalized, 2.0);
      MnistImgWriter.write("logs/param1.png", normalized, 2.0);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private static List<INDArray> normalizedParamW(List<INDArray> param) {
    double min = param.stream()
            .mapToDouble(n -> n.minNumber().doubleValue())
            .min().orElseThrow(() -> new RuntimeException("No value"));
    double max = param.stream()
            .mapToDouble(n -> n.maxNumber().doubleValue())
            .max().orElseThrow(() -> new RuntimeException("No value"));
//    log.info("max = " + max)
//    log.info("min = " + min)

    double alpha = max - min;
    return param.stream().map(array ->
            array.sub(min).div(alpha)
    ).collect(Collectors.toList());
  }

}
