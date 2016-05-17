package jp.hashiwa.dl4j.sample;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Hashiwa on 2016/05/17.
 */
public class MnistImgWriter {
  static final int MNIST_IMAGE_WIDTH = 28;
  static final int MNIST_IMAGE_HEIGHT = 28;
  static final int PIXEL_MAX_VALUE = 0xFF;
  static final int IMAGE_COLUMNS_NUM = 10;
  static final int MARGIN_SIZE = 1;

  private static BufferedImage createJoinedImage(List<INDArray> digits, double imageScale) {
    List<Image> images = digits.stream().map(arr ->
            createImage(arr, imageScale)
    ).collect(Collectors.toList());

    if (images.isEmpty()) {
      return new BufferedImage(0, 0, BufferedImage.TYPE_BYTE_GRAY);
    }

    Image firstImg = images.get(0);
    int imgWidth = firstImg.getWidth(null);
    int imgHeight = firstImg.getHeight(null);

    BufferedImage joinedImg = new BufferedImage(
            (imgWidth + MARGIN_SIZE) * IMAGE_COLUMNS_NUM,
            (imgHeight + MARGIN_SIZE) * (images.size() / IMAGE_COLUMNS_NUM),
            BufferedImage.TYPE_BYTE_GRAY);
    Graphics g = joinedImg.getGraphics();
    for (int cnt=0 ; cnt<images.size() ; cnt++) {
      int i = cnt % IMAGE_COLUMNS_NUM;
      int j = cnt / IMAGE_COLUMNS_NUM;
      g.drawImage(images.get(cnt), i * (imgWidth+MARGIN_SIZE), j * (imgHeight+MARGIN_SIZE), null);
    }

    return joinedImg;
  }

  private static Image createImage(INDArray arr, double imageScale) {
    BufferedImage bi = new BufferedImage(
            MNIST_IMAGE_WIDTH,
            MNIST_IMAGE_HEIGHT,
            BufferedImage.TYPE_BYTE_GRAY);

    for (int i = 0; i < arr.length(); i++) {
      bi.getRaster().setSample(
              i % MNIST_IMAGE_WIDTH,        // x
              i / MNIST_IMAGE_HEIGHT,       // y
              0,                            // b
              (PIXEL_MAX_VALUE - PIXEL_MAX_VALUE * arr.getDouble(i)));
    }

    ImageIcon orig = new ImageIcon(bi);
    Image imageScaled = orig.getImage().getScaledInstance(
            (int) (imageScale * MNIST_IMAGE_WIDTH),
            (int) (imageScale * MNIST_IMAGE_HEIGHT),
            Image.SCALE_REPLICATE);
    return imageScaled;
  }

  /**
   * write image file of INDArray list
   * @param filePath output image file
   * @param digits INDArray list
   * @param imageScale scale
   */
  public static void write(String filePath, List<INDArray> digits, double imageScale) {
    try {
      ImageIO.write(createJoinedImage(digits, imageScale), "png", new File(filePath));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
