package jp.hashiwa.dl4j.sample;

import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by Hashiwa on 2016/03/22.
 */
public class MnistViewer {
  static final int MNIST_IMAGE_WIDTH = 28;
  static final int MNIST_IMAGE_HEIGHT = 28;
  static final int PIXEL_MAX_VALUE = 0xFF;

  private static List<JLabel> getComponents(List<INDArray> digits, double imageScale) {
    return digits.stream().map(arr ->
            new JLabel(createImageIcon(arr, imageScale))
    ).collect(Collectors.toList());
  }

  private static ImageIcon createImageIcon(INDArray arr, double imageScale) {
    BufferedImage bi = new BufferedImage(
            MNIST_IMAGE_WIDTH,
            MNIST_IMAGE_HEIGHT,
            BufferedImage.TYPE_BYTE_GRAY);

//    for (int i = 0; i < MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT; i++) {
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
    return new ImageIcon(imageScaled);
  }

  public static void visualize(String title, List<INDArray> digits, double imageScale) {
    JFrame frame = new JFrame();
    frame.setTitle(title);
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    JPanel panel = new JPanel();
    panel.setLayout(new GridLayout(0, 10));

    List<JLabel> labels = getComponents(digits, imageScale);
    labels.stream().forEach(label -> panel.add(label));

    JScrollPane scrollPane = new JScrollPane(panel);
    frame.getContentPane().add(scrollPane, BorderLayout.CENTER);
    frame.setVisible(true);
    frame.pack();
  }
}
