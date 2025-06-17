import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class VectorQuantization {
    // Configuration
    private static final int CODEBOOK_SIZE = 256; // 8-bit indices
    private static final int BLOCK_SIZE = 2; // 2x2 pixel blocks
    private static final String[] CATEGORIES = { "nature", "faces", "animals" };
    private static final int TRAINING_PER_CATEGORY = 10;
    private static final int TEST_PER_CATEGORY = 5;
    private static final boolean USE_YUV = true; // Switch between RGB and YUV modes

    // Codebooks
    private List<List<Integer>> codebook1; // R or Y
    private List<List<Integer>> codebook2; // G or U
    private List<List<Integer>> codebook3; // B or V

    private final Random random = new Random();

    public static void main(String[] args) {
        VectorQuantization vq = new VectorQuantization();

        try {
            // Load datasets
            System.out.println("Loading images...");
            List<BufferedImage> trainingImages = loadImages("train");
            List<BufferedImage> testImages = loadImages("test");

            // Generate codebooks
            System.out.println("\nGenerating codebooks using " + (USE_YUV ? "YUV" : "RGB") + " color space...");
            vq.generateCodebooks(trainingImages);

            // Test compression
            System.out.println("\nTesting compression on " + testImages.size() + " images...");
            double totalMse = 0;
            double totalRatio = 0;

            for (int i = 0; i < testImages.size(); i++) {
                BufferedImage original = testImages.get(i);

                // Compress and decompress
                long startTime = System.currentTimeMillis();
                int[][][] compressed = vq.compress(original);
                BufferedImage reconstructed = vq.decompress(compressed, original.getWidth(), original.getHeight());
                long endTime = System.currentTimeMillis();

                // Save result
                saveImage(reconstructed, "results/image" + i + "_" + (USE_YUV ? "yuv" : "rgb") + ".jpg");

                // Calculate metrics
                double mse = calculateMSE(original, reconstructed);
                double ratio = calculateCompressionRatio(original, compressed);

                totalMse += mse;
                totalRatio += ratio;

                System.out.printf("Image %d: MSE=%.2f, Ratio=%.2f:1, Time=%dms\n",
                        i, mse, ratio, (endTime - startTime));
            }

            // Print averages
            System.out.printf("\nAverage results: MSE=%.2f, Compression=%.2f:1\n",
                    totalMse / testImages.size(), totalRatio / testImages.size());

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Generate codebooks from training images
    public void generateCodebooks(List<BufferedImage> trainingImages) throws IOException {
        List<List<Integer>> vectors1 = new ArrayList<>();
        List<List<Integer>> vectors2 = new ArrayList<>();
        List<List<Integer>> vectors3 = new ArrayList<>();

        for (BufferedImage img : trainingImages) {
            BufferedImage processedImg = USE_YUV ? convertRGBtoYUV(img) : img;
            extractVectors(processedImg, vectors1, vectors2, vectors3);
        }

        // Generate codebooks using LBG algorithm
        System.out.println("Generating component 1 codebook...");
        codebook1 = generateCodebook(vectors1);
        System.out.println("Generating component 2 codebook...");
        codebook2 = generateCodebook(vectors2);
        System.out.println("Generating component 3 codebook...");
        codebook3 = generateCodebook(vectors3);
    }

    // Extract vectors for each component from an image
    private void extractVectors(BufferedImage img,
            List<List<Integer>> vectors1,
            List<List<Integer>> vectors2,
            List<List<Integer>> vectors3) {
        int width = img.getWidth();
        int height = img.getHeight();

        if (USE_YUV) {
            // Process Y at full resolution
            for (int y = 0; y < height - BLOCK_SIZE + 1; y += BLOCK_SIZE) {
                for (int x = 0; x < width - BLOCK_SIZE + 1; x += BLOCK_SIZE) {
                    vectors1.add(extractBlock(img, x, y, BLOCK_SIZE, 16));
                }
            }

            // Process U and V with 4:2:0 subsampling (half resolution)
            for (int y = 0; y < height - BLOCK_SIZE * 2 + 1; y += BLOCK_SIZE * 2) {
                for (int x = 0; x < width - BLOCK_SIZE * 2 + 1; x += BLOCK_SIZE * 2) {
                    vectors2.add(extractAverageBlock(img, x, y, BLOCK_SIZE * 2, 8));
                    vectors3.add(extractAverageBlock(img, x, y, BLOCK_SIZE * 2, 0));
                }
            }
        } else {
            // Process RGB at full resolution
            for (int y = 0; y < height - BLOCK_SIZE + 1; y += BLOCK_SIZE) {
                for (int x = 0; x < width - BLOCK_SIZE + 1; x += BLOCK_SIZE) {
                    vectors1.add(extractBlock(img, x, y, BLOCK_SIZE, 16));
                    vectors2.add(extractBlock(img, x, y, BLOCK_SIZE, 8));
                    vectors3.add(extractBlock(img, x, y, BLOCK_SIZE, 0));
                }
            }
        }
    }

    // Extract a block from the image for a specific component
    private List<Integer> extractBlock(BufferedImage img, int x, int y, int size, int shift) {
        List<Integer> block = new ArrayList<>();
        for (int dy = 0; dy < size; dy++) {
            for (int dx = 0; dx < size; dx++) {
                int px = x + dx;
                int py = y + dy;
                if (px < img.getWidth() && py < img.getHeight()) {
                    block.add((img.getRGB(px, py) >> shift) & 0xFF);
                } else {
                    block.add(0); // Padding
                }
            }
        }
        return block;
    }

    // Extract average value for a block (used for U and V subsampling)
    private List<Integer> extractAverageBlock(BufferedImage img, int x, int y, int size, int shift) {
        int sum = 0, count = 0;
        for (int dy = 0; dy < size; dy++) {
            for (int dx = 0; dx < size; dx++) {
                int px = x + dx;
                int py = y + dy;
                if (px < img.getWidth() && py < img.getHeight()) {
                    sum += (img.getRGB(px, py) >> shift) & 0xFF;
                    count++;
                }
            }
        }
        List<Integer> result = new ArrayList<>();
        result.add(count > 0 ? sum / count : 128);
        return result;
    }

    // Apply a block to the image at specified position
    private void applyBlock(BufferedImage img, int x, int y, int size, List<Integer> values, int shift) {
        int index = 0;
        for (int dy = 0; dy < size; dy++) {
            for (int dx = 0; dx < size; dx++) {
                int px = x + dx;
                int py = y + dy;
                if (px < img.getWidth() && py < img.getHeight() && index < values.size()) {
                    int rgb = img.getRGB(px, py);
                    int mask = ~(0xFF << shift);
                    rgb = (rgb & mask) | (values.get(index) << shift);
                    img.setRGB(px, py, rgb);
                }
                index++;
            }
        }
    }

    // Compress an image using the codebooks
    public int[][][] compress(BufferedImage original) {
        BufferedImage img = USE_YUV ? convertRGBtoYUV(original) : original;
        int width = img.getWidth();
        int height = img.getHeight();

        // Calculate dimensions for each component
        int blocks1X = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blocks1Y = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int blocks23X = USE_YUV ? (width + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2) : blocks1X;
        int blocks23Y = USE_YUV ? (height + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2) : blocks1Y;

        // Create compressed representation
        int[][][] compressed = new int[3][][];
        compressed[0] = new int[blocks1Y][blocks1X];
        compressed[1] = new int[blocks23Y][blocks23X];
        compressed[2] = new int[blocks23Y][blocks23X];

        // Compress first component (R or Y)
        for (int y = 0; y < blocks1Y; y++) {
            for (int x = 0; x < blocks1X; x++) {
                List<Integer> block = extractBlock(img, x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, 16);
                compressed[0][y][x] = findNearest(block, codebook1);
            }
        }

        // Compress second and third components (G/B or U/V)
        int blockSize = USE_YUV ? BLOCK_SIZE * 2 : BLOCK_SIZE;
        int shift2 = 8;
        int shift3 = 0;

        for (int y = 0; y < blocks23Y; y++) {
            for (int x = 0; x < blocks23X; x++) {
                List<Integer> block2, block3;

                if (USE_YUV) {
                    block2 = extractAverageBlock(img, x * blockSize, y * blockSize, blockSize, shift2);
                    block3 = extractAverageBlock(img, x * blockSize, y * blockSize, blockSize, shift3);
                } else {
                    block2 = extractBlock(img, x * blockSize, y * blockSize, blockSize, shift2);
                    block3 = extractBlock(img, x * blockSize, y * blockSize, blockSize, shift3);
                }

                compressed[1][y][x] = findNearest(block2, codebook2);
                compressed[2][y][x] = findNearest(block3, codebook3);
            }
        }

        return compressed;
    }

    // Decompress an image from its compressed representation
    public BufferedImage decompress(int[][][] compressed, int width, int height) {
        // Create new image for decompression
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        // Initialize with neutral colors if YUV
        if (USE_YUV) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    img.setRGB(x, y, (128 << 16) | (128 << 8) | 128);
                }
            }
        }

        // Decompress first component (R or Y)
        for (int y = 0; y < compressed[0].length; y++) {
            for (int x = 0; x < compressed[0][0].length; x++) {
                List<Integer> values = codebook1.get(compressed[0][y][x]);
                applyBlock(img, x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, values, 16);
            }
        }

        // Decompress second and third components (G/B or U/V)
        int blockSize = USE_YUV ? BLOCK_SIZE * 2 : BLOCK_SIZE;

        for (int y = 0; y < compressed[1].length; y++) {
            for (int x = 0; x < compressed[1][0].length; x++) {
                if (USE_YUV) {
                    // For YUV, apply subsampled values to larger blocks
                    int u = codebook2.get(compressed[1][y][x]).get(0);
                    int v = codebook3.get(compressed[2][y][x]).get(0);

                    for (int dy = 0; dy < blockSize; dy++) {
                        for (int dx = 0; dx < blockSize; dx++) {
                            int px = x * blockSize + dx;
                            int py = y * blockSize + dy;
                            if (px < width && py < height) {
                                int rgb = img.getRGB(px, py);
                                int y_val = (rgb >> 16) & 0xFF;
                                img.setRGB(px, py, (y_val << 16) | (u << 8) | v);
                            }
                        }
                    }
                } else {
                    // For RGB, apply block by block
                    List<Integer> values2 = codebook2.get(compressed[1][y][x]);
                    List<Integer> values3 = codebook3.get(compressed[2][y][x]);
                    applyBlock(img, x * blockSize, y * blockSize, blockSize, values2, 8);
                    applyBlock(img, x * blockSize, y * blockSize, blockSize, values3, 0);
                }
            }
        }

        // Convert back to RGB if using YUV
        return USE_YUV ? convertYUVtoRGB(img) : img;
    }

    // Find the nearest codeword index for a given vector
    private int findNearest(List<Integer> vector, List<List<Integer>> codebook) {
        int nearest = 0;
        double minDist = Double.MAX_VALUE;

        for (int i = 0; i < codebook.size(); i++) {
            double dist = distance(vector, codebook.get(i));
            if (dist < minDist) {
                minDist = dist;
                nearest = i;
            }
        }

        return nearest;
    }

    // Calculate Euclidean distance between vectors
    private double distance(List<Integer> v1, List<Integer> v2) {
        double sum = 0;
        int size = Math.min(v1.size(), v2.size());

        for (int i = 0; i < size; i++) {
            double diff = v1.get(i) - v2.get(i);
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }

    // LBG algorithm for codebook generation
    private List<List<Integer>> generateCodebook(List<List<Integer>> vectors) {
        if (vectors.isEmpty())
            return new ArrayList<>();

        // Start with centroid of all vectors
        List<List<Integer>> codebook = new ArrayList<>();
        codebook.add(calculateAverage(vectors));

        // Split until desired size
        while (codebook.size() < CODEBOOK_SIZE) {
            codebook = splitAndOptimize(codebook, vectors);
            System.out.print("."); // Progress indicator
        }
        System.out.println(" Done!");

        return codebook;
    }

    // Split each codeword and optimize
    private List<List<Integer>> splitAndOptimize(List<List<Integer>> codebook, List<List<Integer>> vectors) {
        List<List<Integer>> newCodebook = new ArrayList<>();

        // Split each codeword
        for (List<Integer> code : codebook) {
            // Create two slightly perturbed versions
            List<Integer> code1 = new ArrayList<>();
            List<Integer> code2 = new ArrayList<>();

            for (int value : code) {
                code1.add(clamp(value - 1));
                code2.add(clamp(value + 1));
            }

            newCodebook.add(code1);
            newCodebook.add(code2);
        }

        // Optimize codebook through multiple iterations
        for (int iter = 0; iter < 5; iter++) {
            // Assign vectors to nearest codewords
            List<List<List<Integer>>> clusters = new ArrayList<>();
            for (int i = 0; i < newCodebook.size(); i++) {
                clusters.add(new ArrayList<>());
            }

            for (List<Integer> vector : vectors) {
                int nearest = findNearest(vector, newCodebook);
                clusters.get(nearest).add(vector);
            }

            // Update codewords to cluster centroids
            for (int i = 0; i < newCodebook.size(); i++) {
                if (!clusters.get(i).isEmpty()) {
                    newCodebook.set(i, calculateAverage(clusters.get(i)));
                }
            }
        }

        return newCodebook;
    }

    // Calculate average vector
    private List<Integer> calculateAverage(List<List<Integer>> vectors) {
        if (vectors.isEmpty()) {
            return new ArrayList<>();
        }

        // Initialize average vector
        int dimensions = vectors.get(0).size();
        List<Integer> avg = new ArrayList<>();
        for (int i = 0; i < dimensions; i++) {
            avg.add(0);
        }

        // Sum all vectors
        for (List<Integer> vec : vectors) {
            for (int i = 0; i < dimensions && i < vec.size(); i++) {
                avg.set(i, avg.get(i) + vec.get(i));
            }
        }

        // Divide by count
        for (int i = 0; i < dimensions; i++) {
            avg.set(i, avg.get(i) / vectors.size());
        }

        return avg;
    }

    // Color space conversion: RGB to YUV
    private BufferedImage convertRGBtoYUV(BufferedImage rgb) {
        BufferedImage yuv = new BufferedImage(rgb.getWidth(), rgb.getHeight(), BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < rgb.getHeight(); y++) {
            for (int x = 0; x < rgb.getWidth(); x++) {
                int pixel = rgb.getRGB(x, y);
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;

                int yVal = clamp((int) (0.299 * r + 0.587 * g + 0.114 * b));
                int u = clamp((int) (-0.14713 * r - 0.28886 * g + 0.436 * b + 128));
                int v = clamp((int) (0.615 * r - 0.51499 * g - 0.10001 * b + 128));

                yuv.setRGB(x, y, (yVal << 16) | (u << 8) | v);
            }
        }

        return yuv;
    }

    // Color space conversion: YUV to RGB
    private BufferedImage convertYUVtoRGB(BufferedImage yuv) {
        BufferedImage rgb = new BufferedImage(yuv.getWidth(), yuv.getHeight(), BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < yuv.getHeight(); y++) {
            for (int x = 0; x < yuv.getWidth(); x++) {
                int pixel = yuv.getRGB(x, y);
                int yVal = (pixel >> 16) & 0xFF;
                int u = (pixel >> 8) & 0xFF;
                int v = pixel & 0xFF;

                int r = clamp((int) (yVal + 1.13983 * (v - 128)));
                int g = clamp((int) (yVal - 0.39465 * (u - 128) - 0.58060 * (v - 128)));
                int b = clamp((int) (yVal + 2.03211 * (u - 128)));

                rgb.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }

        return rgb;
    }

    // Clamp value to 0-255 range
    private int clamp(int value) {
        return Math.max(0, Math.min(255, value));
    }

    // Calculate MSE between original and reconstructed images
    private static double calculateMSE(BufferedImage original, BufferedImage reconstructed) {
        double sum = 0;
        int width = Math.min(original.getWidth(), reconstructed.getWidth());
        int height = Math.min(original.getHeight(), reconstructed.getHeight());
        int pixels = width * height;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int origRGB = original.getRGB(x, y);
                int recRGB = reconstructed.getRGB(x, y);

                int dr = ((origRGB >> 16) & 0xFF) - ((recRGB >> 16) & 0xFF);
                int dg = ((origRGB >> 8) & 0xFF) - ((recRGB >> 8) & 0xFF);
                int db = (origRGB & 0xFF) - (recRGB & 0xFF);

                sum += dr * dr + dg * dg + db * db;
            }
        }

        return sum / (pixels * 3);
    }

    // Calculate compression ratio
    private static double calculateCompressionRatio(BufferedImage original, int[][][] compressed) {
        // Original: 3 bytes per pixel
        double origSize = original.getWidth() * original.getHeight() * 3;

        // Compressed: 1 byte per block per component
        double compSize = compressed[0].length * compressed[0][0].length +
                compressed[1].length * compressed[1][0].length * 2;

        return origSize / compSize;
    }

    // Load images from directory
    private static List<BufferedImage> loadImages(String type) throws IOException {
        List<BufferedImage> images = new ArrayList<>();
        int perCategory = type.equals("train") ? TRAINING_PER_CATEGORY : TEST_PER_CATEGORY;

        for (String category : CATEGORIES) {
            File dir = new File("images/" + category);
            File[] files = dir.listFiles((d, name) -> {
                String lower = name.toLowerCase();
                return lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png");
            });

            if (files == null) {
                System.out.println("Warning: No images found in " + dir.getPath());
                continue;
            }

            // Shuffle and take required number
            List<File> fileList = new ArrayList<>();
            Collections.addAll(fileList, files);
            Collections.shuffle(fileList);

            int count = 0;
            for (File file : fileList) {
                if (count >= perCategory)
                    break;
                try {
                    images.add(ImageIO.read(file));
                    count++;
                } catch (IOException e) {
                    System.out.println("Error reading " + file.getName() + ": " + e.getMessage());
                }
            }
        }

        return images;
    }

    // Save image to file
    private static void saveImage(BufferedImage img, String path) throws IOException {
        File file = new File(path);
        file.getParentFile().mkdirs();
        ImageIO.write(img, "jpg", file);
    }
}