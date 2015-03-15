package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.NoSuchElementException;
import java.util.Scanner;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class BankMarketingTest {
    private static final int CSV_TOTAL_NUMBER_OF_ROWS = 4521;
    private static final int CSV_ATTRIBUTES = 17;

    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 16, hiddenLayer = 8, outputLayer = 1, trainingIterations = 10;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {

        if(args.length < 1) {
            args = new String[] {"10"};
        }

        if(args.length < 2) {
            args = new String[] {args[0],"10"};
        }

        if(args.length < 3) {
            args = new String[] {args[0],args[1],"8"};
        }

        int TRAINING_ITERATIONS = Integer.parseInt(args[0]);
        int SPLIT = Integer.parseInt(args[1]);
        int hiddenNodes = Integer.parseInt(args[2]);

        System.out.println("iterations:" + TRAINING_ITERATIONS + ", split:" + SPLIT + ", hiddenNodes:" + hiddenNodes);


        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenNodes, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        //oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i], SPLIT); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = (instances.length * SPLIT/100) + 1; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                //System.out.println(actual + " / " + predicted);

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": " +
                        "\nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n"
                        + "weights: " + optimalInstance.getData().toString();
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int split) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < (instances.length * split / 100); j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                //System.out.println(output + " / " + example.getLabel() + " = " + measure.value(output, example));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[CSV_TOTAL_NUMBER_OF_ROWS][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/bank.csv")));

            for(int i = 0; i < attributes.length; i++) {
                String line = line = br.readLine();
                String[] fields = line.split(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[CSV_ATTRIBUTES]; // 7 attributes
                attributes[i][1] = new double[1];

                // coverting nominal classes into numeric classes
                int j = 0;
                for(; j < CSV_ATTRIBUTES; j++) {
                    String val = fields[j];

                    switch (j) {
                        // job - unemployed , services , management , blue-collar , self-employed , technician , entrepreneur , admin. , student , housemaid , retired , unknown
                        case 1:
                            if(val.equals("unemployed")) { val = "0"; }
                            else if(val.equals("services")) { val = "1"; }
                            else if(val.equals("management")) { val = "2"; }
                            else if(val.equals("blue-collar")) { val = "3"; }
                            else if(val.equals("self-employed")) { val = "4"; }
                            else if(val.equals("technician")) { val = "5"; }
                            else if(val.equals("entrepreneur")) { val = "6"; }
                            else if(val.equals("admin.")) { val = "7"; }
                            else if(val.equals("student")) { val = "8"; }
                            else if(val.equals("housemaid")) { val = "9"; }
                            else if(val.equals("retired")) { val = "10"; }
                            else if(val.equals("unknown")) { val = "11"; }
                            else { val = "12"; }

                            val = "" + (Double.parseDouble(val) / 12);
                            break;
                        // marital - married , single , divorced
                        case 2:
                            if(val.equals("married")) { val = "0"; }
                            else if(val.equals("single")) { val = "1"; }
                            else if(val.equals("divorced")) { val = "2"; }
                            else { val = "3"; }
                            val = "" + (Double.parseDouble(val) / 3);
                            break;
                        // education - primary , secondary , tertiary , unknown
                        case 3:
                            if(val.equals("primary")) { val = "0"; }
                            else if(val.equals("secondary")) { val = "1"; }
                            else if(val.equals("tertiary")) { val = "2"; }
                            else if(val.equals("unknown")) { val = "3"; }
                            else { val = "4"; }
                            val = "" + (Double.parseDouble(val) / 4);
                            break;
                        // default - no, yes
                        case 4:
                            // housing - no, yes
                        case 6:
                            // loan - no, yes
                        case 7:
                            if(val.equals("no")) { val = "0"; }
                            else if(val.equals("yes")) { val = "1"; }
                            else { val = "2"; }
                            val = "" + (Double.parseDouble(val) / 2);
                            break;
                        // contact - cellular , unknown , telephone
                        case 8:
                            if(val.equals("unemployed")) { val = "0"; }
                            else if(val.equals("services")) { val = "1"; }
                            else if(val.equals("management")) { val = "2"; }
                            else if(val.equals("blue-collar")) { val = "3"; }
                            else if(val.equals("self-employed")) { val = "4"; }
                            else if(val.equals("technician")) { val = "5"; }
                            else if(val.equals("entrepreneur")) { val = "6"; }
                            else if(val.equals("admin.")) { val = "7"; }
                            else if(val.equals("student")) { val = "8"; }
                            else if(val.equals("housemaid")) { val = "9"; }
                            else if(val.equals("retired")) { val = "10"; }
                            else if(val.equals("unknown")) { val = "11"; }
                            else { val = "12"; }
                            val = "" + (Double.parseDouble(val) / 12);
                            break;
                        // month - jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec
                        case 10:
                            if(val.equals("jan")) { val = "0"; }
                            else if(val.equals("feb")) { val = "1"; }
                            else if(val.equals("mar")) { val = "2"; }
                            else if(val.equals("apr")) { val = "3"; }
                            else if(val.equals("may")) { val = "4"; }
                            else if(val.equals("jun")) { val = "5"; }
                            else if(val.equals("jul")) { val = "6"; }
                            else if(val.equals("aug")) { val = "7"; }
                            else if(val.equals("sep")) { val = "8"; }
                            else if(val.equals("oct")) { val = "9"; }
                            else if(val.equals("nov")) { val = "10"; }
                            else if(val.equals("dec")) { val = "11"; }
                            else { val = "12"; }
                            val = "" + (Double.parseDouble(val) / 12);
                            break;
                        // poutcome - unknown , failure , other , success
                        case 15:
                            if(val.equals("unknown")) { val = "0"; }
                            else if(val.equals("failure")) { val = "1"; }
                            else if(val.equals("other")) { val = "2"; }
                            else if(val.equals("success")) { val = "3"; }
                            else { val = "4"; }
                            val = "" + (Double.parseDouble(val) / 4);
                            break;
                        // y - no, yes
                        case 16:
//                            System.out.println(val);
                            if(val.equals("no")) { val = "0"; }
                            else if(val.equals("yes")) { val = "1"; }
                            else { val = "2"; }
                            val = "" + (Double.parseDouble(val) / 2);
                            break;

                    }

                    if(j < CSV_ATTRIBUTES - 1 ) {

                        attributes[i][0][j] = Double.parseDouble(val);
                    } else {
                        //System.out.println(val);
                        attributes[i][1][0] = Double.parseDouble(val);
                    }
                }

            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0] < 15 ? 0 : 1));
        }

        return instances;
    }
}
