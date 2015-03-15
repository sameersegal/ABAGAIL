package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopTest {
    /** The n value */
//    private static final int N = 80;
    
//    private static final int T = N/10;

    public static void main(String[] args) {

        if(args.length < 1) {
            args = new String[]{"80"};
        }

        if(args.length < 2) {
            args = new String[]{args[0],"10000"};
        }

        final int N = Integer.parseInt(args[0]);
        final int T = N / 10;

        final int ITERATIONS = Integer.parseInt(args[1]);


        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        long rhcStart = System.nanoTime();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, ITERATIONS);
        fit.train();
        int rhcTime = (int) ((System.nanoTime() - rhcStart)/Math.pow(10,9));
//        System.out.println("RHC: " + ef.value(rhc.getOptimal()) + ",N" + N + ",iters:" + ITERATIONS + ",time:");

        long saStart = System.nanoTime();
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        fit = new FixedIterationTrainer(sa, ITERATIONS);
        fit.train();
        int saTime = (int) ((System.nanoTime() - saStart)/Math.pow(10,9));
//        System.out.println("SA: " + ef.value(sa.getOptimal()));

        long gaStart = System.nanoTime();
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        fit = new FixedIterationTrainer(ga, ITERATIONS);
        fit.train();
        int gaTime = (int) ((System.nanoTime() - gaStart)/Math.pow(10,9));
//        System.out.println("GA: " + ef.value(ga.getOptimal()));

        long mimicStart = System.nanoTime();
        MIMIC mimic = new MIMIC(200, 5, pop);
        fit = new FixedIterationTrainer(mimic, ITERATIONS);
        fit.train();
        int mimicTime = (int) ((System.nanoTime() - mimicStart)/Math.pow(10,9));
//        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()));


        System.out.println(N + "," + ITERATIONS  + "," + ef.value(rhc.getOptimal())+ "," + ef.value(sa.getOptimal()) + "," + ef.value(ga.getOptimal()) + "," + ef.value(mimic.getOptimal()) + "," + "," + rhcTime + "," + saTime + "," + gaTime + "," + mimicTime);
    }
}
