using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
public class Neuron
{
    // the number of inputs the neuron receives
    public int NumInputs { get; }
    
    public List<double> Inputs { get; set; }
    public double Output { get; set; }
    // the error gradient of the neuron, used for backpropagation
    public double ErrorGradient { get; set; }
    // the weights of each input
    public List<double> Weights { get; set; }
    // the bias of the neuron
    public double Bias { get; set; }

    // used when initalizing the values of the neuron. 
    public float weightRange = 1;

    public Func<double, double> ActivationFunction { get; set; }
    public Func<double, double> ActivationFunctionDerivative { get; set; }


    public Neuron(int numInputs, double bias, List<double> weights, ActivationFunctions.ActivationFunction activationFunction)
    {
        // Validate input
        if (numInputs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numInputs), "Number of inputs must be a positive number.");
        }
        /*
        if (weights == null || weights.Count != numInputs)
        {
            throw new ArgumentException("Weights must be a non-null list with the same length as the number of inputs.", nameof(weights));
        }
        */

        // Initialize class variables
        NumInputs = numInputs;
        this.Bias = bias;
        Weights = weights;
        Inputs = new List<double>();
        (ActivationFunction , ActivationFunctionDerivative ) = ActivationFunctions.GetActivationFunction(activationFunction);
    }

    public Neuron(int numInputs, ActivationFunctions.ActivationFunction activationFunction = ActivationFunctions.ActivationFunction.Sigmoid) : this(numInputs, UnityEngine.Random.Range(-1f, 1f), new List<double>(), activationFunction)
    {
        // float weightRange = (float)2.4 / (float)numInputs;
        
        for (int i = 0; i < numInputs; i++)
        {
            Weights.Add(UnityEngine.Random.Range(-weightRange, weightRange));
        }
    }


    double DotProduct(List<double> x, List<double> y)
    {
        double result = 0;
        for (int i = 0; i < x.Count; i++)
        {
            result += x[i]* y[i];
        }
        return result;
    }


    public double ComputePerceptronOutput(List<double> input)
    {
        // Validate input
        if (input == null || input.Count == 0)
        {
            throw new ArgumentException("Input cannot be null or empty.", nameof(input));
        }

        // Make sure input and weights have the same length
        if (input.Count != Weights.Count)
        {
            throw new ArgumentException("Input and weights must have the same length.", nameof(input));
        }

        // Compute and return the output of the perceptron

        Inputs = input;

        double perceptronOutput = 0;
        perceptronOutput += DotProduct(Weights, input);
        perceptronOutput += Bias;
        Output = ActivationFunction(perceptronOutput);
        return Output;
    }


    public void UpdateWeightsAndBiases(double desiredOutput, double currentOutput, double learningRate)
    {
        // Calculate the error of the output
        double error = desiredOutput - currentOutput;

        // Update the weights using the error gradient
        for (int i = 0; i < Weights.Count; i++)
        {
            Weights[i] += error * Inputs[i] * learningRate * ActivationFunctionDerivative(currentOutput);
        }

        // Update the bias using the error gradient
        Bias += error * learningRate * ActivationFunctionDerivative(currentOutput);
    }

    public double Activate(double x)
    {
        return ActivationFunction(x);
    }

    // Gradient of the activation function
    public double ActivateGradient(double x)
    {
        return ActivationFunctionDerivative(x);
    }

}
