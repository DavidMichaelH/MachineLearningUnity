using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;


public class Layer
{
    public string Name { get; set; }
    public int NumNeurons { get { return Neurons.Count; } }
    public List<Neuron> Neurons { get; private set; }


    public Layer(int numNeurons, int numNeuronInputs, List<List<double>> neuronWeights, List<double> neuronBiases, string name = "")
    {
        // Validate input
        if (numNeurons <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeurons), "Number of neurons must be a positive number.");
        }
        if (numNeuronInputs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeuronInputs), "Number of neuron inputs must be a positive number.");
        }
        if (neuronWeights == null || neuronWeights.Count != numNeurons)
        {
            throw new ArgumentException("Neuron weights must be a non-null list with the same length as the number of neurons.", nameof(neuronWeights));
        }
        if (neuronBiases == null || neuronBiases.Count != numNeurons)
        {
            throw new ArgumentException("Neuron biases must be a non-null list with the same length as the number of neurons.", nameof(neuronBiases));
        }

        // Initialize class variables
        Name = name;
        Neurons = new List<Neuron>();
        for (int i = 0; i < numNeurons; i++)
        {
            Neurons.Add(new Neuron(numNeuronInputs, neuronBiases[i], neuronWeights[i]));
        }
    }

    public Layer(int numNeurons, int numNeuronInputs, string name = "") : this(numNeurons, numNeuronInputs, new List<List<double>>(), new List<double>(), name)
    {
        // Validate input
        if (numNeurons <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeurons), "Number of neurons must be a positive number.");
        }
        if (numNeuronInputs <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numNeuronInputs), "Number of neuron inputs must be a positive number.");
        }

        // Initialize neurons with randomly initialized weights and biases
        Neurons = new List<Neuron>();
        for (int i = 0; i < numNeurons; i++)
        {
            Neurons.Add(new Neuron(numNeuronInputs));
        }
    }


}
