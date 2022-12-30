using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class ANN
{
	 
	public int numInputs;
	public int numOutputs;
	public int numHidden;
	public int NumLayers { get { return numHidden + 1; } }
	public int numNPerHidden;
	public double alpha;
	List<Layer> layers = new List<Layer>();
 

	public ANN(int nI, int nO, int nH, int nPH, double a )
	{
		numInputs = nI;
		numOutputs = nO;
		numHidden = nH;
		numNPerHidden = nPH;
		alpha = a;

		if (numHidden > 0)
		{
			layers.Add(new Layer(numNPerHidden, numInputs));

			for (int i = 0; i < numHidden - 1; i++)
			{
				layers.Add(new Layer(numNPerHidden, numNPerHidden));
			}

			layers.Add(new Layer(numOutputs, numNPerHidden));
		}
		else
		{
			layers.Add(new Layer(numOutputs, numInputs));
		}
	}

	  

	public string PrintWeights()
	{
		string weightStr = "";
		foreach (Layer l in layers)
		{
			foreach (Neuron n in l.Neurons)
			{
				foreach (double w in n.Weights)
				{
					weightStr += w + ",";
				}
			}
		}
		return weightStr;
	}

	public void LoadWeights(string weightStr)
	{
		if (weightStr == "") return;
		string[] weightValues = weightStr.Split(',');
		int w = 0;
		foreach (Layer l in layers)
		{
			foreach (Neuron n in l.Neurons)
			{
				for (int i = 0; i < n.Weights.Count; i++)
				{
					n.Weights[i] = System.Convert.ToDouble(weightValues[w]);
					w++;
				}
			}
		}
	}
	 
	public List<double> Evaluate(List<double> input)
	{
		// Validate input
		if (input == null || input.Count != numInputs)
		{
			throw new ArgumentException("Input must be a non-null list with the same length as the number of inputs.", nameof(input));
		}

		// Propagate the input through the neural network
		List<double> currentInput = input;
		List<double> currentOutput = new List<double>();
		for (int i = 0; i < numHidden+1; i++)
		{
			
			currentOutput.Clear();
			 
			for (int j = 0; j < layers[i].NumNeurons; j++)
			{
				 
				currentOutput.Add(layers[i].Neurons[j].ComputePerceptronOutput(currentInput));
			}
			 
			currentInput = new List<double>(currentOutput);
		}

		// Return the output of the neural network
		return currentInput;
	}


	public void Backpropagate(List<double> desiredOutput)
	{

		// Iterate over the layers of the neural network in reverse order
		for (int i = NumLayers - 1; i >= 0; i--)
		{
			// Calculate the error gradients for the output layer
			if (i == NumLayers-1)
			{
				for (int j = 0; j < layers[i].NumNeurons; j++)
				{
					layers[i].Neurons[j].ErrorGradient =
						(desiredOutput[j] - layers[i].Neurons[j].Output) *
						layers[i].Neurons[j].ActivationFunctionDerivative(layers[i].Neurons[j].Output);
				}
			}
			// Calculate the error gradients for the hidden layers
			else
			{
				for (int j = 0; j < layers[i].NumNeurons; j++)
				{
					double errorGradientSum = 0;
					for (int k = 0; k < layers[i + 1].NumNeurons; k++)
					{
						errorGradientSum += layers[i + 1].Neurons[k].ErrorGradient * layers[i + 1].Neurons[k].Weights[j];
					}
					layers[i].Neurons[j].ErrorGradient =
						layers[i].Neurons[j].ActivationFunctionDerivative(layers[i].Neurons[j].Output) * errorGradientSum;
				}
			}
		}
	}

	public void UpdateWeights()
	{
		// Iterate over the layers of the neural network in reverse order
		for (int i = NumLayers - 1; i >= 0; i--)
		{
			// Iterate over the neurons in the current layer
			for (int j = 0; j < layers[i].NumNeurons; j++)
			{
				// Iterate over the inputs to the current neuron
				for (int k = 0; k < layers[i].Neurons[j].NumInputs; k++)
				{
					// Update the weight of the current input
					 
					layers[i].Neurons[j].Weights[k] += alpha * layers[i].Neurons[j].Inputs[k] * layers[i].Neurons[j].ErrorGradient;
				}
				// Update the bias of the current neuron
				layers[i].Neurons[j].Bias += (-1) * alpha * layers[i].Neurons[j].ErrorGradient;
			}
		}
	}

	public void Train(List<List<double>> inputData, List<List<double>> outputData, int numIterations)
	{
		// Validate input
		if (inputData == null || inputData.Count == 0)
		{
			throw new ArgumentException("Input data cannot be null or empty.", nameof(inputData));
		}
		if (outputData == null || outputData.Count == 0)
		{
			throw new ArgumentException("Output data cannot be null or empty.", nameof(outputData));
		}
		if (inputData.Count != outputData.Count)
		{
			throw new ArgumentException("Input data and output data must have the same length.", nameof(inputData));
		}
		if (numIterations <= 0)
		{
			throw new ArgumentOutOfRangeException(nameof(numIterations), "Number of iterations must be a positive number.");
		}

		// Train the neural network for the specified number of iterations
		for (int i = 0; i < numIterations; i++)
		{
			// Iterate over the input and output data
			for (int j = 0; j < inputData.Count; j++)
			{
				// Calculate the network's output for the current input
				this.Evaluate(inputData[j]);

				// Backpropagate the error and 
				this.Backpropagate(outputData[j]);

				//update the weights and biases
				this.UpdateWeights();
			}
		}
	}





}
