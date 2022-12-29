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
 

	public ANN(int nI, int nO, int nH, int nPH, double a)
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

	 

	public List<double> CalcOutput(List<double> inputValues, List<double> desiredOutput)
	{
		List<double> inputs = new List<double>();
		List<double> outputValues = new List<double>();
		int currentInput = 0;

		if (inputValues.Count != numInputs)
		{
			Debug.Log("ERROR: Number of Inputs must be " + numInputs);
			return outputValues;
		} 

		inputs = new List<double>(inputValues);
		for (int i = 0; i < numHidden + 1; i++)
		{
			if (i > 0)
			{
				inputs = new List<double>(outputValues);
			}
			outputValues.Clear();

			for (int j = 0; j < layers[i].NumNeurons; j++)
			{
				double N = 0;
				layers[i].Neurons[j].Inputs.Clear();

				for (int k = 0; k < layers[i].Neurons[j].NumInputs; k++)
				{
					layers[i].Neurons[j].Inputs.Add(inputs[currentInput]);
					N += layers[i].Neurons[j].Weights[k] * inputs[currentInput];
					currentInput++;
				}

				N -= layers[i].Neurons[j].Bias;
				layers[i].Neurons[j].Output = layers[i].Neurons[j].ActivationFunction(N);
				outputValues.Add(layers[i].Neurons[j].Output);
				currentInput = 0;
			}
		}
		return outputValues;
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
		for (int i = 0; i < numHidden+1; i++)
		{
			List<double> currentOutput = new List<double>();
			for (int j = 0; j < layers[i].NumNeurons; j++)
			{
				currentOutput.Add(layers[i].Neurons[j].ComputePerceptronOutput(currentInput));
			}
			currentInput = currentOutput;
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
						(layers[i].Neurons[j].Output - desiredOutput[j]) *
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
		for (int i = numHidden; i >= 0; i--)
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
				List<double> output = this.Evaluate(inputData[j]);

				// Backpropagate the error and update the weights and biases
				this.Backpropagate(outputData[j]);
				this.UpdateWeights();
			}
		}
	}





}
