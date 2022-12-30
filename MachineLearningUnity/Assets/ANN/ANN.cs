using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

namespace Ann
{
	public class ANN
	{

		public int NumInputs { get; private set; }
		public int NumOutputs;
		public int numHidden;
		public int NumLayers { get; private set; }
		public int numNPerHidden;
		public double alpha;
		public List<Layer> Layers { get; private set; }

		public ANN()
        {

        }

		public void Initialize(List<Layer> layers)
		{
			this.Layers = new List<Layer>(layers);
			NumInputs = layers[0].NumInputs;
			NumOutputs = layers[layers.Count - 1].NumNeurons;
			NumLayers = layers.Count;
		}



		/// <summary>
		/// Deprecated
		/// </summary>
		/// <param name="nI"></param>
		/// <param name="nO"></param>
		/// <param name="nH"></param>
		/// <param name="nPH"></param>
		/// <param name="a"></param>
		public ANN(int nI, int nO, int nH, int nPH, double a)
		{
			Layers = new List<Layer>();

			NumInputs = nI;
			NumOutputs = nO;
			numHidden = nH;
			numNPerHidden = nPH;
			alpha = a;

			if (numHidden > 0)
			{
				Layers.Add(new Layer(numNPerHidden, NumInputs));

				for (int i = 0; i < numHidden - 1; i++)
				{
					Layers.Add(new Layer(numNPerHidden, numNPerHidden));
				}

				Layers.Add(new Layer(NumOutputs, numNPerHidden));
			}
			else
			{
				Layers.Add(new Layer(NumOutputs, NumInputs));
			}
		}



		public string PrintWeights()
		{
			string weightStr = "";
			foreach (Layer l in Layers)
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
			foreach (Layer l in Layers)
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
			if (input == null)
			{
				throw new ArgumentException("Input must be a non-null list.", nameof(input));

			}

			if (input == null || input.Count != NumInputs)
			{
				throw new ArgumentException("Input of size " + input.Count + " be the same length as the number of inputs of size " + NumInputs);

			}

			// Propagate the input through the neural network
			List<double> currentInput = input;
			List<double> currentOutput = new List<double>();
			for (int i = 0; i < NumLayers; i++)
			{

				currentOutput.Clear();

				for (int j = 0; j < Layers[i].NumNeurons; j++)
				{

					currentOutput.Add(Layers[i].Neurons[j].ComputePerceptronOutput(currentInput));
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
				if (i == NumLayers - 1)
				{
					for (int j = 0; j < Layers[i].NumNeurons; j++)
					{
						Layers[i].Neurons[j].ErrorGradient =
							(desiredOutput[j] - Layers[i].Neurons[j].Output) *
							Layers[i].Neurons[j].ActivationFunctionDerivative(Layers[i].Neurons[j].Output);
					}
				}
				// Calculate the error gradients for the hidden layers
				else
				{
					for (int j = 0; j < Layers[i].NumNeurons; j++)
					{
						double errorGradientSum = 0;
						for (int k = 0; k < Layers[i + 1].NumNeurons; k++)
						{
							errorGradientSum += Layers[i + 1].Neurons[k].ErrorGradient * Layers[i + 1].Neurons[k].Weights[j];
						}
						Layers[i].Neurons[j].ErrorGradient =
							Layers[i].Neurons[j].ActivationFunctionDerivative(Layers[i].Neurons[j].Output) * errorGradientSum;
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
				for (int j = 0; j < Layers[i].NumNeurons; j++)
				{
					// Iterate over the inputs to the current neuron
					for (int k = 0; k < Layers[i].Neurons[j].NumInputs; k++)
					{
						// Update the weight of the current input

						Layers[i].Neurons[j].Weights[k] += alpha * Layers[i].Neurons[j].Inputs[k] * Layers[i].Neurons[j].ErrorGradient;
					}
					// Update the bias of the current neuron
					Layers[i].Neurons[j].Bias += (-1) * alpha * Layers[i].Neurons[j].ErrorGradient;
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
}

