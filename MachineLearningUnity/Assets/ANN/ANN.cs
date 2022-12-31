using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using UnityEngine.Events;

namespace Ann
{

	public struct EpochData
    {
		public double loss;
		public double alpha;
		public int epoch;
    }
	public class EpochEvent : UnityEvent<EpochData>
    {

    }
	public class ANN
	{

		public int NumInputs { get; private set; }
		public int NumOutputs;
		public int numHidden;
		public int NumLayers { get; private set; }
		public int numNPerHidden;
		public double alpha;
		public List<Layer> Layers { get; private set; }

		
		public double Loss { get; private set; }
		public double EpochsRemaining { get; private set; }

		public EpochEvent epochEvent = new EpochEvent();

		public UnityEvent trainingCompleted = new UnityEvent();

		public enum Optimizer { SGD, ADAM }


		public List<List<double>> InputData { get; set; }
		public List<List<double>> OutputData { get; set; }

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
		 
		 




		public void Optimize(List<double> desiredOutput, Optimizer optimizer = Optimizer.ADAM)
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
                 
                switch (optimizer)
                {
					case Optimizer.ADAM:
						UpdateWeightsAdam(i);
						break;
					case Optimizer.SGD:
						GradientDecent(i);
						break;
					default:
						throw new ArgumentOutOfRangeException("Invalid choice of optimizer");
				}

			}
		}

		public void UpdateWeightsAdam(int i)
		{ 
			 
			double beta1 = 0.9;
			double beta2 = 0.999;
			double learningRate = 0.05;
			double epsilon = 1e-8;
			double update = 0;
			// Update the weights using Adam optimization
			for (int j = 0; j < Layers[i].NumNeurons; j++)
			{
				for (int k = 0; k < Layers[i].Neurons[j].Weights.Count; k++)
				{
					// Calculate the Adam update for the weight
					double errorGradient = Layers[i].Neurons[j].ErrorGradient;
					double m = Layers[i].Neurons[j].AdamM[k];
					double v = Layers[i].Neurons[j].AdamV[k];

					m = beta1 * m + (1 - beta1) * errorGradient;
					v = beta2 * v + (1 - beta2) * (errorGradient * errorGradient);
					update = (learningRate * m) / (Math.Sqrt(v) + epsilon);

					// Update the weight using the Adam update
					Layers[i].Neurons[j].Weights[k] -= update;


					// Store the updated Adam variables
					Layers[i].Neurons[j].AdamM[k] = m;
					Layers[i].Neurons[j].AdamV[k] = v;
				}
				// Update the bias of the current neuron
				// Calculate the Adam update for the bias weight
				double biasErrorGradient = Layers[i].Neurons[j].ErrorGradient;
				double biasM = Layers[i].Neurons[j].AdamBiasM;
				double biasV = Layers[i].Neurons[j].AdamBiasV;
				biasM = beta1 * biasM + (1 - beta1) * biasErrorGradient;
				biasV = beta2 * biasV + (1 - beta2) * (biasErrorGradient * biasErrorGradient);
				double biasUpdate = (learningRate * biasM) / (Math.Sqrt(biasV) + epsilon);

				// Update the bias weight using the Adam update
				Layers[i].Neurons[j].Bias -= biasUpdate;

				// Store the updated Adam variables
				Layers[i].Neurons[j].AdamBiasM = biasM;
				Layers[i].Neurons[j].AdamBiasV = biasV;

			}
			
		}
		public void GradientDecent(int i)
        {
			alpha = 0.1;

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
		 

		public void ValidateTraingInputArguments(List<List<double>> inputData, List<List<double>> outputData, int numIterations)
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

		}

		public void Train(List<List<double>> inputData, List<List<double>> outputData, int numIterations)
		{
			ValidateTraingInputArguments(inputData, outputData, numIterations);



			EpochsRemaining = numIterations;
			double loss = 0;
			// Train the neural network for the specified number of iterations
			for (int i = 0; i < numIterations; i++)
			{
				EpochsRemaining--;

				Loss = RunEpoch();

				EpochData epochData = new EpochData();
				epochData.epoch = i;
				epochData.loss = loss;
				epochData.alpha = alpha;
				epochEvent.Invoke(epochData);

				Debug.Log(epochData.loss + " for " + epochData.epoch);

				loss = 0;
				
			}

			trainingCompleted.Invoke();
		}


		public IEnumerator Train_CR(List<List<double>> inputData, List<List<double>> outputData, int numIterations)
		{
			ValidateTraingInputArguments(inputData, outputData, numIterations);

			InputData = inputData;
			OutputData = outputData;

			EpochsRemaining = numIterations;
			double loss = 0;
			// Train the neural network for the specified number of iterations
			for (int i = 0; i < numIterations; i++)
			{
				EpochsRemaining--;
				Loss = RunEpoch();

				yield return null;

			}

			trainingCompleted.Invoke();
		}

		public double RunEpoch()
        {
			double loss = 0;
			for (int j = 0; j < InputData.Count; j++)
			{
				// Calculate the network's output for the current input
				List<double> predictedOutput = this.Evaluate(InputData[j]);

				loss += CalculateLoss(OutputData[j], predictedOutput);


				Optimize(OutputData[j]);

			}

			return loss;
		}

		 


		private double CalculateLoss(List<double> expectedOutput, List<double> predictedOutput)
		{
			// Calculate the mean squared error between the expected output and the actual output
			double sumSquaredError = 0;
			for (int i = 0; i < expectedOutput.Count; i++)
			{
				double error = expectedOutput[i] - predictedOutput[i];
				sumSquaredError += error * error;
			}
			return sumSquaredError / expectedOutput.Count;
		}

		public void Train(List<double> inputData, List<double> outputData, int numIterations)
		{
			List<List<double>> x = new List<List<double>>();
			x.Add(inputData);
			List<List<double>> y = new List<List<double>>();
			y.Add(outputData);

			Train(x, y, numIterations);
		}

		



	}
}