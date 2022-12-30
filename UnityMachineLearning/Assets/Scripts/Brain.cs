using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{

	ANN ann;
	double sumSquareError = 0;

	void Start()
	{
		ann = new ANN(2, 1, 1, 2, 0.8);

		List<double> result;


		List<List<double>> inputData = new List<List<double>>();
		List<List<double>> outputData = new List<List<double>>();
		inputData.Add(new List<double>() { 1, 1 });
		outputData.Add(new List<double>() { 1 });

		inputData.Add(new List<double>() { 1, 0 });
		outputData.Add(new List<double>() { 1 });

		inputData.Add(new List<double>() { 0, 1 });
		outputData.Add(new List<double>() { 1 });

		inputData.Add(new List<double>() { 0, 0 });
		outputData.Add(new List<double>() { 0 });

		for (int trials = 0; trials < 100; trials++)
        {
			ann.Train(inputData, outputData, 3);

			double sumSquareError = 0;
			int itr = 0;
			foreach (List<double> input in inputData)
			{
				List<double> predictOut = ann.Evaluate(input);
				 sumSquareError += ComputeError(predictOut, outputData[itr]);
				itr += 1;
			}
			Debug.Log("SSE: " + sumSquareError);
		}

		
		 
	}

	double ComputeError(List<double> x , List<double> y)
    {
		sumSquareError = 0;
		for(int i = 0; i < x.Count; i++)
        {
			sumSquareError += Mathf.Pow((float)(x[i] - y[i]), 2);
		}

		return sumSquareError;


	}
	 
}