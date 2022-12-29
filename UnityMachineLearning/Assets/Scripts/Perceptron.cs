using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;



[System.Serializable]
public class TrainingSet
{
    public double[] input;
    public double output;
}

public enum ActivationFunctionType { ReLU, Tanh, Step, Sigmoid }

public class Perceptron : MonoBehaviour
{

    public TrainingSet[] trainingSet;

    double[] weights;
    double bias = 0;
    double totalError = 0;

    public double learningRate;

    public int numEpochs;

    public ActivationFunctionType activationFunction;


    void InitializeWeights()
    {

        weights = new double[trainingSet[0].input.Length];

        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = UnityEngine.Random.Range(-1f, 1f);
        }

        bias = UnityEngine.Random.Range(-1f, 1f);

    }

    void Train(int epochs)
    {
        InitializeWeights();

        for (int i = 0; i < epochs; i ++)
        {
            totalError = 0;

            for(int j = 0; j < trainingSet.Length; j++)
            {
                UpdateWeights(j);
                PrintWeights();
            }
            Debug.Log("Total Error " + totalError.ToString());
        }
    }
     
    public void UpdateWeights(int j)
    {
        // Validate input
        if (j < 0 || j >= trainingSet.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(j), "j must be a valid index for the trainingSet array.");
        }

        // Update the weights of the perceptron based on the error of the j-th training example
        double error = trainingSet[j].output - ComputePerceptronOutput(trainingSet[j].input);
        totalError += Mathf.Abs((float)error);
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = weights[i] + learningRate * error * trainingSet[j].input[i];
        }
        bias += learningRate * error;
    }

        public void PrintWeights()
    {
        string outputString = "";

        for(int i = 0; i < weights.Length; i++)
        {
            outputString += "w_" + i.ToString() + " = " + weights[i].ToString() + " ";
        }

        outputString += "bias = " + bias.ToString();

        Debug.Log(outputString);
    }



    double DotProduct(double[] x, double[] y)
    {
        double result = 0;
        for (int i = 0; i < x.Length; i++)
        {
            result += x[i] * y[i];
        }
        return result;
    }
 

    double ComputePerceptronOutput(double[] input)
    {
        // Validate input
        if (input == null || input.Length == 0)
        {
            throw new ArgumentException("Input cannot be null or empty.", nameof(input));
        }

        // Make sure input and weights have the same length
        if (input.Length != weights.Length)
        {
            throw new ArgumentException("Input and weights must have the same length.", nameof(input));
        }

        // Compute and return the output of the perceptron
        double perceptronOutput = 0;
        perceptronOutput += DotProduct(weights, input);
        perceptronOutput += bias;
        return ActivationFunction(perceptronOutput);
    }


    double ActivationFunction(double input)
    {
        // Validate input
        if (double.IsNaN(input) || double.IsInfinity(input))
        {
            throw new ArgumentException("Input must be a finite number.", nameof(input));
        }

        // Apply the appropriate activation function based on the value of activationFunction
        switch (activationFunction)
        {
            case ActivationFunctionType.ReLU:
                return input > 0 ? input : 0f;

            case ActivationFunctionType.Tanh:
                return Math.Tanh(input);

            case ActivationFunctionType.Step:
                return (input > 0 ? 1f : 0f);

            case ActivationFunctionType.Sigmoid:
                return 1.0 / (1.0 + Math.Exp(-input));

            // Add a default case to handle unexpected values of activationFunction
            default:
                throw new InvalidOperationException("Unrecognized activation function.");
        }
    }


    // Start is called before the first frame update
    void Start()
    {
        InitializeWeights();
        Train(numEpochs);        
    }

    private void Update()
    {
        if (Input.GetKeyDown("space"))
        {
            InitializeWeights();
        }
    }

 
    void LoadWeights()
    {
        string filePath = Application.dataPath + "/weights.txt";
        // Check if the file exists
        if (File.Exists(filePath))
        {
            // Create a StreamReader object to read from the file
            StreamReader sr;
            try
            {
                // Use File.OpenText to open the file and obtain a StreamReader object for reading from it
                sr = File.OpenText(filePath);
            }
            catch (Exception ex)
            {
                // If an exception is thrown when opening the file, display an error message and return
                Debug.LogError("Error opening file: " + ex.Message);
                return;
            }

            // Read the first line of the file
            string line = sr.ReadLine();
            if (string.IsNullOrEmpty(line))
            {
                // If the line is empty or null, display an error message and return
                Debug.LogError("Error: file is empty");
                return;
            }

            // Split the line into an array of strings using the comma separator
            string[] w = line.Split(',');
            if (w.Length != 3)
            {
                // If the array does not have 3 elements, display an error message and return
                Debug.LogError("Error: file does not contain the expected number of values");
                return;
            }

            // Convert the strings to doubles and assign the values to the weights and bias
            for (int i = 0; i < w.Length-1; i++)
            {
                weights[i] = System.Convert.ToDouble(w[i]);
            }
            bias = System.Convert.ToDouble(w[w.Length-1]);

            Debug.Log("loading");

            // Close the StreamReader object
            sr.Close();
        }
        else
        {
            // If the file does not exist, display an error message
            Debug.LogError("Error: file does not exist");
        }
    }


    void SaveWeights()
    {
        string filePath = Application.dataPath + "/weights.txt";

        // Create a StreamWriter object to write to the file
        StreamWriter sr;
        try
        {
            // Use File.CreateText to create the file and obtain a StreamWriter object for writing to it
            sr = File.CreateText(filePath);
        }
        catch (Exception ex)
        {
            // If an exception is thrown when creating the file, display an error message and return
            Debug.LogError("Error creating file: " + ex.Message);
            return;
        }

        // Validate the input data
        if (weights == null)
        {
            Debug.LogError("Error: weights array is null");
            return;
        }
        if (double.IsNaN(bias) || double.IsInfinity(bias))
        {
            Debug.LogError("Error: bias is not a valid number");
            return;
        }

        // Build the output string
        string output = "";
        for (int i = 0; i < weights.Length; i++)
        {
            output += weights[i] + ",";
        }
        output += bias;

        // Write the output string to the file
        try
        {
            sr.WriteLine(output);
        }
        catch (Exception ex)
        {
            // If an exception is thrown when writing to the file, display an error message
            Debug.LogError("Error writing to file: " + ex.Message);
        }

        // Close the StreamWriter object
        sr.Close();
    }




}
