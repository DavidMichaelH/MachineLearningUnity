using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using Ann;

public class ANNDrive : MonoBehaviour {

    public Model model;

    public float visibleDistance = 50;
    public int epochs = 5000;
    public float speed = 50.0F;
    public float rotationSpeed = 100.0F;

    bool trainingDone = false;
    float trainingProgress = 0;
    double sse = 0;
    double lastSSE = 1; 

    public float translation;
    public float rotation;

    public bool loadFromFile = true;

    public Transform startTrans;

    // Use this for initialization
    void Start () {
        //ann = new ANN(5,2,1,10,0.5);
        
        List<ModelLayer> layers = new List<ModelLayer>();


        layers.Add(new ModelLayer(16, Activations.Activation.Sigmoid, "layer_1"));
        layers.Add(new ModelLayer(2, Activations.Activation.Sigmoid, "output"));

        model = new Model(5, layers, 0.05, "myFirstModel");
         




        if (loadFromFile)
        {
			LoadWeightsFromFile();

            trainingDone = true;
        }
        else
        	StartCoroutine(LoadAndTrain());
	}

    /*
    void OnGUI()
    {
        GUI.Label (new Rect (25, 25, 250, 30), "SSE: " + lastSSE);
        GUI.Label (new Rect (25, 40, 250, 30), "Alpha: " + model.alpha);
        GUI.Label (new Rect (25, 55, 250, 30), "Trained: " + trainingProgress);
    }
    */

    IEnumerator LoadTrainingSet()
    {

        string path = Application.dataPath + "/SelfDrivingCar/ModelData/trainingData.txt"; //trainingDataStripped
        string line;
        if(File.Exists(path))
        {
            int lineCount = File.ReadAllLines(path).Length;
            StreamReader tdf = File.OpenText(path);
            List<double> calcOutputs = new List<double>();
            List<double> inputs = new List<double>();
            List<double> outputs = new List<double>();


            for(int i = 0; i < epochs; i++)
            { 
                //set file pointer to beginning of file
                sse = 0;
                tdf.BaseStream.Position = 0;
                string currentWeights = model.PrintWeights();
                while((line = tdf.ReadLine()) != null)  
                {  
                    string[] data = line.Split(',');
                    //if nothing to be learned ignore this line
                    float thisError = 0;
                    if(System.Convert.ToDouble(data[5]) != 0 && System.Convert.ToDouble(data[6]) != 0)
                    {
                        inputs.Clear();
                        outputs.Clear();
                        inputs.Add(System.Convert.ToDouble(data[0]));
                        inputs.Add(System.Convert.ToDouble(data[1]));
                        inputs.Add(System.Convert.ToDouble(data[2]));
                        inputs.Add(System.Convert.ToDouble(data[3]));
                        inputs.Add(System.Convert.ToDouble(data[4]));

                        double o1 = Map(0, 1, -1, 1, System.Convert.ToSingle(data[5]));
                        outputs.Add(o1);
                        double o2 = Map(0, 1, -1, 1, System.Convert.ToSingle(data[6]));
                        outputs.Add(o2);

                        model.Train(inputs, outputs, 1);
                        calcOutputs =  model.Evaluate(inputs);
    

                        thisError = ((Mathf.Pow((float)(outputs[0] - calcOutputs[0]),2) +
                            Mathf.Pow((float)(outputs[1] - calcOutputs[1]),2)))/2.0f;
                         
                    }
                    sse += thisError;
                     
                } 
                trainingProgress = (float)i/(float)epochs;
                sse /= lineCount;
                
                
                 
                //if sse isn't better then reload previous set of weights
                //and decrease alpha
                if(lastSSE < sse)
                {
                    model.LoadWeights(currentWeights);
                    model.alpha = Mathf.Clamp((float)model.alpha - 0.001f,0.01f,0.9f);
                }
                else //increase alpha
                {
                    model.alpha = Mathf.Clamp((float)model.alpha + 0.001f,0.01f,0.9f);
                	lastSSE = sse;
                }
                
                
                yield return null;
            }
                
        }
        trainingDone = true;
         
    }

    IEnumerator LoadAndTrain()
    {

        string path = Application.dataPath + "/SelfDrivingCar/ModelData/trainingData.txt"; //trainingDataUdemy
        string line;
        if (File.Exists(path))
        {
            int lineCount = File.ReadAllLines(path).Length;
            StreamReader tdf = File.OpenText(path);
            

            List<List<double>> inputData = new List<List<double>>();
            List<List<double>> outputData = new List<List<double>>();

            
                
            tdf.BaseStream.Position = 0;
                 
            while ((line = tdf.ReadLine()) != null)
            {
                string[] data = line.Split(',');


                List<double> input = new List<double>();
                List<double> output = new List<double>();
                 
                input.Add(System.Convert.ToDouble(data[0]));
                input.Add(System.Convert.ToDouble(data[1]));
                input.Add(System.Convert.ToDouble(data[2]));
                input.Add(System.Convert.ToDouble(data[3]));
                input.Add(System.Convert.ToDouble(data[4]));

                inputData.Add(input);

                double o1 = Map(0, 1, -1, 1, System.Convert.ToSingle(data[5]));
                output.Add(o1);
                double o2 = Map(0, 1, -1, 1, System.Convert.ToSingle(data[6]));
                output.Add(o2);

                outputData.Add(output);

            }

            StartCoroutine(model.Train_CR(inputData, outputData, epochs));

            yield return null;

        }
        trainingDone = true;
        SaveWeightsToFile();
    }

    void SaveWeightsToFile()
    {
        string path = Application.dataPath + "/SelfDrivingCar/ModelData/weights.txt";
        StreamWriter wf = File.CreateText(path);
        wf.WriteLine (model.PrintWeights());
        wf.Close();
    }

    void LoadWeightsFromFile()
    {
    	string path = Application.dataPath + "/SelfDrivingCar/ModelData/weights.txt";
    	StreamReader wf = File.OpenText(path);

        if(File.Exists(path))
        {
        	string line = wf.ReadLine();
            model.LoadWeights(line);
        }
    }

    float Map (float newfrom, float newto, float origfrom,float origto, float value) 
    {
    	if (value <= origfrom)
        	return newfrom;
    	else if (value >= origto)
        	return newto;
    	return (newto - newfrom) * ((value - origfrom) / (origto - origfrom)) + newfrom;
	}

    float Round(float x) 
    {   
        return (float)System.Math.Round(x, System.MidpointRounding.AwayFromZero) / 2.0f;
    }

    void Update() {
       
        if (Input.GetKeyDown(KeyCode.Space))
        {
            transform.position = startTrans.position;
            transform.rotation = startTrans.rotation;
        }

        if(!trainingDone) return;

        List<double> calcOutputs = new List<double>();
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        //raycasts
        RaycastHit hit;
        float fDist = 0, rDist = 0, lDist = 0, r45Dist = 0, l45Dist = 0;

        //forward
        if (Physics.Raycast(transform.position, this.transform.forward, out hit, visibleDistance))
        {
            fDist = 1-(hit.distance/visibleDistance); //Round
        }

        //right
        if (Physics.Raycast(transform.position, this.transform.right, out hit, visibleDistance))
        {
            rDist = 1-(hit.distance/visibleDistance);
        }

        //left
        if (Physics.Raycast(transform.position, -this.transform.right, out hit, visibleDistance))
        {
            lDist = 1-(hit.distance/visibleDistance);
        }

        //right 45
        if (Physics.Raycast(transform.position, 
                            Quaternion.AngleAxis(-45, Vector3.up) * this.transform.right, out hit, visibleDistance))
        {
            r45Dist = 1-(hit.distance/visibleDistance);
        }

        //left 45
        if (Physics.Raycast(transform.position, 
                            Quaternion.AngleAxis(45, Vector3.up) * -this.transform.right, out hit, visibleDistance))
        {
            l45Dist = 1-(hit.distance/visibleDistance);
        }

        inputs.Add(fDist);
        inputs.Add(rDist);
        inputs.Add(lDist);
        inputs.Add(r45Dist);
        inputs.Add(l45Dist);
        outputs.Add(0);
        outputs.Add(0);

         
        calcOutputs = model.Evaluate(inputs);
        //calcOutputs = model.CalcOutput(inputs,outputs);

        float translationInput = Map(-1,1,0,1,(float) calcOutputs[0]);
        float rotationInput = Map(-1,1,0,1,(float) calcOutputs[1]);
        translation = translationInput * speed * Time.deltaTime;
        rotation = rotationInput * rotationSpeed * Time.deltaTime;
        this.transform.Translate(0, 0, translation);
        this.transform.Rotate(0, rotation, 0);        

    }


    void OnGUI()
    {
        GUI.skin.label.fontSize = 20; // Set the font size to 20

        GUI.Label(new Rect(25, 25, 250, 30), "SSE: " + model.Loss);
        GUI.Label(new Rect(25, 45, 250, 30), "Alpha: " + model.alpha);
        GUI.Label(new Rect(25, 65, 250, 30), "Trained: " + model.EpochsRemaining);
    }



}
