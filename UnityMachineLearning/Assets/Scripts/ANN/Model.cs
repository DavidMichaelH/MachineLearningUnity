using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace Ann
{
    public class Model : ANN 
    {

        public string modelName;
        public float alpha;
        public ANN Ann { get; private set; } 

        public Model(int numInputs, List<ModelLayer> modelLayers, string modelName = "")
        {
            this.modelName = modelName;


            List<Layer> layers = new List<Layer>();

            int numNeurons = modelLayers[0].NumNeurons;
            int numNeuronInputs = numInputs;
            string name = modelLayers[0].Name;
            Activations.Activation activation = modelLayers[0].Activation;

            layers.Add(new Layer(numNeurons, numNeuronInputs, name, activation));

            for (int layerIndex = 1; layerIndex < modelLayers.Count; layerIndex++)
            {
                numNeurons = modelLayers[layerIndex].NumNeurons;
                numNeuronInputs = modelLayers[layerIndex-1].NumNeurons;
                name = modelLayers[layerIndex].Name;
                activation = modelLayers[layerIndex].Activation;

                layers.Add(new Layer(numNeurons, numNeuronInputs, name, activation));

            }



            Initialize(layers);
 

        }
    }

    
    public class ModelLayer
    {
        public int NumNeurons { get; private set; }
        public Activations.Activation Activation { get; private set; }
        public string Name { get; private set; }
        public ModelLayer(int numNeurons, Activations.Activation activation, string name = "") : base(name)
        {
            NumNeurons = numNeurons;
            Activation = activation;
            Name = name;
        }

    }
 

}