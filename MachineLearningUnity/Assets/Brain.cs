using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Ann;

public class Brain : MonoBehaviour {

	public GameObject paddle;
	public GameObject ball;
	public bool human = false;
	public string backwallTag = "backwalll"; //This should be the agents side
	//public Text score;
	Rigidbody2D brb;
	float yvel;
	float paddleMaxSpeed = 15;
	public float numSaved = 0;
	public float numMissed = 0;
	public bool godMode = false;  

	Model model;

	void Start () {
		//ann = new ANN(6, 1, 1, 4, 0.05);

		List<ModelLayer> layers = new List<ModelLayer>();

		layers.Add(new ModelLayer(128, Activations.Activation.Sigmoid, "layer_1"));
		layers.Add(new ModelLayer(1, Activations.Activation.TanH, "output"));

		model = new Model(6, layers,0.05, "myFirstModel");


		brb = ball.GetComponent<Rigidbody2D>();
	}


	List<double> Run(double bx, double by, double bvx, double bvy, double px, double py, double pv, bool train)
	{
		List<double> inputs = new List<double>();
		List<double> outputs = new List<double>();
		inputs.Add(bx);
		inputs.Add(by);
		inputs.Add(bvx);
		inputs.Add(bvy);
		inputs.Add(px);
		inputs.Add(py);
		outputs.Add(pv);

		if (train)
        {
			List<List<double>> x = new List<List<double>>();
			x.Add(inputs);
			List<List<double>> y = new List<List<double>>();
			y.Add(outputs);
			model.Train(x,y, 10);

			return model.Evaluate(inputs);
		}
			
		else
			return (model.Evaluate(inputs));
	}
	
	// Update is called once per frame
	void Update () {
		if(!human)
		{ 
			float posy = Mathf.Clamp(paddle.transform.position.y+(yvel*Time.deltaTime*paddleMaxSpeed),8.8f,17.4f);
			paddle.transform.position = new Vector3(paddle.transform.position.x, posy, paddle.transform.position.z);
			List<double> output = new List<double>();
			int layerMask = 1 << 9;
			RaycastHit2D hit = Physics2D.Raycast(ball.transform.position, brb.velocity, 1000, layerMask);
	        
	        if (hit.collider != null) 
	        {
	        	if(hit.collider.gameObject.tag == "tops") //reflect off top
	        	{
					Vector3 reflection = Vector3.Reflect(brb.velocity,hit.normal);
	        		hit = Physics2D.Raycast(hit.point, reflection, 1000, layerMask);
	        	}
	        	if(hit.collider != null && hit.collider.gameObject.tag == backwallTag)
	        	{
			        float dy = (hit.point.y - paddle.transform.position.y);

					output = Run(ball.transform.position.x, 
									ball.transform.position.y, 
									brb.velocity.x, brb.velocity.y, 
									paddle.transform.position.x,
									paddle.transform.position.y, 
									dy,true);
				
					yvel = (float) output[0];

                    if (godMode)
                    {
						yvel = dy;
					}

				}
	        }
	        else
	        	yvel = 0;
	    }
        //score.text = numMissed + "";
	}
}
