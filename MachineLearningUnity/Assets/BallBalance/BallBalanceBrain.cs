using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using Ann;

 


public class BallBalanceBrain : MonoBehaviour {

	public GameObject ball;							//object to monitor
	
	//Model model;

	float reward = 0.0f;							//reward to associate with actions
	//List<Replay> replayMemory = new List<Replay>();	//memory - list of past actions and rewards
	int mCapacity = 10000;							//memory capacity
	
	float discount = 0.99f;							//how much future states affect rewards
	float exploreRate = 100.0f;						//chance of picking random action
	float maxExploreRate = 100.0f;					//max chance value
    float minExploreRate = 0.01f;					//min chance value
    float exploreDecay = 0.0001f;					//chance decay amount for each update

	Vector3 ballStartPos;							//record start position of object
	int failCount = 0;								//count when the ball is dropped
	float tiltSpeed = 0.5f;						    //max angle to apply to tilting each update
													//make sure this is large enough so that the q value
													//multiplied by it is enough to recover balance
													//when the ball gets a good speed up
	float timer = 0;								//timer to keep track of balancing
	float maxBalanceTime = 0;                       //record time ball is kept balanced	


	public QLearning qLearning;

	// Use this for initialization
	void Start () {

		qLearning = new QLearning();
		//ann = new ANN(3,2,1,6,0.2f);
		/*
		List<ModelLayer> layers = new List<ModelLayer>();

		layers.Add(new ModelLayer(6, Activations.Activation.Sigmoid, "layer_1"));
		layers.Add(new ModelLayer(2, Activations.Activation.TanH, "output"));

		model = new Model(3, layers, 0.1, "myFirstModel");
		*/

		ballStartPos = ball.transform.position;
		Time.timeScale = 5.0f;		
	}

	GUIStyle guiStyle = new GUIStyle();
	void OnGUI()
	{
		guiStyle.fontSize = 25;
		guiStyle.normal.textColor = Color.white;
		GUI.BeginGroup (new Rect (10, 10, 600, 150));
		GUI.Box (new Rect (0,0,140,140), "Stats", guiStyle);
		GUI.Label(new Rect (10,25,500,30), "Fails: " + failCount, guiStyle);
		GUI.Label(new Rect (10,50,500,30), "Decay Rate: " + exploreRate, guiStyle);
		GUI.Label(new Rect (10,75,500,30), "Last Best Balance: " + maxBalanceTime, guiStyle);
		GUI.Label(new Rect (10,100,500,30), "This Balance: " + timer, guiStyle);
		GUI.EndGroup ();
	}
	
	// Update is called once per frame
	void Update () {
		if(Input.GetKeyDown("space"))
			ResetBall();	
	}

	void FixedUpdate () {
		timer += Time.deltaTime;
		List<double> states = new List<double>();
		//List<double> qs = new List<double>();


		states.Add(this.transform.rotation.x); //BallBrain
		states.Add(ball.transform.position.z); //BallBrain
		states.Add(ball.GetComponent<Rigidbody>().angularVelocity.x); //BallBrain

		if (ball.GetComponent<BallState>().dropped)
			reward = -1.0f;
		else
			reward = 0.1f;

		List<double> qs = qLearning.SubmitState(states,reward);

		double maxQ = qs.Max();
		int maxQIndex = qs.ToList().IndexOf(maxQ);
		 

		if(maxQIndex == 0)
			this.transform.Rotate(Vector3.right,tiltSpeed * (float)qs[maxQIndex]);
		else if (maxQIndex == 1)
			this.transform.Rotate(Vector3.right,-tiltSpeed * (float)qs[maxQIndex]);
		
		   

		if(ball.GetComponent<BallState>().dropped) 
		{

			qLearning.TrainFromReplay();

			qLearning.ClearMemory();

			if (timer > maxBalanceTime)
			{
			 	maxBalanceTime = timer;
			} 

			timer = 0;

			ball.GetComponent<BallState>().dropped = false;
			this.transform.rotation = Quaternion.identity;
			ResetBall();
			
			failCount++;
		}	
	}

	void ResetBall()
	{
		ball.transform.position = ballStartPos;
		ball.GetComponent<Rigidbody>().velocity = new Vector3(0,0,0);
		ball.GetComponent<Rigidbody>().angularVelocity = new Vector3(0,0,0);
	}

	List<double> SoftMax(List<double> values) 
    {
      double max = values.Max();

      float scale = 0.0f;
      for (int i = 0; i < values.Count; ++i)
        scale += Mathf.Exp((float)(values[i] - max));

      List<double> result = new List<double>();
      for (int i = 0; i < values.Count; ++i)
        result.Add(Mathf.Exp((float)(values[i] - max)) / scale);

      return result; 
    }
}
