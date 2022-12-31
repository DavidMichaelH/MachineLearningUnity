using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using Ann;


public class Replay
{
	public List<double> states;
	public double reward;

	public Replay(List<double> states, double r)
	{
		this.states = new List<double>(states);
		this.reward = r;
	}
}

public class QLearning
{

	public GameObject ball;                         //object to monitor

	Model model;

	float reward = 0.0f;                            //reward to associate with actions
	List<Replay> replayMemory; //memory - list of past actions and rewards
	int mCapacity = 10000;                          //memory capacity

	float discount = 0.99f;                         //how much future states affect rewards
	float exploreRate = 100.0f;                     //chance of picking random action
	float maxExploreRate = 100.0f;                  //max chance value
	float minExploreRate = 0.01f;                   //min chance value
	float exploreDecay = 0.0001f;                   //chance decay amount for each update

	Vector3 ballStartPos;                           //record start position of object
	int failCount = 0;                              //count when the ball is dropped
	float tiltSpeed = 0.5f;                         //max angle to apply to tilting each update
													//make sure this is large enough so that the q value
													//multiplied by it is enough to recover balance
													//when the ball gets a good speed up
	float timer = 0;                                //timer to keep track of balancing
	float maxBalanceTime = 0;                       //record time ball is kept balanced	
													// Use this for initialization


	public QLearning()
    {
		List<ModelLayer> layers = new List<ModelLayer>();

		layers.Add(new ModelLayer(6, Activations.Activation.Sigmoid, "layer_1"));
		layers.Add(new ModelLayer(2, Activations.Activation.TanH, "output"));

		model = new Model(3, layers, 0.1, "myFirstModel");

		replayMemory = new List<Replay>();
	}

	 
	public List<double> SubmitState(List<double> state,double reward)
    {
		Replay lastMemory = new Replay(state,reward);
		List<double> qs = SoftMax(model.Evaluate(state));

		if (replayMemory.Count > mCapacity)
			replayMemory.RemoveAt(0);

		replayMemory.Add(lastMemory);
 
		return qs; 
	}

	public int argMax(List<double> qs)
    {
		double maxQ = qs.Max();
		int maxQIndex = qs.ToList().IndexOf(maxQ);
		return maxQIndex;
	}
	 

	public void TrainFromReplay()
    {
		double maxQ;
		for (int i = replayMemory.Count - 1; i >= 0; i--)
		{
			List<double> toutputsOld = new List<double>();
			List<double> toutputsNew = new List<double>();
			

            try 
			{ 
				toutputsOld = SoftMax(model.Evaluate(replayMemory[i].states)); 
			}
            catch
            {
				Debug.Log("replayMemory[i] = " + replayMemory[i].ToString());
				Debug.Log("replayMemory[i] = " + replayMemory[i].states.ToString());
			}
			 

			double maxQOld = toutputsOld.Max();
			int action = toutputsOld.ToList().IndexOf(maxQOld);

			double feedback;
			if (i == replayMemory.Count - 1 || replayMemory[i].reward == -1)
            {
				feedback = replayMemory[i].reward;
			}
			else
			{
				toutputsNew = SoftMax(model.Evaluate(replayMemory[i + 1].states));
				maxQ = toutputsNew.Max();
				feedback = (replayMemory[i].reward + discount * maxQ);
			}

			toutputsOld[action] = feedback;
			model.Train(replayMemory[i].states, toutputsOld, 1, ANN.Optimizer.SGD);
		}
	}
 

	public void ClearMemory()
    {
		replayMemory.Clear();
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