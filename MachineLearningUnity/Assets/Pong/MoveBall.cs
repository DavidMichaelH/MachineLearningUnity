using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System;
public class MoveBall : MonoBehaviour {

	Vector3 ballStartPosition;
	Rigidbody2D rb;
	float speed = 600;
	public AudioSource blip;
	public AudioSource blop;
	public Text playerScore;
	public Text agentScore;

	// Use this for initialization
	void Start () {
		rb = this.GetComponent<Rigidbody2D>();
		ballStartPosition = this.transform.position;
		ResetBall();
	}



	void OnCollisionEnter2D(Collision2D col)
	{
		if (col.gameObject.tag == "backwalll")
		{
		 
			Debug.Log("Score = " + playerScore.text);
			playerScore.text = Int32.Parse(playerScore.text) + 1 + "";
			 //agentScore.text = Int32.Parse(agentScore.text) + 1 + "";
			blop.Play();

		}
		else if (col.gameObject.tag == "backwallr")
		{
			agentScore.text = Int32.Parse(agentScore.text) + 1 + "";
			blop.Play();

		}
		else
		{
			blip.Play();
		}
	}


	public void ResetBall()
	{
		this.transform.position = ballStartPosition;
		rb.velocity = Vector3.zero;
		Vector3 dir = new Vector3(UnityEngine.Random.Range(-100,300), UnityEngine.Random.Range(-100,100),0).normalized;
		rb.AddForce(dir*speed);
	}
	
	// Update is called once per frame
	void Update () {
		if(Input.GetKeyDown("space"))
		{
			ResetBall();
		}
	}
}
