using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ServeBall : MonoBehaviour {

	public GameObject ball;

	public bool backWall = false;
	public Brain b;

	void OnCollisionEnter2D(Collision2D col)
	{
		if(col.gameObject.tag == "ball" && backWall)
		{
			b.numMissed += 1;
			ball.GetComponent<MoveBall>().ResetBall();
		}
	}

	// Use this for initialization
	void Start () {

	}
	
	// Update is called once per frame
	void Update () {

	}
}
