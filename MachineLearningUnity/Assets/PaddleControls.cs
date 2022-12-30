using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PaddleControls : MonoBehaviour {
	
	// Update is called once per frame
	void Update () {
		if(Input.GetKey("up"))
			this.transform.Translate(0,0.3f,0);
		else if(Input.GetKey("down"))
			this.transform.Translate(0,-0.3f,0);
	}
}
