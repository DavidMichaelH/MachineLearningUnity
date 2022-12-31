using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PaddleInput : MonoBehaviour
{
    public float verticleSpeed = 5f;
    public PaddleController paddleContoller; 

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
        paddleContoller.SetPaddleVeclocity(verticleSpeed * Input.GetAxis("Horizontal"));
    }
}
