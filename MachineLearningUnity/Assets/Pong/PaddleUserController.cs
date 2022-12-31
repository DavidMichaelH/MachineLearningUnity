using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PaddleUserController : MonoBehaviour
{
    [SerializeField]
    private PaddleController paddleCntrlr;
    [SerializeField]
    private float PaddleVel = 15f;

    // Start is called before the first frame update
    void Start()
    {
        paddleCntrlr = GetComponent<PaddleController>();
    }

    // Update is called once per frame
    void Update()
    {
    
        if(Input.GetKey(KeyCode.UpArrow))
        {
            paddleCntrlr.SetPaddleVeclocity(PaddleVel);
        }
        else if (Input.GetKey(KeyCode.DownArrow))
        {
            paddleCntrlr.SetPaddleVeclocity(-PaddleVel);
        }
        else
        {
            paddleCntrlr.SetPaddleVeclocity(0f);
        }

    }
}
