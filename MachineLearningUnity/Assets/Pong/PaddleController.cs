using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PaddleController : MonoBehaviour
{
    public GameObject paddle;
    private float veclocity;
    float paddleMinY = 8.8f;
    float paddleMaxY = 17.4f;
    float paddleMaxSpeed = 15f;


    // Start is called before the first frame update
    void Start()
    {
        paddle = gameObject;
    }

    // Update is called once per frame
    void Update()
    {
        UpdatePaddlePostion();
    }

    public void SetPaddleVeclocity(float vel)
    {
        veclocity = vel;
    }

    public Vector3 GetPaddlePosition()
    {
        return paddle.transform.position;
    }


    private void UpdatePaddlePostion()
    {
        float posy = Mathf.Clamp(transform.position.y + (veclocity * Time.deltaTime * paddleMaxSpeed), paddleMinY, paddleMaxY);
        paddle.transform.position = new Vector3(paddle.transform.position.x, posy, paddle.transform.position.z);
    }
}
