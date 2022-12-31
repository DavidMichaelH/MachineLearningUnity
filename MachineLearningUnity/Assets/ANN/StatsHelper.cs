using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Ann;
public class StatsHelper : MonoBehaviour
{

    public ANNDrive drive;
    public ANN ann;

    // Start is called before the first frame update
    void OnEnable()
    {
        Debug.Log(drive == null);
        Debug.Log(drive.model == null);
        Debug.Log(drive.model.epochEvent == null);
        drive.model.epochEvent.AddListener(UpdateText);
    }

    public void UpdateText(EpochData epochData)
    {
        Debug.Log(epochData.loss + " for " + epochData.epoch);
    }
}
